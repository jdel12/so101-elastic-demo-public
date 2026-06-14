"""
Inference Telemetry Collector — Model Decision Observatory.

Captures per-step telemetry from VLA inference and indexes to Elasticsearch.
Designed for high-frequency (20-30 FPS) operation without blocking the
inference loop.

Data flow:
  inference node → TelemetryCollector.record_step() (non-blocking)
    → background bulk buffer → ES _bulk API every FLUSH_INTERVAL or FLUSH_SIZE
  chunk boundary frames → filesystem + async CLIP embed → ES frame index

At trial end, computes a summary document with failure classification,
action statistics, and training context for cross-trial analysis.

Indices:
  robot-inference-telemetry  — per-step actions, joints, timing, z-scores
  robot-inference-frames     — camera frames at chunk boundaries with CLIP vectors
"""

import base64
import json
import os
import ssl
import threading
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from forward_kinematics import forward_kinematics

ES_URL = os.environ.get('ES_URL', 'https://elasticsearch-es-http.default.svc:9200')
ES_USER = os.environ.get('ES_USER', 'elastic')
ES_PASSWORD = os.environ.get('ES_PASSWORD', '')
JINA_CLIP_URL = os.environ.get('JINA_CLIP_URL', 'http://localhost:8900')

IDX_TELEMETRY = 'robot-inference-telemetry'
IDX_FRAMES = 'robot-inference-frames'
IDX_TRAINING_EMBEDDINGS = 'robot-episode-embeddings'

FRAME_STORE_DIR = os.environ.get(
    'FRAME_STORE_DIR',
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'inference-frames'),
)

FLUSH_SIZE = 100
FLUSH_INTERVAL = 5.0
MAX_BUFFER_SIZE = 500

CAMERA_ROLES = {'cam0': 'wrist', 'cam1': 'overhead'}


def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _es_auth():
    return base64.b64encode(f'{ES_USER}:{ES_PASSWORD}'.encode()).decode()


def _sanitize_for_json(obj):
    """Convert numpy/torch scalars to Python natives for json.dumps."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


def _es_bulk(docs):
    if not docs:
        return
    try:
        lines = []
        for index, doc in docs:
            lines.append(json.dumps({'index': {'_index': index}}))
            lines.append(json.dumps(_sanitize_for_json(doc)))
        body = '\n'.join(lines) + '\n'
        req = Request(f'{ES_URL}/_bulk', data=body.encode(), method='POST')
        req.add_header('Content-Type', 'application/x-ndjson')
        req.add_header('Authorization', f'Basic {_es_auth()}')
        resp = urlopen(req, context=_ssl_ctx(), timeout=10)
        result = json.loads(resp.read())
        if result.get('errors'):
            err_items = [
                i for i in result['items']
                if i.get('index', {}).get('error')
            ]
            if err_items:
                print(f'[TELEMETRY] bulk errors: {len(err_items)}/{len(docs)}')
                print(f'[TELEMETRY] first error: {err_items[0]}')
    except Exception as e:
        print(f'[TELEMETRY] bulk write failed: {e}')


def _embed_clip(jpeg_bytes):
    try:
        b64 = base64.b64encode(jpeg_bytes).decode()
        payload = json.dumps({
            'inputs': [f'data:image/jpeg;base64,{b64}'],
            'type': 'image',
        }).encode()
        req = Request(f'{JINA_CLIP_URL}/embed', data=payload, method='POST')
        req.add_header('Content-Type', 'application/json')
        resp = urlopen(req, context=_ssl_ctx(), timeout=10)
        data = json.loads(resp.read())
        if 'data' in data and data['data']:
            return data['data'][0].get('embedding')
    except Exception as e:
        print(f'[TELEMETRY] CLIP embed failed: {e}')
    return None


def _nearest_training_distance(embedding):
    if not embedding:
        return None
    try:
        query = {
            'knn': {
                'field': 'clip_embedding',
                'query_vector': embedding,
                'k': 1,
                'num_candidates': 50,
            },
            '_source': False,
        }
        body = json.dumps(query).encode()
        req = Request(
            f'{ES_URL}/{IDX_TRAINING_EMBEDDINGS}/_search',
            data=body, method='POST',
        )
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Basic {_es_auth()}')
        resp = urlopen(req, context=_ssl_ctx(), timeout=5)
        data = json.loads(resp.read())
        hits = data.get('hits', {}).get('hits', [])
        if hits:
            return 1.0 - hits[0].get('_score', 0.0)
        return None
    except Exception:
        return None


IDX_SETUP = 'robot-inference-telemetry'


def _read_checkpoint_config(checkpoint_dir):
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


V1_REF = {
    'z_score_std': [1.06, 0.41, 0.33, 0.50, 0.24, 1.20],
    'z_score_magnitude_mean': 0.83,
    'action_variance_mean': 11.6,
    'action_range': [44.85, 86.57, 58.67, 27.16, 4.91, 29.58],
    'gripper_transitions': 3,
    'max_tracking_error': 12.3,
}


def _classify_failure(stats):
    """Quantitative diagnosis against V1 reference profile."""
    if stats['total_steps'] < 10:
        return 'aborted'

    zscore_mag = stats.get('z_score_magnitude_mean', 0)
    action_var = stats.get('action_variance_mean', 0)
    gripper_tx = stats.get('gripper_transitions', 0)
    j5_zscore_std = stats.get('z_score_std_j5', 0)
    pipeline_dist = stats.get('pipeline_distortion_mean', 0)
    max_te = stats.get('max_tracking_error', 0)

    diagnosis = {}
    diagnosis['vs_v1_zscore_ratio'] = round(zscore_mag / V1_REF['z_score_magnitude_mean'], 2) if V1_REF['z_score_magnitude_mean'] else 0
    diagnosis['vs_v1_variance_ratio'] = round(action_var / V1_REF['action_variance_mean'], 2) if V1_REF['action_variance_mean'] else 0
    diagnosis['vs_v1_gripper_ratio'] = round(gripper_tx / max(V1_REF['gripper_transitions'], 1), 1)
    stats['diagnosis'] = diagnosis

    if zscore_mag < 0.3:
        return 'full_mode_collapse'
    if j5_zscore_std < 0.1 and action_var > 3.0:
        return 'gripper_mode_collapse'
    if action_var < 3.0 and pipeline_dist > action_var:
        return 'pipeline_overdamped'
    if action_var < 3.0:
        return 'mode_collapse'

    range_divergence = 0
    max_range_ratio = 0
    for j in range(6):
        model_range = stats.get(f'action_range_j{j}', 0)
        ref_range = V1_REF['action_range'][j]
        if ref_range > 0:
            ratio = model_range / ref_range
            max_range_ratio = max(max_range_ratio, ratio, 1.0 / ratio if ratio > 0 else 99)
            if ratio < 0.3 or ratio > 3.0:
                range_divergence += 1
    diagnosis['max_range_ratio'] = round(max_range_ratio, 2)
    if range_divergence >= 2 or max_range_ratio > 5.0:
        return 'wrong_trajectory_shape'

    if gripper_tx > V1_REF['gripper_transitions'] * 10:
        return 'noisy_output'
    if gripper_tx == 0 and action_var > 5.0:
        return 'gripper_frozen'
    if max_te > V1_REF['max_tracking_error'] * 2.5:
        return 'servo_tracking_failure'

    if action_var > 8.0 and gripper_tx > 0 and max_te < V1_REF['max_tracking_error'] * 2:
        return 'healthy'

    return 'unknown'


class TelemetryCollector:

    def __init__(self, config):
        self.config = config
        self._buffer = []
        self._buffer_lock = threading.Lock()
        self._frame_queue = []
        self._frame_lock = threading.Lock()
        self._running = True
        self._trial_id = None
        self._run_id = config.get('run_id', str(uuid.uuid4())[:8])

        self._trial_steps = []

        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True,
        )
        self._flush_thread.start()
        self._frame_thread = threading.Thread(
            target=self._frame_loop, daemon=True,
        )
        self._frame_thread.start()
        Path(FRAME_STORE_DIR).mkdir(parents=True, exist_ok=True)
        print(f'[TELEMETRY] collector started, flush every {FLUSH_INTERVAL}s or {FLUSH_SIZE} docs')

    def emit_model_setup(self, policy_config=None):
        """Emit a model_setup doc to ES on model load. Catches config mismatches
        that are invisible at runtime (empty_cameras, action dim, broken symlinks)."""
        checkpoint_dir = self.config.get('checkpoint_dir', '')
        ckpt_config = _read_checkpoint_config(checkpoint_dir)
        policy_type = self.config.get('policy_type', 'unknown')

        checks = {}

        num_cameras_deployed = len(CAMERA_ROLES)
        num_image_views = ckpt_config.get('num_image_views', None)
        empty_cameras = ckpt_config.get('empty_cameras', None)
        if num_image_views is not None:
            effective = num_image_views - (empty_cameras or 0)
            checks['camera_count'] = {
                'num_image_views': num_image_views,
                'empty_cameras': empty_cameras or 0,
                'effective_cameras': effective,
                'deployed_cameras': num_cameras_deployed,
                'ok': effective == num_cameras_deployed,
            }
            if empty_cameras == 0 and policy_type == 'smolvla':
                checks['smolvla_empty_cameras_bug'] = {
                    'risk': 'prepare_images breaks with empty_cameras=0, drops visual context',
                    'ok': False,
                }

        output_action_dim = ckpt_config.get('output_action_dim', None)
        if output_action_dim is not None:
            checks['action_dim'] = {
                'output_action_dim': output_action_dim,
                'expected': 6,
                'ok': output_action_dim == 6,
            }

        chunk_size = ckpt_config.get('chunk_size', ckpt_config.get('n_action_steps', None))
        if chunk_size is not None:
            checks['chunk_size'] = {
                'value': chunk_size,
                'ok': 1 <= chunk_size <= 30,
            }

        fps_train = ckpt_config.get('fps', None)
        fps_inference = self.config.get('fps', 30)
        if fps_train is not None:
            checks['fps_match'] = {
                'training_fps': fps_train,
                'inference_fps': fps_inference,
                'ok': fps_train == fps_inference,
            }

        for fname in ['policy_preprocessor.json', 'policy_postprocessor.json']:
            fpath = os.path.join(checkpoint_dir, fname)
            checks[fname.replace('.json', '')] = {
                'exists': os.path.exists(fpath),
                'ok': os.path.exists(fpath),
            }

        norm_path = os.path.join(checkpoint_dir, 'unnormalizer.safetensors')
        if not os.path.exists(norm_path):
            norm_path = os.path.join(checkpoint_dir, 'normalizer.safetensors')
        checks['normalization_stats'] = {
            'path': norm_path,
            'exists': os.path.exists(norm_path),
            'ok': os.path.exists(norm_path),
        }

        if os.path.islink(checkpoint_dir) and not os.path.exists(checkpoint_dir):
            checks['broken_symlink'] = {
                'checkpoint_dir': checkpoint_dir,
                'ok': False,
            }

        all_ok = all(c.get('ok', True) for c in checks.values())

        doc = {
            'doc_type': 'model_setup',
            'timestamp': _iso_now(),
            'run_id': self._run_id,
            'model_name': self.config.get('model_name', 'unknown'),
            'policy_type': policy_type,
            'checkpoint_dir': checkpoint_dir,
            'checks': checks,
            'all_checks_passed': all_ok,
            'config_raw': {k: v for k, v in ckpt_config.items()
                          if k in ('num_image_views', 'empty_cameras',
                                   'output_action_dim', 'chunk_size',
                                   'n_action_steps', 'fps', 'type',
                                   '_target_', 'image_features',
                                   'input_features', 'output_features')},
        }

        with self._buffer_lock:
            self._buffer.append((IDX_SETUP, doc))
        self._flush_now()

        status = 'PASS' if all_ok else 'FAIL'
        failed = [k for k, v in checks.items() if not v.get('ok', True)]
        if failed:
            print(f'[TELEMETRY] model_setup {status}: FAILED checks: {failed}')
        else:
            print(f'[TELEMETRY] model_setup {status}: all checks passed')

    def record_input_audit(self, observation, batch, step=0):
        """Log input tensor shapes on step 0. Catches camera count mismatches,
        wrong image resolution, missing state vectors. Zero ongoing cost."""
        if step != 0 or not self._trial_id:
            return

        image_keys = [k for k in observation if 'image' in k.lower()]
        state_keys = [k for k in observation if 'state' in k.lower()]

        image_shapes = {}
        for k in image_keys:
            v = observation[k]
            if hasattr(v, 'shape'):
                image_shapes[k] = list(v.shape)

        batch_image_shapes = {}
        if isinstance(batch, dict):
            for k in batch:
                if 'image' in k.lower() and hasattr(batch[k], 'shape'):
                    batch_image_shapes[k] = list(batch[k].shape)

        state_shapes = {}
        for k in state_keys:
            v = observation[k]
            if hasattr(v, 'shape'):
                state_shapes[k] = list(v.shape)

        doc = {
            **self._trial_meta,
            'doc_type': 'input_audit',
            'timestamp': _iso_now(),
            'step': step,
            'num_cameras_in_observation': len(image_keys),
            'camera_keys': image_keys,
            'image_shapes': image_shapes,
            'batch_image_shapes': batch_image_shapes,
            'state_keys': state_keys,
            'state_shapes': state_shapes,
            'has_task': 'task' in observation,
            'batch_keys': list(batch.keys()) if isinstance(batch, dict) else [],
        }

        with self._buffer_lock:
            self._buffer.append((IDX_SETUP, doc))
        print(f'[TELEMETRY] input_audit: {len(image_keys)} cameras, '
              f'shapes={image_shapes}')

    def start_trial(self, task_instruction):
        self._trial_id = uuid.uuid4().hex[:12]
        self._trial_start = time.time()
        self._trial_steps = []
        self._trial_meta = {
            'trial_id': self._trial_id,
            'run_id': self._run_id,
            'model_name': self.config.get('model_name', 'unknown'),
            'policy_type': self.config.get('policy_type', 'unknown'),
            'task_instruction': ''.join(c for c in task_instruction if c.isprintable() or c == ' '),
            'config_fps': self.config.get('fps', 30),
            'config_ema_alpha': self.config.get('ema_alpha', 0.6),
            'config_max_delta': self.config.get('max_delta', 2.0),
            'config_n_action_steps': self.config.get('n_action_steps', 15),
            'training_steps': self.config.get('training_steps', 0),
            'training_loss': self.config.get('training_loss', 0.0),
            'frozen_components': self.config.get('frozen_components', ''),
            'dataset_episodes': self.config.get('dataset_episodes', 0),
        }
        trial_dir = Path(FRAME_STORE_DIR) / self._trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        print(f'[TELEMETRY] trial started: {self._trial_id}')
        return self._trial_id

    def end_trial(self, outcome=None, outcome_notes=None):
        if not self._trial_id:
            return

        summary = self._compute_trial_summary(outcome, outcome_notes)
        with self._buffer_lock:
            if len(self._buffer) < MAX_BUFFER_SIZE:
                self._buffer.append((IDX_TELEMETRY, summary))

        self._flush_now()
        print(f'[TELEMETRY] trial ended: {self._trial_id} — '
              f'{summary.get("failure_mode", "?")} '
              f'({summary.get("total_steps", 0)} steps)')
        self._trial_id = None
        self._trial_steps = []

    def record_step(
        self,
        step,
        raw_actions,
        after_ema,
        commanded,
        actual_joint_state,
        zscore_actions,
        clamp_amounts,
        velocity,
        tracking_error,
        pipeline_distortion,
        inference_latency_ms,
        is_chunk_boundary,
        chunk_id,
        chunk_step_index,
        queue_depth,
        frame_ages_ms,
    ):
        if not self._trial_id:
            return

        doc = {
            **self._trial_meta,
            'doc_type': 'step',
            'timestamp': _iso_now(),
            'step': step,
            'queue_depth': queue_depth,
            'is_chunk_boundary': is_chunk_boundary,
            'chunk_id': chunk_id,
            'chunk_step_index': chunk_step_index,
            'pipeline_distortion': round(pipeline_distortion, 4),
        }

        if inference_latency_ms is not None:
            doc['inference_latency_ms'] = round(inference_latency_ms, 2)

        for i in range(6):
            doc[f'raw_j{i}'] = round(float(raw_actions[i]), 3)
            doc[f'ema_j{i}'] = round(float(after_ema[i]), 3)
            doc[f'commanded_j{i}'] = round(float(commanded[i]), 3)
            doc[f'clamp_j{i}'] = round(float(clamp_amounts[i]), 3)

            if actual_joint_state is not None and i < len(actual_joint_state):
                doc[f'actual_j{i}'] = round(float(actual_joint_state[i]), 3)
            if velocity is not None:
                doc[f'velocity_j{i}'] = velocity[i]
            if tracking_error is not None:
                doc[f'tracking_error_j{i}'] = tracking_error[i]

        if zscore_actions is not None and is_chunk_boundary:
            for i in range(min(6, len(zscore_actions))):
                doc[f'zscore_j{i}'] = round(float(zscore_actions[i]), 5)

        if frame_ages_ms:
            for cam, age in frame_ages_ms.items():
                doc[f'frame_age_{cam}_ms'] = round(age, 1)

        try:
            fk = forward_kinematics(commanded)
            doc['ee_x'] = round(fk['x'] * 1000, 1)
            doc['ee_y'] = round(fk['y'] * 1000, 1)
            doc['ee_z'] = round(fk['z'] * 1000, 1)
            doc['elbow_x'] = round(fk['elbow_x'] * 1000, 1)
            doc['elbow_y'] = round(fk['elbow_y'] * 1000, 1)
            doc['elbow_z'] = round(fk['elbow_z'] * 1000, 1)
        except Exception:
            pass

        self._trial_steps.append(doc)

        with self._buffer_lock:
            if len(self._buffer) < MAX_BUFFER_SIZE:
                self._buffer.append((IDX_TELEMETRY, doc))
            elif step % 100 == 0:
                print(f'[TELEMETRY] buffer full, dropping step {step}')

    def record_frame(self, step, camera_topic, jpeg_bytes, resolution,
                     chunk_id=None):
        if not self._trial_id:
            return

        frame_dir = Path(FRAME_STORE_DIR) / self._trial_id
        filename = f'step{step:05d}_{camera_topic}.jpg'
        frame_path = frame_dir / filename

        frame_doc = {
            'trial_id': self._trial_meta['trial_id'],
            'run_id': self._trial_meta['run_id'],
            'model_name': self._trial_meta['model_name'],
            'policy_type': self._trial_meta['policy_type'],
            'task_instruction': self._trial_meta['task_instruction'],
            'timestamp': _iso_now(),
            'step': step,
            'camera_topic': camera_topic,
            'camera_role': CAMERA_ROLES.get(camera_topic, camera_topic),
            'frame_path': str(frame_path),
            'frame_size_bytes': len(jpeg_bytes),
            'frame_resolution_w': resolution[0] if resolution else None,
            'frame_resolution_h': resolution[1] if resolution else None,
            'chunk_id': chunk_id,
        }

        with self._frame_lock:
            self._frame_queue.append((frame_doc, jpeg_bytes, frame_path))

    def _compute_trial_summary(self, outcome=None, outcome_notes=None):
        steps = self._trial_steps
        n = len(steps)
        duration = time.time() - self._trial_start

        summary = {
            **self._trial_meta,
            'doc_type': 'trial_summary',
            'timestamp': _iso_now(),
            'total_steps': n,
            'trial_duration_sec': round(duration, 2),
            'trial_outcome': outcome or 'unscored',
            'trial_outcome_notes': outcome_notes or '',
        }

        if n < 2:
            summary['failure_mode'] = 'aborted'
            return summary

        commanded = np.array([
            [s.get(f'commanded_j{j}', 0) for j in range(6)] for s in steps
        ])
        raw = np.array([
            [s.get(f'raw_j{j}', 0) for j in range(6)] for s in steps
        ])

        var_per_joint = np.std(commanded, axis=0)
        summary['action_variance_mean'] = round(float(np.mean(var_per_joint)), 3)
        for j in range(6):
            summary[f'action_variance_j{j}'] = round(float(var_per_joint[j]), 3)

        range_per_joint = np.ptp(commanded, axis=0)
        for j in range(6):
            summary[f'action_range_j{j}'] = round(float(range_per_joint[j]), 2)

        zscore_vals = []
        for s in steps:
            if s.get('zscore_j0') is not None:
                zscore_vals.append([s.get(f'zscore_j{j}', 0) for j in range(6)])
        if zscore_vals:
            zs = np.array(zscore_vals)
            summary['z_score_magnitude_mean'] = round(float(np.mean(np.abs(zs))), 4)
            for j in range(6):
                summary[f'z_score_std_j{j}'] = round(float(np.std(zs[:, j])), 4)
        else:
            summary['z_score_magnitude_mean'] = 0.0

        j5 = commanded[:, 5]
        direction_changes = 0
        for i in range(2, n):
            prev_dir = j5[i - 1] - j5[i - 2]
            curr_dir = j5[i] - j5[i - 1]
            if prev_dir * curr_dir < 0 and abs(curr_dir) > 1.0:
                direction_changes += 1
        summary['gripper_transitions'] = direction_changes

        boundary_steps = [s for s in steps if s.get('is_chunk_boundary')]
        chunk_discs = []
        for i in range(1, len(boundary_steps)):
            idx = boundary_steps[i]['step']
            prev_idx = boundary_steps[i - 1]['step']
            if prev_idx < n and idx < n:
                disc = float(np.max(np.abs(raw[idx] - commanded[prev_idx])))
                chunk_discs.append(disc)
        summary['chunk_discontinuity_mean'] = (
            round(float(np.mean(chunk_discs)), 2) if chunk_discs else 0.0
        )
        summary['chunk_discontinuity_max'] = (
            round(float(np.max(chunk_discs)), 2) if chunk_discs else 0.0
        )

        clamp_count = sum(
            1 for s in steps
            if any(abs(s.get(f'clamp_j{j}', 0)) > 0.01 for j in range(6))
        )
        summary['total_clamp_activations'] = clamp_count
        summary['clamp_rate_pct'] = round(clamp_count / max(n, 1) * 100, 1)

        if any(s.get('tracking_error_j0') is not None for s in steps):
            max_te = 0.0
            for s in steps:
                for j in range(6):
                    te = abs(s.get(f'tracking_error_j{j}', 0))
                    if te > max_te:
                        max_te = te
            summary['max_tracking_error'] = round(float(max_te), 2)
            te_all = []
            for s in steps:
                for j in range(6):
                    te_all.append(abs(s.get(f'tracking_error_j{j}', 0)))
            summary['mean_tracking_error'] = round(float(np.mean(te_all)), 2)

        ee_positions = [(s.get('ee_x'), s.get('ee_y'), s.get('ee_z'))
                        for s in steps if s.get('ee_x') is not None]
        if ee_positions:
            xs, ys, zs = zip(*ee_positions)
            summary['workspace_x_range'] = round(max(xs) - min(xs), 1)
            summary['workspace_y_range'] = round(max(ys) - min(ys), 1)
            summary['workspace_z_range'] = round(max(zs) - min(zs), 1)
            summary['workspace_lateral_max'] = round(max(abs(y) for y in ys), 1)

        pipeline_dists = [s.get('pipeline_distortion', 0) for s in steps]
        summary['pipeline_distortion_mean'] = round(
            float(np.mean(pipeline_dists)), 3
        )
        summary['pipeline_distortion_max'] = round(
            float(np.max(pipeline_dists)), 3
        )

        latencies = [
            s['inference_latency_ms'] for s in steps
            if s.get('inference_latency_ms') is not None
        ]
        if latencies:
            summary['inference_latency_p50'] = round(float(np.median(latencies)), 1)
            summary['inference_latency_p95'] = round(
                float(np.percentile(latencies, 95)), 1
            )

        summary['failure_mode'] = _classify_failure(summary)

        av = summary.get('action_variance_mean', 0)
        pd_ = summary.get('pipeline_distortion_mean', 0)
        mte = summary.get('mean_tracking_error', 0)
        motion = min(1.0, av / 15.0)
        fidelity = max(0.0, 1.0 - pd_ / (av + 1.0))
        tracking = max(0.0, 1.0 - mte / 20.0)
        gt = summary.get('gripper_transitions', 0)
        gripper = max(0.0, 1.0 - max(0, gt - V1_REF['gripper_transitions']) / 30.0)
        summary['pipeline_score'] = round(
            (motion + fidelity + tracking + gripper) / 4.0, 3
        )
        summary['score_motion'] = round(motion, 3)
        summary['score_fidelity'] = round(fidelity, 3)
        summary['score_tracking'] = round(tracking, 3)
        summary['score_gripper'] = round(gripper, 3)

        return summary

    def _flush_loop(self):
        while self._running:
            time.sleep(FLUSH_INTERVAL)
            try:
                self._flush_now()
            except Exception as e:
                print(f'[TELEMETRY] flush loop error (continuing): {e}')

    def _flush_now(self):
        with self._buffer_lock:
            batch = list(self._buffer)
            self._buffer.clear()
        if batch:
            _es_bulk(batch)

    def _frame_loop(self):
        while self._running:
            time.sleep(0.5)
            with self._frame_lock:
                batch = list(self._frame_queue)
                self._frame_queue.clear()

            for frame_doc, jpeg_bytes, frame_path in batch:
                try:
                    frame_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(frame_path, 'wb') as f:
                        f.write(jpeg_bytes)
                except Exception as e:
                    print(f'[TELEMETRY] frame save failed: {e}')
                    continue

                embedding = _embed_clip(jpeg_bytes)
                if embedding:
                    frame_doc['clip_embedding'] = embedding
                    dist = _nearest_training_distance(embedding)
                    if dist is not None:
                        frame_doc['nearest_training_distance'] = round(dist, 4)

                with self._buffer_lock:
                    if len(self._buffer) < MAX_BUFFER_SIZE:
                        self._buffer.append((IDX_FRAMES, frame_doc))

    def shutdown(self):
        self._running = False
        self._flush_now()
        print('[TELEMETRY] collector shut down')


def _iso_now():
    t = time.time()
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t)) + f'.{int(t * 1000) % 1000:03d}Z'
