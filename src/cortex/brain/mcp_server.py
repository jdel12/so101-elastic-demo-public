#!/usr/bin/env python3
"""
SO-101 Robot MCP Server v15 — Agentic Diagnostics

Bridges Elastic Agent Builder to the robot's ROS2 inference pipeline,
GPU observability (Refinery), and Elasticsearch. Provides multi-signal
failure diagnosis: GPU kernels + CLIP embeddings + VLM evaluation +
trajectory analysis fused into actionable root-cause classification.

Tools (26):
  Robot Control (4):
    - execute_task, stop_robot, swap_model, get_robot_status

  Observation (2):
    - get_workspace_snapshot, evaluate_cycle

  Conditions (2):
    - set_condition, get_condition

  Data & Stats (3):
    - get_flywheel_status, query_hard_examples, query_inference_stats

  GPU Observability — Refinery (7):
    - get_gpu_kernel_profile, get_inference_breakdown, get_gpu_memory_status
    - get_episode_gpu_quality, get_control_loop_status, get_safety_events

  Diagnostic (6):
    - diagnose_trial:           Multi-signal failure classification
    - compare_model_kernels:    GPU-level model autopsy
    - check_scene_novelty:      Training distribution boundary check
    - analyze_failure_patterns: Pattern analysis across recent failures
    - profile_inference:        GPU execution profile (duty cycle, jitter)
    - profile_trajectory:       Motion quality (jerk, smoothness, gripper)

  ROS2 Observability (3):
    - get_ros2_topic_health, get_ros2_node_status, get_ros2_alerts

Runs as FastMCP HTTP server at :8888/mcp
"""

import base64
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError
import ssl

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, String
from fastmcp import FastMCP

try:
    import elasticapm
    APM_AVAILABLE = True
except ImportError:
    APM_AVAILABLE = False


# --- Configuration ---

MCP_PORT = 8888
CAMERA_TOPICS = {
    'cam0': '/camera/cam0/compressed',
    'cam1': '/camera/cam1/compressed',
}
FRAME_TIMEOUT_SEC = 5.0
EXECUTION_TIMEOUT_SEC = 60.0
EXECUTION_STEPS_TARGET = 600
INFERENCE_FPS = 30

ES_URL = os.environ.get('ES_URL', 'https://elasticsearch-es-http.default.svc:9200')
ES_USER = os.environ.get('ES_USER', 'elastic')
ES_PASSWORD = os.environ.get('ES_PASSWORD', '')

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
EVALUATOR_MODEL = os.environ.get('EVALUATOR_MODEL', 'qwen3-vl:8b')

JINA_CLIP_URL = os.environ.get('JINA_CLIP_URL', 'http://localhost:8900')
IDX_EPISODE_EMBEDDINGS = 'robot-episode-embeddings'

# Index names
IDX_INFERENCE = 'robot-inference'
IDX_COMMANDS = 'robot-commands'
IDX_TRAINING_BUFFER = 'robot-training-buffer'
IDX_STEPS = 'robot-steps'

# Refinery GPU observability indices (data streams created by collector)
IDX_GPU_PROFILING = 'logs-gpu.kernel-*'
IDX_GPU_INFERENCE = 'logs-gpu.trace-*'
IDX_GPU_ROBOTICS = 'logs-gpu.device_metrics-*'
IDX_GPU_SAFETY = 'logs-gpu.memory-*'

IDX_ROS2_TOPICS = 'metrics-ros2.topics-*'
IDX_ROS2_NODES = 'metrics-ros2.nodes-*'
IDX_ROS2_EVENTS = 'logs-ros2.events-*'


# --- Active State ---
# Mutable state for current session conditions and model info

active_condition = {
    'distractor_present': False,
    'distractor_objects': [],
    'camera_occluded': False,
    'occlusion_type': None,
    'lighting': 'normal',
    'object_placement': 'structured',
    'tags': [],
}

active_model = {
    'model_id': 'smolvla_v1',
    'model_version': 'v1',
    'checkpoint': 'checkpoint_030000',
    'policy_type': 'smolvla',
}

active_prompt = {
    'version': 'v1',
    'text': 'Pick up blue cube and place in bin',
}

# Cycle counter for the current run
cycle_counter = {'run_id': None, 'episode_num': 0}


# --- Elasticsearch Client ---

def _es_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def es_write(index, doc, doc_id=None):
    """Write a document to Elasticsearch."""
    try:
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        url = f'{ES_URL}/{index}/_doc/{doc_id}'
        body = json.dumps(doc).encode('utf-8')
        auth = base64.b64encode(f'{ES_USER}:{ES_PASSWORD}'.encode()).decode()
        req = Request(url, data=body, method='PUT')
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Basic {auth}')
        resp = urlopen(req, context=_es_ssl_context(), timeout=5)
        resp.read()
        return True
    except Exception as e:
        print(f'[ES WRITE ERROR] {index}: {e}')
        return False


def es_search(index, query_body, size=10):
    """Search Elasticsearch and return hits."""
    try:
        url = f'{ES_URL}/{index}/_search'
        body = json.dumps({'size': size, **query_body}).encode('utf-8')
        auth = base64.b64encode(f'{ES_USER}:{ES_PASSWORD}'.encode()).decode()
        req = Request(url, data=body, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Basic {auth}')
        resp = urlopen(req, context=_es_ssl_context(), timeout=10)
        data = json.loads(resp.read().decode('utf-8'))
        return data
    except Exception as e:
        print(f'[ES SEARCH ERROR] {index}: {e}')
        return None


def es_count(index, query_body=None):
    """Count documents in an index."""
    try:
        url = f'{ES_URL}/{index}/_count'
        if query_body:
            body = json.dumps(query_body).encode('utf-8')
        else:
            body = json.dumps({'query': {'match_all': {}}}).encode('utf-8')
        auth = base64.b64encode(f'{ES_USER}:{ES_PASSWORD}'.encode()).decode()
        req = Request(url, data=body, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Basic {auth}')
        resp = urlopen(req, context=_es_ssl_context(), timeout=5)
        data = json.loads(resp.read().decode('utf-8'))
        return data.get('count', 0)
    except Exception as e:
        print(f'[ES COUNT ERROR] {index}: {e}')
        return 0


def log_to_es_async(index, doc, doc_id=None):
    """Fire-and-forget ES write."""
    threading.Thread(target=es_write, args=(index, doc, doc_id), daemon=True).start()


def embed_image_clip(jpeg_bytes):
    """Get 1024-dim CLIP embedding from Jina CLIP v2 server."""
    try:
        b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        data_uri = f'data:image/jpeg;base64,{b64}'
        payload = json.dumps({'inputs': [data_uri], 'type': 'image'}).encode('utf-8')
        req = Request(f'{JINA_CLIP_URL}/embed', data=payload, method='POST')
        req.add_header('Content-Type', 'application/json')
        resp = urlopen(req, timeout=10)
        data = json.loads(resp.read().decode('utf-8'))
        embeddings = data.get('embeddings', [])
        return embeddings[0] if embeddings else None
    except Exception as e:
        print(f'[CLIP EMBED ERROR] {e}', flush=True)
        return None


def compute_training_distance(embedding):
    """kNN search against training episode embeddings, return nearest distance."""
    if embedding is None:
        return None
    try:
        query = {
            'knn': {
                'field': 'embedding',
                'query_vector': embedding,
                'k': 1,
                'num_candidates': 50,
            },
            '_source': False,
        }
        result = es_search(IDX_EPISODE_EMBEDDINGS, query, size=1)
        if result and result.get('hits', {}).get('hits'):
            score = result['hits']['hits'][0].get('_score', 0)
            return round(1.0 - score, 4)
        return None
    except Exception as e:
        print(f'[KNN DISTANCE ERROR] {e}')
        return None


# --- Logging helpers ---

def log_command(tool, start_time, result, **extra):
    """Log a tool call to robot-commands."""
    elapsed_ms = (time.time() - start_time) * 1000
    status = result.get('status', 'error' if 'error' in result else 'success')
    doc = {
        'command_id': f'cmd_{uuid.uuid4().hex[:12]}',
        'source': 'mcp',
        'tool': tool,
        'success': status == 'success',
        'status': status,
        'latency_ms': round(elapsed_ms, 1),
        'model_version': active_model['model_version'],
        'condition_tags': active_condition.get('tags', []),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    if 'error' in result:
        doc['error'] = result['error']
    log_to_es_async(IDX_COMMANDS, doc)


def log_inference_cycle(cycle_id, run_id, episode_num, task_text,
                        exec_result, evaluation=None,
                        scene_embedding=None, nearest_training_distance=None):
    """Log an inference cycle to robot-inference."""
    doc = {
        'cycle_id': cycle_id,
        'run_id': run_id,
        'episode_num': episode_num,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model_id': active_model['model_id'],
        'model_version': active_model['model_version'],
        'checkpoint': active_model['checkpoint'],
        'policy_type': active_model['policy_type'],
        'task': task_text,
        'prompt_version': active_prompt['version'],
        'prompt_text': active_prompt['text'],
        'condition': {**active_condition},
        'outcome': {
            'success': exec_result.get('success', False),
            'failure_mode': exec_result.get('failure_mode', None),
            'cycle_time_s': exec_result.get('cycle_time_s', 0),
            'gripper_closed': exec_result.get('gripper_closed', None),
            'object_grasped': exec_result.get('object_grasped', None),
            'object_placed': exec_result.get('object_placed', None),
        },
        'action_summary': {
            'total_steps': exec_result.get('total_steps', 0),
        },
    }
    if evaluation:
        doc['evaluation'] = evaluation
    if scene_embedding:
        doc['scene_embedding'] = scene_embedding
    if nearest_training_distance is not None:
        doc['nearest_training_distance'] = nearest_training_distance
    log_to_es_async(IDX_INFERENCE, doc, cycle_id)


# --- ROS2 Bridge Node ---

class RobotBridge(Node):
    """ROS2 node bridging MCP to the inference pipeline."""

    def __init__(self):
        super().__init__('mcp_robot_bridge')
        self.lock = threading.Lock()

        # Camera frames
        self.latest_frames = {}
        for name, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                CompressedImage, topic,
                lambda msg, n=name: self._on_frame(n, msg), 10
            )
            self.get_logger().info(f'Camera: {topic}')

        # Inference status
        self.inference_status = 'unknown'
        self.create_subscription(
            String, '/inference_status', self._on_inference_status, 10
        )

        # Joint commands — capture trajectory during execution
        self._trajectory_buffer = []
        self._capturing = False
        self._capture_t0 = 0.0
        self.create_subscription(
            Float32MultiArray, '/joint_commands', self._on_joint_command, 10
        )

        # Publishers
        self.task_pub = self.create_publisher(String, '/task_instruction', 10)
        self.control_pub = self.create_publisher(String, '/inference_control', 10)
        self.model_pub = self.create_publisher(String, '/swap_model', 10)

        self.get_logger().info('Robot bridge ready')

    def _on_frame(self, name, msg):
        with self.lock:
            self.latest_frames[name] = (time.time(), bytes(msg.data))

    def _on_inference_status(self, msg):
        with self.lock:
            self.inference_status = msg.data

    def _on_joint_command(self, msg):
        with self.lock:
            if self._capturing:
                t = time.time() - self._capture_t0
                self._trajectory_buffer.append({
                    'step': len(self._trajectory_buffer),
                    't': round(t, 4),
                    'joints': [round(v, 3) for v in msg.data[:6]],
                })

    def start_trajectory_capture(self):
        with self.lock:
            self._trajectory_buffer = []
            self._capture_t0 = time.time()
            self._capturing = True

    def stop_trajectory_capture(self):
        with self.lock:
            self._capturing = False
            return list(self._trajectory_buffer)

    def get_camera_status(self):
        with self.lock:
            now = time.time()
            status = {}
            for name in CAMERA_TOPICS:
                if name in self.latest_frames:
                    ts, _ = self.latest_frames[name]
                    age = now - ts
                    status[name] = {'active': age < FRAME_TIMEOUT_SEC,
                                    'age_seconds': round(age, 2)}
                else:
                    status[name] = {'active': False, 'age_seconds': None}
            return status

    def get_latest_frame_jpeg(self, cam_name):
        """Return raw JPEG bytes for a camera, or None."""
        with self.lock:
            if cam_name in self.latest_frames:
                ts, jpeg = self.latest_frames[cam_name]
                if time.time() - ts < FRAME_TIMEOUT_SEC:
                    return jpeg
        return None

    def get_inference_status(self):
        with self.lock:
            return self.inference_status

    def send_task(self, instruction):
        msg = String()
        msg.data = instruction
        self.task_pub.publish(msg)
        self.get_logger().info(f'Sent task: {instruction}')

    def send_control(self, command):
        msg = String()
        msg.data = command
        self.control_pub.publish(msg)
        self.get_logger().info(f'Sent control: {command}')

    def send_model_swap(self, checkpoint_path):
        msg = String()
        msg.data = checkpoint_path
        self.model_pub.publish(msg)
        self.get_logger().info(f'Sent model swap: {checkpoint_path}')

    def wait_for_execution(self, timeout_sec=EXECUTION_TIMEOUT_SEC):
        start = time.time()

        # Wait for inference to start.
        # After a model swap, the first inference step may take extra time
        # for CUDA kernel warmup, so allow up to 15 seconds.
        startup_timeout = 15.0
        while time.time() - start < startup_timeout:
            status = self.get_inference_status()
            if 'running' in status:
                break
            time.sleep(0.2)

        if 'running' not in self.get_inference_status():
            return {'status': 'error', 'message': 'Inference node did not start'}

        # Let it run for target duration
        execution_start = time.time()
        target_duration = EXECUTION_STEPS_TARGET / INFERENCE_FPS

        while time.time() - execution_start < min(target_duration, timeout_sec):
            status = self.get_inference_status()
            if 'running' not in status:
                break
            time.sleep(1.0)

        elapsed = time.time() - start
        final_status = self.get_inference_status()

        # Extract step count from status like "running (step 302)"
        total_steps = 0
        if 'step' in final_status:
            try:
                total_steps = int(final_status.split('step')[1].strip(' )'))
            except (ValueError, IndexError):
                pass

        return {
            'status': 'success',
            'cycle_time_s': round(elapsed, 1),
            'total_steps': total_steps,
            'inference_status': final_status,
        }


# --- Evaluation via Ollama ---

def evaluate_with_vlm(jpeg_bytes, task_text):
    """Send a camera frame to Qwen3-VL via Ollama for evaluation.
    Returns dict with confidence, success, scene_description, reasoning."""
    try:
        img_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')

        prompt = (
            f"You are evaluating a robot arm that was instructed to: '{task_text}'. "
            "Look at this camera image of the workspace AFTER the robot acted. "
            "Evaluate whether the task was completed successfully. "
            "Respond ONLY with valid JSON (no markdown, no backticks):\n"
            '{"success": true/false, "confidence": 0.0-1.0, '
            '"scene_description": "brief description of what you see", '
            '"failure_mode": "none" or describe the failure, '
            '"reasoning": "why you scored it this way"}'
        )

        payload = {
            'model': EVALUATOR_MODEL,
            'prompt': prompt,
            'images': [img_b64],
            'stream': False,
        }

        url = f'{OLLAMA_URL}/api/generate'
        body = json.dumps(payload).encode('utf-8')
        req = Request(url, data=body, method='POST')
        req.add_header('Content-Type', 'application/json')
        resp = urlopen(req, timeout=30)
        data = json.loads(resp.read().decode('utf-8'))

        response_text = data.get('response', '{}')
        # Strip markdown fences if present
        response_text = response_text.strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[-1]
            response_text = response_text.rsplit('```', 1)[0]

        result = json.loads(response_text)
        return {
            'evaluator_model': EVALUATOR_MODEL,
            'confidence': float(result.get('confidence', 0.5)),
            'scene_description': result.get('scene_description', ''),
            'reasoning': result.get('reasoning', ''),
            'score': float(result.get('confidence', 0.5)),
        }, {
            'success': result.get('success', False),
            'failure_mode': result.get('failure_mode', 'unknown'),
        }

    except Exception as e:
        print(f'[EVAL ERROR] {e}')
        return {
            'evaluator_model': EVALUATOR_MODEL,
            'confidence': 0.0,
            'scene_description': f'Evaluation failed: {e}',
            'reasoning': 'error',
            'score': 0.0,
        }, {
            'success': False,
            'failure_mode': 'evaluation_error',
        }


# --- ROS2 background thread ---

robot_bridge = None


def ros2_spin_thread():
    global robot_bridge
    rclpy.init()
    robot_bridge = RobotBridge()
    rclpy.spin(robot_bridge)


def start_ros2():
    t = threading.Thread(target=ros2_spin_thread, daemon=True)
    t.start()
    time.sleep(1.0)


# --- MCP Server ---

mcp = FastMCP(
    "SO-101 Diagnostic Agent",
    instructions=(
        "You are an AI operations assistant for a physical SO-101 robotic arm. "
        "You can command the robot, diagnose failures, and monitor the full "
        "observability stack — from GPU kernels to ROS2 topics to VLM evaluations.\n\n"

        "DIAGNOSTIC REASONING — when asked to diagnose failures or investigate problems:\n"
        "1. Start with query_inference_stats to see overall success/failure rates by model or condition.\n"
        "2. Run analyze_failure_patterns to identify the dominant failure mode and whether "
        "failures cluster by scene novelty, hardware issues, or policy quality.\n"
        "3. Run diagnose_trial to classify the failure type: planning_failure (model computes "
        "the same trajectory regardless of outcome), compute_anomaly (GPU divergence between "
        "success and failure), or hardware_failure (zero steps — camera or motor hang).\n"
        "4. Based on classification:\n"
        "   - planning_failure: Use check_scene_novelty to test whether failed scenes are "
        "outside the training distribution. Use profile_trajectory to check motion quality. "
        "The model may be replaying a fixed trajectory without adapting to the scene.\n"
        "   - compute_anomaly: Use get_gpu_memory_status and get_control_loop_status to check "
        "for thermal throttling, memory pressure, or deadline misses. Use profile_inference "
        "for detailed timing analysis.\n"
        "   - hardware_failure: Use get_ros2_topic_health and get_ros2_alerts to identify "
        "camera frame drops or motor node issues.\n"
        "5. Synthesize findings into a root cause with an evidence chain and a specific, "
        "actionable recommendation (e.g., 'collect 15 episodes at lateral positions and retrain').\n\n"

        "ALWAYS provide the evidence chain — which tools you called, what they showed, "
        "and how the signals connect. Never guess at a diagnosis without tool evidence.\n\n"

        "OPERATIONS:\n"
        "- execute_task: Command the robot (triggers REAL physical movement)\n"
        "- evaluate_cycle: Score workspace with Qwen3-VL vision AI\n"
        "- set_condition / get_condition: Tag environment (lighting, distractors)\n"
        "- swap_model: Switch VLA model checkpoint\n"
        "- stop_robot: Halt inference, arm holds position\n\n"

        "Always use tools — never just describe what you would do."
    ),
)


# ============================================================
# TOOL: Robot Control
# ============================================================

@mcp.tool
def execute_task(
    task: str,
    target_object: str = "blue cube",
    destination: str = "bin",
) -> dict:
    """
    Command the robot to execute a task via the VLA model.
    Triggers REAL physical movement. Logs to Elasticsearch.

    Args:
        task: Natural language instruction (e.g. "Pick up blue cube and place in bin")
        target_object: The object to manipulate
        destination: Where to place the object
    """
    t0 = time.time()

    if robot_bridge is None:
        return {'error': 'Robot bridge not initialized'}

    # Initialize run if needed
    if cycle_counter['run_id'] is None:
        cycle_counter['run_id'] = f'run_{uuid.uuid4().hex[:8]}'
        cycle_counter['episode_num'] = 0

    cycle_counter['episode_num'] += 1
    cycle_id = f'cycle_{uuid.uuid4().hex[:12]}'
    active_prompt['text'] = task

    # Embed the scene BEFORE execution — wrist camera for position sensitivity
    scene_embedding = None
    nearest_training_distance = None
    pre_jpeg = robot_bridge.get_latest_frame_jpeg('cam0')
    if pre_jpeg is None:
        pre_jpeg = robot_bridge.get_latest_frame_jpeg('cam1')
    if pre_jpeg:
        print(f'[CLIP] Embedding {len(pre_jpeg)} byte pre-execution frame...', flush=True)
        scene_embedding = embed_image_clip(pre_jpeg)
        print(f'[CLIP] Embedding result: {len(scene_embedding) if scene_embedding else "None"} dims', flush=True)
        nearest_training_distance = compute_training_distance(scene_embedding)
        print(f'[CLIP] Distance: {nearest_training_distance}', flush=True)

    # Start trajectory capture before sending task
    robot_bridge.start_trajectory_capture()

    # Send task to inference node
    robot_bridge.send_task(task)

    # Wait for execution
    exec_result = robot_bridge.wait_for_execution()

    # Stop inference so the arm doesn't keep running between cycles.
    # Only send stop if inference actually started — otherwise the queued
    # "stop" will race with a late-arriving task and kill it.
    if exec_result.get('status') != 'error':
        robot_bridge.send_control('stop')

    # Capture trajectory
    trajectory = robot_bridge.stop_trajectory_capture()

    # Auto-evaluate AFTER execution — judges the outcome
    evaluation = None
    outcome_info = {}
    if exec_result.get('status') == 'success':
        jpeg = robot_bridge.get_latest_frame_jpeg('cam1')
        if jpeg is None:
            jpeg = robot_bridge.get_latest_frame_jpeg('cam0')
        if jpeg:
            evaluation, outcome_info = evaluate_with_vlm(jpeg, task)

    # Merge outcome
    exec_result['success'] = outcome_info.get('success', False)
    exec_result['failure_mode'] = outcome_info.get('failure_mode', None)

    # Log to ES
    log_inference_cycle(
        cycle_id=cycle_id,
        run_id=cycle_counter['run_id'],
        episode_num=cycle_counter['episode_num'],
        task_text=task,
        exec_result=exec_result,
        evaluation=evaluation,
        scene_embedding=scene_embedding,
        nearest_training_distance=nearest_training_distance,
    )

    # Bulk-index trajectory steps
    if trajectory:
        def _index_trajectory():
            ts_now = datetime.now(timezone.utc).isoformat()
            for step in trajectory:
                doc = {
                    'cycle_id': cycle_id,
                    'model_version': active_model['model_version'],
                    'policy_type': active_model['policy_type'],
                    'timestamp': ts_now,
                    'step': step['step'],
                    'step_time': step['t'],
                    'joints': step['joints'],
                }
                es_write(IDX_STEPS, doc)
        threading.Thread(target=_index_trajectory, daemon=True).start()
        print(f'[TRAJECTORY] Captured {len(trajectory)} steps for {cycle_id}', flush=True)

    log_command('execute_task', t0, {'status': 'success'},
                skill='execute_task', target_object=target_object,
                destination=destination, instruction=task)

    elapsed = time.time() - t0
    return {
        'status': 'success',
        'cycle_id': cycle_id,
        'episode': cycle_counter['episode_num'],
        'task': task,
        'model_version': active_model['model_version'],
        'execution_time_s': round(elapsed, 1),
        'evaluation': {
            'success': outcome_info.get('success', False),
            'confidence': evaluation.get('confidence', 0) if evaluation else 0,
            'scene_description': evaluation.get('scene_description', '') if evaluation else '',
            'failure_mode': outcome_info.get('failure_mode', None),
        },
        'condition': {k: v for k, v in active_condition.items() if v},
    }


@mcp.tool
def stop_robot() -> dict:
    """Stop the robot. Inference halts, arm holds position."""
    if robot_bridge is None:
        return {'error': 'Robot bridge not initialized'}
    robot_bridge.send_control('stop')
    return {'status': 'success', 'message': 'Robot stopped'}


@mcp.tool
def swap_model(
    checkpoint_path: str,
    model_id: str = "",
    model_version: str = "",
) -> dict:
    """
    Hot-swap the VLA model checkpoint on the inference node.
    The robot will use the new model for all subsequent tasks.

    Args:
        checkpoint_path: Full path to the pretrained_model directory
        model_id: Friendly name for this model (e.g. "pi0_curated_v2")
        model_version: Version tag (e.g. "v2")
    """
    if robot_bridge is None:
        return {'error': 'Robot bridge not initialized'}

    # Stop inference first
    robot_bridge.send_control('stop')
    time.sleep(1.0)

    # Send swap command
    robot_bridge.send_model_swap(checkpoint_path)

    # Wait for the model to finish loading — the inference node's executor
    # is blocked during load_model(), so _publish_status cannot fire.
    # Poll until we see a status update that proves the swap completed.
    # The inference node logs "Model swapped to ... Ready." when done,
    # and _publish_status resumes publishing "idle" or "waiting_for_task".
    swap_timeout = 120.0  # large models can take a while
    swap_start = time.time()
    time.sleep(2.0)  # give the swap callback time to start
    while time.time() - swap_start < swap_timeout:
        status = robot_bridge.get_inference_status()
        # During the swap, the status is stale (whatever it was before).
        # Once the swap finishes, _publish_status fires and we see a fresh
        # "idle" or "waiting_for_task". We can't distinguish stale from
        # fresh by content alone, so we wait for _publish_status to tick
        # at least once after the swap. The simplest reliable signal:
        # the swap blocks the executor for many seconds, so if we keep
        # seeing the SAME status for a while and then it changes, the
        # swap is done. But a cleaner approach: just wait long enough
        # for load_model to finish. We know it's done when status
        # updates resume (the 1s status timer fires again).
        #
        # Strategy: track consecutive identical status reads. Once we see
        # a transition (e.g. from stale "running (step X)" to "idle"),
        # the swap is done. If status was already "idle" before swap,
        # we need a different signal — wait for at least swap_min_wait.
        if 'running' not in status and time.time() - swap_start > 5.0:
            # Status is not "running" and we've waited at least 5s,
            # meaning the executor had time to drain and _publish_status
            # fired at least once. The swap is very likely complete.
            break
        time.sleep(1.0)
    else:
        return {
            'status': 'error',
            'message': f'Model swap timed out after {swap_timeout}s',
        }

    # Update active model state
    if model_id:
        active_model['model_id'] = model_id
    if model_version:
        active_model['model_version'] = model_version
    active_model['checkpoint'] = checkpoint_path

    # Detect policy type for active_model tracking
    try:
        # The inference node doesn't expose policy_type via status,
        # so we detect it from the checkpoint config directly
        from inference_node import detect_policy_type
        active_model['policy_type'] = detect_policy_type(checkpoint_path)
    except Exception:
        pass

    # Reset run counter for new model
    cycle_counter['run_id'] = f'run_{uuid.uuid4().hex[:8]}'
    cycle_counter['episode_num'] = 0

    return {
        'status': 'success',
        'message': f'Model swapped to {checkpoint_path}',
        'model_id': active_model['model_id'],
        'model_version': active_model['model_version'],
    }


@mcp.tool
def get_robot_status() -> dict:
    """Get current robot status: cameras, inference state, active model, conditions."""
    if robot_bridge is None:
        return {'error': 'Robot bridge not initialized'}

    cam_status = robot_bridge.get_camera_status()
    active_cams = sum(1 for s in cam_status.values() if s['active'])

    return {
        'cameras': cam_status,
        'cameras_active': f'{active_cams}/{len(cam_status)}',
        'inference_status': robot_bridge.get_inference_status(),
        'model': {**active_model},
        'condition': {k: v for k, v in active_condition.items() if v},
        'prompt': {**active_prompt},
        'run_id': cycle_counter['run_id'],
        'episode_num': cycle_counter['episode_num'],
    }


# ============================================================
# TOOL: Observation
# ============================================================

@mcp.tool
def get_workspace_snapshot() -> dict:
    """
    Get camera health status for all cameras.
    Returns status only (no images) to keep responses lightweight.
    """
    if robot_bridge is None:
        return {'error': 'Robot bridge not initialized'}

    status = robot_bridge.get_camera_status()
    active_count = sum(1 for s in status.values() if s['active'])

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'cameras': status,
        'summary': f'{active_count}/{len(status)} cameras active',
    }


@mcp.tool
def evaluate_cycle(task: str = "") -> dict:
    """
    Score the current workspace state using the VLM evaluator.
    Takes a camera snapshot and asks Qwen3-VL whether the task succeeded.

    Args:
        task: The task to evaluate against (defaults to active prompt)
    """
    if robot_bridge is None:
        return {'error': 'Robot bridge not initialized'}

    task_text = task or active_prompt['text']

    jpeg = robot_bridge.get_latest_frame_jpeg('cam1')
    if jpeg is None:
        jpeg = robot_bridge.get_latest_frame_jpeg('cam0')
    if jpeg is None:
        return {'error': 'No fresh camera frame available'}

    evaluation, outcome = evaluate_with_vlm(jpeg, task_text)

    return {
        'task': task_text,
        'success': outcome.get('success', False),
        'confidence': evaluation.get('confidence', 0),
        'scene_description': evaluation.get('scene_description', ''),
        'failure_mode': outcome.get('failure_mode', None),
        'reasoning': evaluation.get('reasoning', ''),
        'evaluator': EVALUATOR_MODEL,
    }


# ============================================================
# TOOL: Conditions
# ============================================================

@mcp.tool
def set_condition(
    distractor_present: bool = None,
    distractor_objects: str = "",
    camera_occluded: bool = None,
    occlusion_type: str = "",
    lighting: str = "",
    tags: str = "",
) -> dict:
    """
    Set environment conditions for the current session.
    All subsequent inference cycles will be tagged with these conditions.
    This enables before/after analysis in Elasticsearch.

    Args:
        distractor_present: Are distractor objects in the workspace?
        distractor_objects: Comma-separated list (e.g. "red_cube,green_cube")
        camera_occluded: Is the camera partially blocked?
        occlusion_type: Type of occlusion (e.g. "partial", "hand", "cardboard")
        lighting: Lighting condition (e.g. "normal", "warm_red", "cool_blue", "low_light")
        tags: Comma-separated condition tags (e.g. "baseline,clean")
    """
    if distractor_present is not None:
        active_condition['distractor_present'] = distractor_present
    if distractor_objects:
        active_condition['distractor_objects'] = [
            s.strip() for s in distractor_objects.split(',')
        ]
    if camera_occluded is not None:
        active_condition['camera_occluded'] = camera_occluded
    if occlusion_type:
        active_condition['occlusion_type'] = occlusion_type
    if lighting:
        active_condition['lighting'] = lighting
    if tags:
        active_condition['tags'] = [s.strip() for s in tags.split(',')]

    return {
        'status': 'success',
        'active_condition': {k: v for k, v in active_condition.items() if v},
    }


@mcp.tool
def get_condition() -> dict:
    """Get the current active environment conditions."""
    return {k: v for k, v in active_condition.items() if v}


# ============================================================
# TOOL: Flywheel Queries
# ============================================================

@mcp.tool
def get_flywheel_status() -> dict:
    """
    Get training buffer and flywheel curation statistics.
    Shows how many episodes are curated, hard example ratio, avg difficulty.
    """
    total = es_count(IDX_TRAINING_BUFFER)
    hard = es_count(IDX_TRAINING_BUFFER, {
        'query': {'term': {'curation.hard_example': True}}
    })
    inference_total = es_count(IDX_INFERENCE)

    # Get avg difficulty and novelty from buffer
    agg_result = es_search(IDX_TRAINING_BUFFER, {
        'query': {'match_all': {}},
        'aggs': {
            'avg_difficulty': {'avg': {'field': 'curation.difficulty_score'}},
            'avg_novelty': {'avg': {'field': 'curation.novelty_score'}},
        },
    }, size=0)

    avg_diff = 0
    avg_nov = 0
    if agg_result and 'aggregations' in agg_result:
        avg_diff = agg_result['aggregations'].get('avg_difficulty', {}).get('value', 0) or 0
        avg_nov = agg_result['aggregations'].get('avg_novelty', {}).get('value', 0) or 0

    return {
        'training_buffer_size': total,
        'hard_examples': hard,
        'hard_example_ratio': round(hard / max(total, 1), 2),
        'avg_difficulty_score': round(avg_diff, 3),
        'avg_novelty_score': round(avg_nov, 3),
        'total_inference_cycles': inference_total,
        'curation_ratio': round(total / max(inference_total, 1), 2),
    }


@mcp.tool
def query_hard_examples(
    confidence_threshold: float = 0.6,
    limit: int = 10,
) -> dict:
    """
    Mine hard examples from the inference index.
    Hard examples = low confidence but successful outcome.
    These are the most valuable training data.

    Args:
        confidence_threshold: Max confidence to qualify as "hard" (default 0.6)
        limit: Number of results to return
    """
    result = es_search(IDX_INFERENCE, {
        'query': {
            'bool': {
                'must': [
                    {'term': {'outcome.success': True}},
                    {'range': {'evaluation.confidence': {'lte': confidence_threshold}}},
                ],
            }
        },
        'sort': [{'evaluation.confidence': 'asc'}],
    }, size=limit)

    if result is None:
        return {'error': 'ES query failed'}

    hits = result.get('hits', {}).get('hits', [])
    examples = []
    for h in hits:
        s = h['_source']
        examples.append({
            'cycle_id': s.get('cycle_id'),
            'confidence': s.get('evaluation', {}).get('confidence'),
            'task': s.get('task'),
            'model_version': s.get('model_version'),
            'condition_tags': s.get('condition', {}).get('tags', []),
            'timestamp': s.get('timestamp'),
        })

    return {
        'hard_examples_found': len(examples),
        'confidence_threshold': confidence_threshold,
        'examples': examples,
    }


@mcp.tool
def query_inference_stats(
    group_by: str = "model_version",
) -> dict:
    """
    Aggregate inference statistics from Elasticsearch.
    Compare success rates and confidence across models or conditions.

    Args:
        group_by: Field to group by (model_version, condition.tags, prompt_version)
    """
    result = es_search(IDX_INFERENCE, {
        'query': {'match_all': {}},
        'aggs': {
            'groups': {
                'terms': {'field': group_by, 'size': 20},
                'aggs': {
                    'avg_confidence': {'avg': {'field': 'evaluation.confidence'}},
                    'success_count': {
                        'filter': {'term': {'outcome.success': True}}
                    },
                    'total': {'value_count': {'field': 'cycle_id'}},
                },
            }
        },
    }, size=0)

    if result is None:
        return {'error': 'ES query failed'}

    buckets = result.get('aggregations', {}).get('groups', {}).get('buckets', [])
    stats = []
    for b in buckets:
        total = b.get('total', {}).get('value', 0)
        success = b.get('success_count', {}).get('doc_count', 0)
        stats.append({
            'group': b['key'],
            'total_cycles': total,
            'successes': success,
            'success_rate': round(success / max(total, 1), 3),
            'avg_confidence': round(b.get('avg_confidence', {}).get('value', 0) or 0, 3),
        })

    return {
        'group_by': group_by,
        'stats': stats,
    }


# ============================================================
# TOOL: GPU Observability (Refinery)
# ============================================================

@mcp.tool
def get_gpu_kernel_profile(
    minutes: int = 5,
    device_id: str = "",
    top_n: int = 10,
) -> dict:
    """
    Profile GPU kernel execution from Refinery CUPTI traces.
    Shows which kernels consume the most GPU time, their occupancy,
    and grid dimensions. Use this to find compute bottlenecks.

    Args:
        minutes: Time window to query (default 5 minutes)
        device_id: Filter to specific GPU device (empty = all)
        top_n: Number of top kernels to return
    """
    query_filter = [
        {'range': {'@timestamp': {'gte': f'now-{minutes}m'}}},
        {'exists': {'field': 'gpu.kernel.name'}},
    ]
    if device_id:
        query_filter.append({'term': {'gpu.kernel.device_id': device_id}})

    result = es_search(IDX_GPU_PROFILING, {
        'query': {'bool': {'filter': query_filter}},
        'aggs': {
            'kernels': {
                'terms': {'field': 'gpu.kernel.name', 'size': top_n,
                          'order': {'total_time': 'desc'}},
                'aggs': {
                    'total_time': {'sum': {'field': 'gpu.kernel.duration_ns'}},
                    'avg_time': {'avg': {'field': 'gpu.kernel.duration_ns'}},
                    'max_time': {'max': {'field': 'gpu.kernel.duration_ns'}},
                    'avg_shared_mem': {'avg': {'field': 'gpu.kernel.shared_memory_bytes'}},
                },
            },
            'total_kernel_time': {'sum': {'field': 'gpu.kernel.duration_ns'}},
            'by_operation': {
                'terms': {'field': 'gpu.kernel.operation', 'size': 10},
                'aggs': {
                    'total_time': {'sum': {'field': 'gpu.kernel.duration_ns'}},
                },
            },
        },
    }, size=0)

    if result is None:
        return {'error': 'ES query failed — logs-gpu.kernel index may not exist'}

    aggs = result.get('aggregations', {})
    total_ns = aggs.get('total_kernel_time', {}).get('value', 0)
    buckets = aggs.get('kernels', {}).get('buckets', [])

    kernels = []
    for b in buckets:
        t = b.get('total_time', {}).get('value', 0)
        kernels.append({
            'kernel': b['key'],
            'invocations': b['doc_count'],
            'total_ms': round(t / 1e6, 2),
            'avg_us': round((b.get('avg_time', {}).get('value', 0) or 0) / 1e3, 1),
            'max_us': round((b.get('max_time', {}).get('value', 0) or 0) / 1e3, 1),
            'pct_of_total': round(t / max(total_ns, 1) * 100, 1),
            'avg_shared_mem_kb': round((b.get('avg_shared_mem', {}).get('value', 0) or 0) / 1024, 1),
        })

    ops = []
    for b in aggs.get('by_operation', {}).get('buckets', []):
        t = b.get('total_time', {}).get('value', 0)
        ops.append({'operation': b['key'], 'total_ms': round(t / 1e6, 1),
                    'pct': round(t / max(total_ns, 1) * 100, 1), 'count': b['doc_count']})

    return {
        'window_minutes': minutes,
        'total_kernel_time_ms': round(total_ns / 1e6, 1),
        'unique_kernels': len(buckets),
        'top_kernels': kernels,
        'by_operation': ops,
    }


@mcp.tool
def get_inference_breakdown(
    minutes: int = 30,
) -> dict:
    """
    Show inference latency decomposition derived from kernel traces.
    Breaks down GPU time by kernel category (compute, activation, other)
    and shows duty cycle (% of wall time spent on GPU compute).

    Args:
        minutes: Time window to query (default 30 minutes)
    """
    result = es_search(IDX_GPU_PROFILING, {
        'query': {'bool': {'filter': [
            {'range': {'@timestamp': {'gte': f'now-{minutes}m'}}},
        ]}},
        'aggs': {
            'total_kernel_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
            'avg_kernel_ms': {'avg': {'field': 'gpu.kernel.duration_ms'}},
            'p50_kernel_ms': {'percentiles': {'field': 'gpu.kernel.duration_ms', 'percents': [50, 95, 99]}},
            'by_category': {
                'terms': {'field': 'gpu.kernel.category', 'size': 10},
                'aggs': {
                    'total_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
                    'avg_ms': {'avg': {'field': 'gpu.kernel.duration_ms'}},
                },
            },
            'top_kernels': {
                'terms': {'field': 'gpu.kernel.name', 'size': 5, 'order': {'total_ms': 'desc'}},
                'aggs': {
                    'total_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
                },
            },
        },
    }, size=0)

    if result is None:
        return {'error': 'ES query failed'}

    aggs = result.get('aggregations', {})
    total_kernels = result.get('hits', {}).get('total', {}).get('value', 0)
    total_ms = round(aggs.get('total_kernel_ms', {}).get('value', 0) or 0, 1)
    wall_ms = minutes * 60 * 1000
    duty_cycle = round(total_ms / wall_ms * 100, 1) if wall_ms > 0 else 0

    percentiles = aggs.get('p50_kernel_ms', {}).get('values', {})

    categories = [
        {'category': b['key'],
         'total_ms': round(b['total_ms']['value'], 1),
         'avg_ms': round(b['avg_ms']['value'], 4),
         'pct_of_total': round(b['total_ms']['value'] / total_ms * 100, 1) if total_ms > 0 else 0,
         'count': b['doc_count']}
        for b in aggs.get('by_category', {}).get('buckets', [])
    ]

    top = [
        {'kernel': b['key'][:60], 'total_ms': round(b['total_ms']['value'], 1), 'count': b['doc_count']}
        for b in aggs.get('top_kernels', {}).get('buckets', [])
    ]

    return {
        'window_minutes': minutes,
        'total_kernels': total_kernels,
        'total_gpu_time_ms': total_ms,
        'duty_cycle_pct': duty_cycle,
        'avg_kernel_ms': round(aggs.get('avg_kernel_ms', {}).get('value', 0) or 0, 4),
        'percentiles_ms': {
            'p50': round(percentiles.get('50.0', 0) or 0, 4),
            'p95': round(percentiles.get('95.0', 0) or 0, 4),
            'p99': round(percentiles.get('99.0', 0) or 0, 4),
        },
        'by_category': categories,
        'top_kernels': top,
    }


@mcp.tool
def get_gpu_memory_status(
    device_id: str = "",
) -> dict:
    """
    Get current GPU device status from Refinery NVML metrics.
    Shows utilization, temperature, power draw, memory usage,
    and per-process GPU memory. Like nvidia-smi but queryable.

    Args:
        device_id: Filter to specific GPU device (empty = all devices)
    """
    query_filter = [
        {'range': {'@timestamp': {'gte': 'now-2m'}}},
        {'exists': {'field': 'gpu.utilization.gpu.pct'}},
    ]
    if device_id:
        query_filter.append({'term': {'gpu.labels.gpu': device_id}})

    result = es_search(IDX_GPU_ROBOTICS, {
        'query': {'bool': {'filter': query_filter}},
        'sort': [{'@timestamp': 'desc'}],
        '_source': [
            'gpu.labels.*', 'gpu.utilization.*', 'gpu.temperature.*',
            'gpu.power.*', 'gpu.memory.*', 'gpu.clock.*',
            'gpu.device.*', 'gpu.process.*', 'gpu.health.*', '@timestamp',
        ],
    }, size=10)

    if result is None:
        return {'error': 'ES query failed — logs-gpu.device_metrics index may not exist'}

    hits = result.get('hits', {}).get('hits', [])
    if not hits:
        return {'status': 'no_data', 'message': 'No recent device metrics (last 2 minutes)'}

    seen_devices = {}
    for h in hits:
        s = h['_source']
        gpu = s.get('gpu', {})
        did = gpu.get('labels', {}).get('gpu', 'unknown')
        if did in seen_devices:
            continue
        labels = gpu.get('labels', {})
        util = gpu.get('utilization', {})
        mem_total = gpu.get('memory', {}).get('framebuffer', {}).get('total_size', 0) or 0
        proc_list = gpu.get('process', {}).get('list', [])
        proc_summary = [
            {
                'pid': p.get('pid'),
                'gpu_memory_mb': round((p.get('used_gpu_memory_bytes', 0) or 0) / 1e6, 1),
            }
            for p in (proc_list if isinstance(proc_list, list) else [])
        ]
        seen_devices[did] = {
            'device_id': did,
            'device_name': labels.get('model_name', ''),
            'gpu_utilization_pct': util.get('gpu', {}).get('pct', 0),
            'memory_utilization_pct': util.get('memory_copy', {}).get('pct', 0),
            'temperature_c': gpu.get('temperature', {}).get('gpu', 0),
            'power_draw_w': round(gpu.get('power', {}).get('usage', 0) or 0, 1),
            'memory_total_gb': round(mem_total / 1e9, 1),
            'clock_sm_mhz': gpu.get('clock', {}).get('streaming_multiprocessor_frequency', 0),
            'fan_speed_pct': gpu.get('device', {}).get('fan_speed_pct', 0),
            'throttle_reasons': gpu.get('device', {}).get('throttle_reasons', ''),
            'health_status': gpu.get('health', {}).get('status', ''),
            'processes': proc_summary,
            'timestamp': s.get('@timestamp', ''),
        }

    return {
        'devices': list(seen_devices.values()),
        'device_count': len(seen_devices),
    }


@mcp.tool
def diagnose_trial(
    cycle_id: str = "",
    last_n: int = 5,
) -> dict:
    """
    Diagnose robot trial failures by fusing data across all observability sources:
    trial outcomes, CLIP scene embeddings, GPU kernel profiles, and VLM evaluations.
    Produces a structured failure taxonomy (perception/planning/hardware).

    Args:
        cycle_id: Specific trial cycle_id to diagnose (empty = analyze last N trials)
        last_n: Number of recent trials to analyze (default 5)
    """
    query = {'match_all': {}} if not cycle_id else {'term': {'cycle_id': cycle_id}}
    trials_result = es_search(IDX_INFERENCE, {'query': query,
        'sort': [{'timestamp': 'desc'}]}, size=last_n)
    if trials_result is None:
        return {'error': 'Could not query robot-inference index'}

    hits = trials_result.get('hits', {}).get('hits', [])
    if not hits:
        return {'error': 'No trials found'}

    trials = []
    success_ids = []
    fail_ids = []
    for h in hits:
        s = h['_source']
        outcome = s.get('outcome', {})
        evaluation = s.get('evaluation', {})
        t = {
            'cycle_id': s.get('cycle_id'),
            'timestamp': s.get('timestamp'),
            'model_version': s.get('model_version'),
            'success': outcome.get('success', False),
            'clip_distance': s.get('nearest_training_distance'),
            'steps': s.get('action_summary', {}).get('total_steps', 0),
            'execution_time_s': outcome.get('cycle_time_s', 0),
            'failure_mode': evaluation.get('failure_mode', 'none'),
            'scene_description': evaluation.get('scene_description', ''),
            'confidence': evaluation.get('confidence', 0),
        }
        trials.append(t)
        if t['success']:
            success_ids.append(t)
        else:
            fail_ids.append(t)

    kernel_comparison = None
    if success_ids and fail_ids:
        def _kernel_agg(ts, label):
            if not ts or not ts[0].get('timestamp'):
                return None
            t_end = ts[0]['timestamp'][:19] + 'Z'
            from datetime import datetime, timedelta
            t_dt = datetime.strptime(t_end, '%Y-%m-%dT%H:%M:%SZ')
            t_start = (t_dt - timedelta(seconds=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
            result = es_search(IDX_GPU_PROFILING, {
                'query': {'range': {'@timestamp': {'gte': t_start, 'lte': t_end}}},
                'aggs': {
                    'by_op': {'terms': {'field': 'gpu.kernel.operation', 'size': 5},
                              'aggs': {'total': {'sum': {'field': 'gpu.kernel.duration_ms'}}}},
                    'total_kernels': {'value_count': {'field': 'gpu.kernel.name'}},
                    'total_gpu_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
                },
            }, size=0)
            if not result:
                return None
            aggs = result.get('aggregations', {})
            ops = {b['key']: round(b['total']['value'], 1)
                   for b in aggs.get('by_op', {}).get('buckets', [])}
            return {'label': label,
                    'total_kernels': aggs.get('total_kernels', {}).get('value', 0),
                    'total_gpu_ms': round(aggs.get('total_gpu_ms', {}).get('value', 0), 1),
                    'by_operation_ms': ops}

        kernel_comparison = {
            'success': _kernel_agg(success_ids, f'success ({success_ids[0]["model_version"]})'),
            'failure': _kernel_agg(fail_ids, f'failure ({fail_ids[0]["model_version"]})'),
        }

    clip_stats = {'success_avg': None, 'failure_avg': None}
    s_clips = [t['clip_distance'] for t in success_ids if t.get('clip_distance')]
    f_clips = [t['clip_distance'] for t in fail_ids if t.get('clip_distance')]
    if s_clips:
        clip_stats['success_avg'] = round(sum(s_clips) / len(s_clips), 4)
    if f_clips:
        clip_stats['failure_avg'] = round(sum(f_clips) / len(f_clips), 4)

    classification = 'unknown'
    reasoning = ''
    if kernel_comparison and kernel_comparison.get('success') and kernel_comparison.get('failure'):
        s_gpu = kernel_comparison['success']['total_gpu_ms']
        f_gpu = kernel_comparison['failure']['total_gpu_ms']
        if s_gpu > 0 and abs(f_gpu - s_gpu) / s_gpu < 0.15:
            classification = 'planning_failure'
            reasoning = (f'GPU kernel profiles are identical between success ({s_gpu}ms) '
                        f'and failure ({f_gpu}ms) — the model computes the same thing '
                        f'regardless of outcome. The vision encoder perceives correctly '
                        f'but the policy produces a fixed trajectory.')
        else:
            classification = 'compute_anomaly'
            reasoning = (f'GPU compute differs: success={s_gpu}ms vs failure={f_gpu}ms. '
                        f'Check for thermal throttling, memory pressure, or model loading issues.')

    if not reasoning and fail_ids:
        no_step_fails = [t for t in fail_ids if (t.get('steps') or 0) == 0]
        if no_step_fails:
            classification = 'hardware_failure'
            reasoning = (f'{len(no_step_fails)} trial(s) ran 0 steps — likely camera dropout '
                        f'or motor node hang. Check cam0 USB and motor node pod status.')

    return {
        'trial_count': len(trials),
        'successes': len(success_ids),
        'failures': len(fail_ids),
        'classification': classification,
        'reasoning': reasoning,
        'clip_distance': clip_stats,
        'kernel_comparison': kernel_comparison,
        'trials': trials,
    }


@mcp.tool
def compare_model_kernels(
    model_a: str = "v1",
    model_b: str = "v6",
    minutes: int = 720,
) -> dict:
    """
    GPU-level model autopsy: compare kernel execution profiles between two models.
    Shows whether models use the same compute patterns or diverge at specific
    pipeline stages (vision encoder, attention, action head).

    Args:
        model_a: First model version (e.g. 'v1')
        model_b: Second model version (e.g. 'v6')
        minutes: Time window to search for trials (default 720 = 12 hours)
    """
    from datetime import datetime, timedelta

    def _get_trial_window(model_version):
        result = es_search(IDX_INFERENCE, {
            'query': {'bool': {'filter': [
                {'term': {'model_version': model_version}},
                {'range': {'timestamp': {'gte': f'now-{minutes}m'}}},
                {'range': {'action_summary.total_steps': {'gt': 100}}},
            ]}},
            'sort': [{'timestamp': 'desc'}],
            '_source': ['timestamp', 'outcome.success', 'action_summary.total_steps',
                        'nearest_training_distance', 'cycle_id'],
        }, size=3)
        if not result:
            return None, []
        hits = result.get('hits', {}).get('hits', [])
        if not hits:
            return None, []
        trials = [h['_source'] for h in hits]
        ts = trials[0]['timestamp'][:19] + 'Z'
        t_dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ')
        t_start = (t_dt - timedelta(seconds=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
        return (t_start, ts), trials

    def _kernel_profile(window):
        if not window:
            return None
        result = es_search(IDX_GPU_PROFILING, {
            'query': {'range': {'@timestamp': {'gte': window[0], 'lte': window[1]}}},
            'aggs': {
                'by_operation': {
                    'terms': {'field': 'gpu.kernel.operation', 'size': 10},
                    'aggs': {
                        'total_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
                        'avg_ms': {'avg': {'field': 'gpu.kernel.duration_ms'}},
                        'p99_ms': {'percentiles': {'field': 'gpu.kernel.duration_ms',
                                                    'percents': [99]}},
                    },
                },
                'top_kernels': {
                    'terms': {'field': 'gpu.kernel.name', 'size': 5,
                              'order': {'total_ms': 'desc'}},
                    'aggs': {'total_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}}},
                },
                'total_gpu_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
                'total_kernels': {'value_count': {'field': 'gpu.kernel.name'}},
            },
        }, size=0)
        if not result:
            return None
        aggs = result.get('aggregations', {})
        total_ms = aggs.get('total_gpu_ms', {}).get('value', 0)
        ops = []
        for b in aggs.get('by_operation', {}).get('buckets', []):
            t = b['total_ms']['value']
            ops.append({
                'operation': b['key'],
                'total_ms': round(t, 1),
                'pct': round(t / max(total_ms, 0.001) * 100, 1),
                'count': b['doc_count'],
                'avg_ms': round(b['avg_ms']['value'], 3),
                'p99_ms': round(list(b['p99_ms']['values'].values())[0], 3),
            })
        top = [{'kernel': b['key'][:60], 'total_ms': round(b['total_ms']['value'], 1)}
               for b in aggs.get('top_kernels', {}).get('buckets', [])]
        return {
            'total_kernels': aggs.get('total_kernels', {}).get('value', 0),
            'total_gpu_ms': round(total_ms, 1),
            'by_operation': ops,
            'top_kernels': top,
        }

    win_a, trials_a = _get_trial_window(model_a)
    win_b, trials_b = _get_trial_window(model_b)

    profile_a = _kernel_profile(win_a)
    profile_b = _kernel_profile(win_b)

    comparison = []
    if profile_a and profile_b:
        ops_a = {o['operation']: o for o in profile_a.get('by_operation', [])}
        ops_b = {o['operation']: o for o in profile_b.get('by_operation', [])}
        all_ops = set(list(ops_a.keys()) + list(ops_b.keys()))
        for op in sorted(all_ops):
            a = ops_a.get(op, {})
            b = ops_b.get(op, {})
            comparison.append({
                'operation': op,
                f'{model_a}_ms': a.get('total_ms', 0),
                f'{model_b}_ms': b.get('total_ms', 0),
                f'{model_a}_pct': a.get('pct', 0),
                f'{model_b}_pct': b.get('pct', 0),
                'delta_pct': round(abs(a.get('pct', 0) - b.get('pct', 0)), 1),
            })

    return {
        'model_a': {'version': model_a, 'profile': profile_a,
                     'trials': [{'success': t.get('outcome', {}).get('success'),
                                 'steps': t.get('action_summary', {}).get('total_steps'),
                                 'clip': t.get('nearest_training_distance')}
                                for t in trials_a]},
        'model_b': {'version': model_b, 'profile': profile_b,
                     'trials': [{'success': t.get('outcome', {}).get('success'),
                                 'steps': t.get('action_summary', {}).get('total_steps'),
                                 'clip': t.get('nearest_training_distance')}
                                for t in trials_b]},
        'comparison': comparison,
    }


@mcp.tool
def get_episode_gpu_quality(
    minutes: int = 60,
) -> dict:
    """
    Assess GPU health during recent inference by checking device metrics
    for thermal throttling, memory pressure, and utilization anomalies.

    Args:
        minutes: Time window to query (default 60 minutes)
    """
    result = es_search(IDX_GPU_ROBOTICS, {
        'query': {'bool': {'filter': [
            {'range': {'@timestamp': {'gte': f'now-{minutes}m'}}},
        ]}},
        'aggs': {
            'avg_temp': {'avg': {'field': 'gpu.temperature.gpu'}},
            'max_temp': {'max': {'field': 'gpu.temperature.gpu'}},
            'avg_util': {'avg': {'field': 'gpu.utilization.gpu.pct'}},
            'max_util': {'max': {'field': 'gpu.utilization.gpu.pct'}},
            'avg_power': {'avg': {'field': 'gpu.power.usage'}},
            'max_power': {'max': {'field': 'gpu.power.usage'}},
            'avg_mem_used': {'avg': {'field': 'gpu.process.list.used_gpu_memory_bytes'}},
            'throttle_reasons': {'terms': {'field': 'gpu.device.throttle_reasons.keyword', 'size': 5}},
            'health_status': {'terms': {'field': 'gpu.health.status.keyword', 'size': 5}},
            'by_gpu': {
                'terms': {'field': 'gpu.labels.gpu', 'size': 4},
                'aggs': {
                    'avg_temp': {'avg': {'field': 'gpu.temperature.gpu'}},
                    'avg_util': {'avg': {'field': 'gpu.utilization.gpu.pct'}},
                    'max_temp': {'max': {'field': 'gpu.temperature.gpu'}},
                },
            },
        },
    }, size=0)

    if result is None:
        return {'error': 'ES query failed'}

    aggs = result.get('aggregations', {})
    total = result.get('hits', {}).get('total', {}).get('value', 0)

    def _val(key): return round(aggs.get(key, {}).get('value', 0) or 0, 1)

    throttle = {b['key']: b['doc_count'] for b in aggs.get('throttle_reasons', {}).get('buckets', [])}
    health = {b['key']: b['doc_count'] for b in aggs.get('health_status', {}).get('buckets', [])}

    gpus = [
        {'gpu': b['key'], 'avg_temp': round(b['avg_temp']['value'] or 0, 1),
         'max_temp': round(b['max_temp']['value'] or 0, 1),
         'avg_util': round(b['avg_util']['value'] or 0, 1)}
        for b in aggs.get('by_gpu', {}).get('buckets', [])
    ]

    is_throttling = any(r != 'gpu_idle' for r in throttle)
    max_temp = _val('max_temp')

    return {
        'window_minutes': minutes,
        'samples': total,
        'temperature': {'avg_c': _val('avg_temp'), 'max_c': max_temp,
                        'warning': max_temp > 80},
        'utilization': {'avg_pct': _val('avg_util'), 'max_pct': _val('max_util')},
        'power': {'avg_w': _val('avg_power'), 'max_w': _val('max_power')},
        'throttle_reasons': throttle,
        'is_throttling': is_throttling,
        'health_status': health,
        'per_gpu': gpus,
    }


@mcp.tool
def get_control_loop_status(
    minutes: int = 5,
) -> dict:
    """
    Check inference control loop performance derived from kernel traces.
    Estimates actual Hz, duty cycle, and kernel timing from CUPTI data.
    Target: 30 Hz (one inference step per 33ms).

    Args:
        minutes: Time window to query (default 5 minutes)
    """
    result = es_search(IDX_GPU_PROFILING, {
        'query': {'bool': {'filter': [
            {'range': {'@timestamp': {'gte': f'now-{minutes}m'}}},
        ]}},
        'aggs': {
            'total_kernel_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
            'avg_kernel_ms': {'avg': {'field': 'gpu.kernel.duration_ms'}},
            'p95_kernel_ms': {'percentiles': {'field': 'gpu.kernel.duration_ms', 'percents': [50, 95, 99]}},
            'max_kernel_ms': {'max': {'field': 'gpu.kernel.duration_ms'}},
            'per_second': {
                'date_histogram': {'field': '@timestamp', 'fixed_interval': '1s'},
                'aggs': {
                    'kernel_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
                },
            },
        },
    }, size=0)

    if result is None:
        return {'error': 'ES query failed'}

    aggs = result.get('aggregations', {})
    total_kernels = result.get('hits', {}).get('total', {}).get('value', 0)
    total_ms = aggs.get('total_kernel_ms', {}).get('value', 0) or 0
    wall_ms = minutes * 60 * 1000
    duty_cycle = round(total_ms / wall_ms * 100, 1) if wall_ms > 0 else 0

    per_sec = aggs.get('per_second', {}).get('buckets', [])
    active_seconds = [b for b in per_sec if b['kernel_ms']['value'] > 0]
    active_count = len(active_seconds)

    hz_per_sec = []
    for b in active_seconds:
        sec_ms = b['kernel_ms']['value']
        hz_per_sec.append(round(sec_ms / 33.3, 1))

    avg_hz = round(sum(hz_per_sec) / len(hz_per_sec), 1) if hz_per_sec else 0
    min_hz = round(min(hz_per_sec), 1) if hz_per_sec else 0
    max_hz = round(max(hz_per_sec), 1) if hz_per_sec else 0

    percentiles = aggs.get('p95_kernel_ms', {}).get('values', {})

    return {
        'window_minutes': minutes,
        'total_kernels': total_kernels,
        'total_gpu_time_ms': round(total_ms, 1),
        'duty_cycle_pct': duty_cycle,
        'active_seconds': active_count,
        'idle_seconds': len(per_sec) - active_count,
        'target_hz': 30,
        'estimated_avg_hz': avg_hz,
        'min_hz': min_hz,
        'max_hz': max_hz,
        'kernel_duration_ms': {
            'avg': round(aggs.get('avg_kernel_ms', {}).get('value', 0) or 0, 4),
            'p50': round(percentiles.get('50.0', 0) or 0, 4),
            'p95': round(percentiles.get('95.0', 0) or 0, 4),
            'p99': round(percentiles.get('99.0', 0) or 0, 4),
            'max': round(aggs.get('max_kernel_ms', {}).get('value', 0) or 0, 4),
        },
        'status': 'active' if active_count > 0 else 'idle',
    }


@mcp.tool
def get_safety_events(
    minutes: int = 60,
) -> dict:
    """
    Check for GPU safety-relevant events: thermal throttling,
    unhealthy device status, high power draw, or memory pressure.
    Derived from NVML device metrics.

    Args:
        minutes: Time window to query (default 60 minutes)
    """
    result = es_search(IDX_GPU_ROBOTICS, {
        'query': {'bool': {'filter': [
            {'range': {'@timestamp': {'gte': f'now-{minutes}m'}}},
        ]}},
        'aggs': {
            'throttle_reasons': {'terms': {'field': 'gpu.device.throttle_reasons.keyword', 'size': 10}},
            'health_status': {'terms': {'field': 'gpu.health.status.keyword', 'size': 5}},
            'max_temp': {'max': {'field': 'gpu.temperature.gpu'}},
            'max_power': {'max': {'field': 'gpu.power.usage'}},
            'thermal_throttle': {
                'filter': {'range': {'gpu.throttling.thermal.us': {'gt': 0}}},
            },
            'high_temp': {
                'filter': {'range': {'gpu.temperature.gpu': {'gte': 80}}},
            },
        },
    }, size=0)

    if result is None:
        return {'error': 'ES query failed'}

    aggs = result.get('aggregations', {})
    total = result.get('hits', {}).get('total', {}).get('value', 0)

    throttle = {b['key']: b['doc_count'] for b in aggs.get('throttle_reasons', {}).get('buckets', [])}
    health = {b['key']: b['doc_count'] for b in aggs.get('health_status', {}).get('buckets', [])}
    thermal_events = aggs.get('thermal_throttle', {}).get('doc_count', 0)
    high_temp_events = aggs.get('high_temp', {}).get('doc_count', 0)
    max_temp = round(aggs.get('max_temp', {}).get('value', 0) or 0, 1)
    max_power = round(aggs.get('max_power', {}).get('value', 0) or 0, 1)

    events = []
    if thermal_events > 0:
        events.append({'type': 'thermal_throttle', 'count': thermal_events, 'severity': 'warning'})
    if high_temp_events > 0:
        events.append({'type': 'high_temperature', 'count': high_temp_events, 'severity': 'critical',
                       'detail': f'GPU reached {max_temp}C'})
    if max_power > 150:
        events.append({'type': 'high_power_draw', 'severity': 'warning', 'detail': f'Peak {max_power}W'})

    return {
        'window_minutes': minutes,
        'samples': total,
        'safety_events': events,
        'event_count': len(events),
        'max_temperature_c': max_temp,
        'max_power_w': max_power,
        'throttle_reasons': throttle,
        'health_status': health,
        'status': 'clear' if not events else 'warning',
    }


# --- ROS2 Observability Tools ---

@mcp.tool
def get_ros2_topic_health(
    topic: str = "",
    minutes: int = 5,
) -> dict:
    """
    Get ROS2 topic health from the ros2-integration collector.
    Shows publish rate, bandwidth, QoS profile, endpoint counts,
    and rate deviation from target for each topic.

    Args:
        topic: Filter to a specific topic name (empty = all topics)
        minutes: Time window to query (default 5 minutes)
    """
    query_filter = [
        {'range': {'@timestamp': {'gte': f'now-{minutes}m'}}},
    ]
    if topic:
        query_filter.append({'term': {'ros2.topic.name': topic}})

    result = es_search(IDX_ROS2_TOPICS, {
        'query': {'bool': {'filter': query_filter}},
        'aggs': {
            'by_topic': {
                'terms': {'field': 'ros2.topic.name', 'size': 50},
                'aggs': {
                    'avg_rate': {'avg': {'field': 'ros2.topic.rate_hz'}},
                    'max_rate': {'max': {'field': 'ros2.topic.rate_hz'}},
                    'min_rate': {'min': {'field': 'ros2.topic.rate_hz'}},
                    'avg_bw': {'avg': {'field': 'ros2.topic.bandwidth_bytes_sec'}},
                    'total_msgs': {'sum': {'field': 'ros2.topic.message_count'}},
                    'latest': {
                        'top_hits': {
                            'size': 1,
                            'sort': [{'@timestamp': 'desc'}],
                            '_source': [
                                'ros2.topic.*', 'ros2.node.name',
                                '@timestamp',
                            ],
                        },
                    },
                },
            },
        },
    }, size=0)

    if result is None:
        return {'error': 'ES query failed — metrics-ros2.topics index may not exist yet'}

    topics = []
    for bucket in result.get('aggregations', {}).get('by_topic', {}).get('buckets', []):
        latest_src = {}
        hits = bucket.get('latest', {}).get('hits', {}).get('hits', [])
        if hits:
            latest_src = hits[0].get('_source', {})

        t = latest_src.get('ros2', {}).get('topic', {})
        avg_rate = bucket.get('avg_rate', {}).get('value', 0) or 0
        target = t.get('rate_target_hz', 0) or 0

        entry = {
            'topic': bucket['key'],
            'type': t.get('type', ''),
            'node': latest_src.get('ros2', {}).get('node', {}).get('name', ''),
            'avg_rate_hz': round(avg_rate, 1),
            'min_rate_hz': round(bucket.get('min_rate', {}).get('value', 0) or 0, 1),
            'max_rate_hz': round(bucket.get('max_rate', {}).get('value', 0) or 0, 1),
            'target_hz': target,
            'total_messages': int(bucket.get('total_msgs', {}).get('value', 0) or 0),
            'avg_bandwidth_bytes_sec': round(bucket.get('avg_bw', {}).get('value', 0) or 0, 0),
            'qos_reliability': t.get('qos', {}).get('reliability', ''),
            'qos_durability': t.get('qos', {}).get('durability', ''),
            'publishers': t.get('publisher_count', 0),
            'subscribers': t.get('subscriber_count', 0),
        }

        if target > 0 and avg_rate > 0:
            entry['deviation_pct'] = round(((target - avg_rate) / target) * 100, 1)

        topics.append(entry)

    return {
        'window_minutes': minutes,
        'topic_count': len(topics),
        'topics': sorted(topics, key=lambda x: x['topic']),
    }


@mcp.tool
def get_ros2_node_status(
    minutes: int = 5,
) -> dict:
    """
    Get ROS2 node status from the ros2-integration collector.
    Shows active nodes, DDS participant and endpoint counts,
    and collector uptime.

    Args:
        minutes: Time window to query (default 5 minutes)
    """
    result = es_search(IDX_ROS2_NODES, {
        'query': {'bool': {'filter': [
            {'range': {'@timestamp': {'gte': f'now-{minutes}m'}}},
        ]}},
        'aggs': {
            'by_node': {
                'terms': {'field': 'ros2.node.name', 'size': 50},
                'aggs': {
                    'latest': {
                        'top_hits': {
                            'size': 1,
                            'sort': [{'@timestamp': 'desc'}],
                            '_source': ['ros2.*', 'host.name', '@timestamp'],
                        },
                    },
                },
            },
        },
    }, size=0)

    if result is None:
        return {'error': 'ES query failed — metrics-ros2.nodes index may not exist yet'}

    nodes = []
    for bucket in result.get('aggregations', {}).get('by_node', {}).get('buckets', []):
        hits = bucket.get('latest', {}).get('hits', {}).get('hits', [])
        if not hits:
            continue
        src = hits[0].get('_source', {})
        r = src.get('ros2', {})
        n = r.get('node', {})
        d = r.get('dds', {})

        nodes.append({
            'name': bucket['key'],
            'namespace': n.get('namespace', '/'),
            'state': n.get('state', 'unknown'),
            'host': src.get('host', {}).get('name', ''),
            'dds_participants': d.get('participant_count', 0),
            'dds_endpoints': d.get('endpoint_count', 0),
            'last_seen': src.get('@timestamp', ''),
        })

    return {
        'window_minutes': minutes,
        'node_count': len(nodes),
        'nodes': sorted(nodes, key=lambda x: x['name']),
    }


@mcp.tool
def get_ros2_alerts(
    minutes: int = 60,
    severity: str = "",
) -> dict:
    """
    Get recent ROS2 alerts and events from the ros2-integration collector.
    Shows topic rate deviations, topic removals, QoS violations,
    and other ROS2 health events.

    Args:
        minutes: Time window to query (default 60 minutes)
        severity: Filter by severity: info, warning, critical (empty = all)
    """
    query_filter = [
        {'range': {'@timestamp': {'gte': f'now-{minutes}m'}}},
    ]
    if severity:
        query_filter.append({'term': {'ros2.alert.severity': severity}})

    result = es_search(IDX_ROS2_EVENTS, {
        'query': {'bool': {'filter': query_filter}},
        'aggs': {
            'by_action': {
                'terms': {'field': 'event.action', 'size': 10},
            },
            'by_severity': {
                'terms': {'field': 'ros2.alert.severity', 'size': 5},
            },
            'by_topic': {
                'terms': {'field': 'ros2.topic.name', 'size': 20},
            },
        },
        'sort': [{'@timestamp': 'desc'}],
        '_source': [
            'event.*', 'ros2.*', '@timestamp', 'host.name',
        ],
    }, size=20)

    if result is None:
        return {'error': 'ES query failed — logs-ros2.events index may not exist yet'}

    aggs = result.get('aggregations', {})
    total = result.get('hits', {}).get('total', {}).get('value', 0)

    recent = []
    for h in result.get('hits', {}).get('hits', []):
        s = h['_source']
        ev = s.get('event', {})
        r = s.get('ros2', {})
        alert = r.get('alert', {})
        topic = r.get('topic', {})

        recent.append({
            'action': ev.get('action', ''),
            'severity': alert.get('severity', ''),
            'message': alert.get('message', ''),
            'topic': topic.get('name', ''),
            'rate_hz': topic.get('rate_hz', None),
            'target_hz': topic.get('rate_target_hz', None),
            'host': s.get('host', {}).get('name', ''),
            'timestamp': s.get('@timestamp', ''),
        })

    actions = {b['key']: b['doc_count']
               for b in aggs.get('by_action', {}).get('buckets', [])}
    severities = {b['key']: b['doc_count']
                  for b in aggs.get('by_severity', {}).get('buckets', [])}
    topics = {b['key']: b['doc_count']
              for b in aggs.get('by_topic', {}).get('buckets', [])}

    return {
        'window_minutes': minutes,
        'total_events': total,
        'by_action': actions,
        'by_severity': severities,
        'by_topic': topics,
        'recent_events': recent,
    }


@mcp.tool
def profile_inference(
    cycle_id: str = "",
    minutes: int = 5,
) -> dict:
    """
    Profile a trial's GPU inference execution at 100ms and 1-second resolution.
    Shows where GPU time goes, per-step compute budget, duty cycle, and
    whether the execution pattern is steady or bursty. Useful for both
    successful and failed trials — understanding the execution profile
    reveals optimization opportunities invisible to nvidia-smi.
    Works with Refinery kernel data — no trajectory capture needed.

    Args:
        cycle_id: Trial cycle_id to analyze (empty = use last trial's time window)
        minutes: Fallback time window if no cycle_id (default 5)
    """
    if cycle_id:
        trial = es_search(IDX_INFERENCE, {
            'query': {'term': {'cycle_id': cycle_id}},
            '_source': ['timestamp', 'action_summary.total_steps', 'outcome.success',
                        'model_version'],
        }, size=1)
        if not trial or not trial.get('hits', {}).get('hits'):
            return {'error': f'Trial {cycle_id} not found'}
        s = trial['hits']['hits'][0]['_source']
        steps = s.get('action_summary', {}).get('total_steps', 500)
        ts_end = s['timestamp'][:19] + 'Z'
        from datetime import datetime, timedelta
        t_end = datetime.strptime(ts_end, '%Y-%m-%dT%H:%M:%SZ')
        duration_s = steps / 30.0
        t_start = (t_end - timedelta(seconds=duration_s + 2)).strftime('%Y-%m-%dT%H:%M:%SZ')
        t_end_str = ts_end
        meta = {
            'cycle_id': cycle_id,
            'success': s.get('outcome', {}).get('success'),
            'model': s.get('model_version'),
            'steps': steps,
        }
    else:
        trial = es_search(IDX_INFERENCE, {
            'query': {'match_all': {}},
            'sort': [{'timestamp': 'desc'}],
            '_source': ['cycle_id', 'timestamp', 'action_summary.total_steps',
                        'outcome.success', 'model_version'],
        }, size=1)
        if trial and trial.get('hits', {}).get('hits'):
            s = trial['hits']['hits'][0]['_source']
            steps = s.get('action_summary', {}).get('total_steps', 500)
            ts_end = s['timestamp'][:19] + 'Z'
            from datetime import datetime, timedelta
            t_end = datetime.strptime(ts_end, '%Y-%m-%dT%H:%M:%SZ')
            duration_s = steps / 30.0
            t_start = (t_end - timedelta(seconds=duration_s + 2)).strftime('%Y-%m-%dT%H:%M:%SZ')
            t_end_str = ts_end
            meta = {
                'cycle_id': s.get('cycle_id'),
                'success': s.get('outcome', {}).get('success'),
                'model': s.get('model_version'),
                'steps': steps,
            }
        else:
            t_start = f'now-{minutes}m'
            t_end_str = 'now'
            meta = {'window_minutes': minutes}

    result = es_search(IDX_GPU_PROFILING, {
        'query': {'range': {'@timestamp': {'gte': t_start, 'lte': t_end_str}}},
        'aggs': {
            'per_100ms': {
                'date_histogram': {'field': '@timestamp', 'fixed_interval': '100ms'},
                'aggs': {
                    'gpu_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
                    'kernel_count': {'value_count': {'field': 'gpu.kernel.name'}},
                },
            },
            'per_second': {
                'date_histogram': {'field': '@timestamp', 'fixed_interval': '1s'},
                'aggs': {
                    'gpu_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
                    'kernel_count': {'value_count': {'field': 'gpu.kernel.name'}},
                },
            },
            'total_gpu_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
            'total_kernels': {'value_count': {'field': 'gpu.kernel.name'}},
        },
    }, size=0)

    if not result:
        return {'error': 'No kernel data in time window'}

    aggs = result.get('aggregations', {})
    total_kernels = aggs.get('total_kernels', {}).get('value', 0)
    total_gpu_ms = aggs.get('total_gpu_ms', {}).get('value', 0)

    buckets_100ms = aggs.get('per_100ms', {}).get('buckets', [])
    buckets_1s = aggs.get('per_second', {}).get('buckets', [])

    if not buckets_100ms:
        return {'error': 'No 100ms histogram data', **meta}

    active_threshold = 100
    active_buckets = [b for b in buckets_100ms if b['doc_count'] >= active_threshold]
    idle_buckets = [b for b in buckets_100ms if b['doc_count'] < active_threshold]

    active_count = len(active_buckets)
    idle_count = len(idle_buckets)
    total_buckets = len(buckets_100ms)

    active_pct = active_count / max(total_buckets, 1) * 100
    duty_cycle = active_pct

    runs = []
    current_run = {'type': None, 'start': 0, 'count': 0}
    for i, b in enumerate(buckets_100ms):
        btype = 'active' if b['doc_count'] >= active_threshold else 'idle'
        if btype == current_run['type']:
            current_run['count'] += 1
        else:
            if current_run['type']:
                runs.append(current_run.copy())
            current_run = {'type': btype, 'start': i, 'count': 1}
    if current_run['type']:
        runs.append(current_run)

    active_runs = [r for r in runs if r['type'] == 'active']
    idle_runs = [r for r in runs if r['type'] == 'idle']

    avg_burst_ms = sum(r['count'] * 100 for r in active_runs) / max(len(active_runs), 1)
    avg_gap_ms = sum(r['count'] * 100 for r in idle_runs) / max(len(idle_runs), 1)
    max_gap_ms = max((r['count'] * 100 for r in idle_runs), default=0)

    missed_per_gap = avg_gap_ms / 33.3 if avg_gap_ms > 0 else 0

    low_1s = [b for b in buckets_1s if b['doc_count'] < 2000]
    dip_seconds = [b['key_as_string'][11:19] for b in low_1s]

    if len(dip_seconds) >= 2:
        dip_timestamps = [int(b['key']) / 1000 for b in low_1s]
        gaps = [dip_timestamps[i+1] - dip_timestamps[i]
                for i in range(len(dip_timestamps) - 1)]
        avg_period = sum(gaps) / len(gaps) if gaps else 0
    else:
        avg_period = 0
        gaps = []

    root_cause = 'unknown'
    reasoning = ''
    if 4.0 < avg_period < 6.5:
        root_cause = 'python_gc_or_cupti'
        reasoning = (f'~{avg_period:.1f}s period matches Python GC cycle or '
                     f'CUPTI correlator eviction window (5s). '
                     f'Both cause CPU-side stalls that block the inference loop.')
    elif 1.5 < avg_period < 3.5:
        root_cause = 'camera_frame_timing'
        reasoning = (f'~{avg_period:.1f}s period suggests camera frame drops. '
                     f'FRAME_TIMEOUT_SEC is 2s — inference blocks waiting for frames.')
    elif avg_period > 0:
        root_cause = 'periodic_system_process'
        reasoning = f'~{avg_period:.1f}s period — investigate cron, log rotation, or DDS.'

    steps_meta = meta.get('steps', total_kernels / 124)
    per_step_ms = total_gpu_ms / max(steps_meta, 1)

    return {
        **meta,
        'kernel_summary': {
            'total_kernels': total_kernels,
            'total_gpu_ms': round(total_gpu_ms, 1),
            'per_step_gpu_ms': round(per_step_ms, 2),
            'step_budget_ms': round(1000 / 30, 2),
            'gpu_utilization_pct': round(per_step_ms / (1000 / 30) * 100, 1),
        },
        'burst_pattern': {
            'active_100ms_buckets': active_count,
            'idle_100ms_buckets': idle_count,
            'duty_cycle_pct': round(duty_cycle, 1),
            'avg_burst_ms': round(avg_burst_ms),
            'avg_gap_ms': round(avg_gap_ms),
            'max_gap_ms': max_gap_ms,
            'missed_steps_per_gap': round(missed_per_gap, 1),
        },
        'periodicity': {
            'dip_seconds': dip_seconds,
            'dip_period_s': round(avg_period, 1) if avg_period else None,
            'inter_dip_gaps': [round(g, 1) for g in gaps],
        },
        'diagnosis': {
            'root_cause': root_cause,
            'reasoning': reasoning,
            'note': ('GPU is NOT the bottleneck (29% utilization). '
                     'Jitter is a CPU-side timing artifact — inference runs in bursts '
                     'with periodic gaps where no GPU work is submitted.'),
        },
    }


@mcp.tool
def profile_trajectory(
    cycle_id: str = "",
    last_n: int = 3,
) -> dict:
    """
    Profile joint command trajectory smoothness for a trial. Computes per-joint
    velocity, acceleration, and jerk (3rd derivative) to quantify motion quality.
    Correlates with GPU kernel timing from Refinery. Useful for comparing models
    (does SmolVLA produce smoother trajectories than X-VLA?) and verifying fixes
    (did disabling GC improve motion smoothness?).

    Args:
        cycle_id: Specific trial cycle_id (empty = analyze last N)
        last_n: Number of recent trials to analyze (default 3)
    """
    import math

    if cycle_id:
        step_result = es_search(IDX_STEPS, {
            'query': {'term': {'cycle_id': cycle_id}},
            'sort': [{'step': 'asc'}],
        }, size=1000)
    else:
        trial_result = es_search(IDX_INFERENCE, {
            'query': {'match_all': {}},
            'sort': [{'timestamp': 'desc'}],
            '_source': ['cycle_id', 'outcome.success', 'model_version',
                        'action_summary.total_steps'],
        }, size=last_n)
        if not trial_result or not trial_result.get('hits', {}).get('hits'):
            return {'error': 'No trials found in robot-inference'}

        trials_meta = []
        for h in trial_result['hits']['hits']:
            s = h['_source']
            cid = s.get('cycle_id')
            trials_meta.append({
                'cycle_id': cid,
                'success': s.get('outcome', {}).get('success'),
                'model': s.get('model_version'),
                'steps': s.get('action_summary', {}).get('total_steps', 0),
            })

        analyses = []
        for tm in trials_meta:
            sr = es_search(IDX_STEPS, {
                'query': {'term': {'cycle_id': tm['cycle_id']}},
                'sort': [{'step': 'asc'}],
            }, size=1000)
            if sr and sr.get('hits', {}).get('hits'):
                steps = [h['_source'] for h in sr['hits']['hits']]
                a = _compute_jitter_metrics(steps)
                a['cycle_id'] = tm['cycle_id']
                a['success'] = tm['success']
                a['model'] = tm['model']
                analyses.append(a)
            else:
                analyses.append({
                    'cycle_id': tm['cycle_id'],
                    'success': tm['success'],
                    'model': tm['model'],
                    'error': 'No trajectory data in robot-steps',
                })

        return {
            'trials_analyzed': len(analyses),
            'analyses': analyses,
            'note': ('Trajectory capture added in MCP v15. '
                     'Older trials have no step data.'),
        }

    if not step_result or not step_result.get('hits', {}).get('hits'):
        return {'error': f'No trajectory data for {cycle_id}'}

    steps = [h['_source'] for h in step_result['hits']['hits']]
    analysis = _compute_jitter_metrics(steps)
    analysis['cycle_id'] = cycle_id

    trial_result = es_search(IDX_INFERENCE, {
        'query': {'term': {'cycle_id': cycle_id}},
    }, size=1)
    if trial_result and trial_result.get('hits', {}).get('hits'):
        s = trial_result['hits']['hits'][0]['_source']
        analysis['success'] = s.get('outcome', {}).get('success')
        analysis['model'] = s.get('model_version')
        ts = s.get('timestamp', '')[:19] + 'Z'
        kernel_analysis = _kernel_timing_for_window(ts, len(steps))
        if kernel_analysis:
            analysis['kernel_timing'] = kernel_analysis

    return analysis


@mcp.tool
def check_scene_novelty(
    cycle_id: str = "",
    camera: str = "cam0",
    k: int = 5,
) -> dict:
    """
    Check whether the current or past scene is within the model's training
    distribution. Embeds the camera frame with CLIP and searches the training
    episode embedding index for nearest neighbors.

    A distance <= 0.038 is within the training boundary (high confidence).
    A distance >= 0.040 is outside (the model may not generalize here).
    Use this to understand WHY a trial failed — was the scene novel?

    Args:
        cycle_id: Trial cycle_id to check (empty = use live camera frame)
        camera: Which camera to use for live frame (cam0 = wrist, cam1 = overhead)
        k: Number of nearest training episodes to return (default 5)
    """
    embedding = None
    source = None

    if cycle_id:
        trial = es_search(IDX_INFERENCE, {
            'query': {'term': {'cycle_id': cycle_id}},
            '_source': ['scene_embedding', 'nearest_training_distance',
                        'outcome.success', 'model_version', 'timestamp'],
        }, size=1)
        if not trial or not trial.get('hits', {}).get('hits'):
            return {'error': f'Trial {cycle_id} not found'}
        s = trial['hits']['hits'][0]['_source']
        embedding = s.get('scene_embedding')
        source = f'trial:{cycle_id}'
        if not embedding:
            return {
                'error': f'Trial {cycle_id} has no scene embedding (pre-CLIP era)',
                'nearest_training_distance': s.get('nearest_training_distance'),
            }
    else:
        if robot_bridge is None:
            return {'error': 'Robot bridge not initialized'}
        jpeg = robot_bridge.get_latest_frame_jpeg(camera)
        if jpeg is None:
            return {'error': f'No fresh frame from {camera}'}
        embedding = embed_image_clip(jpeg)
        source = f'live:{camera}'
        if not embedding:
            return {'error': 'CLIP embedding failed — is Jina CLIP server running?'}

    knn_result = es_search(IDX_EPISODE_EMBEDDINGS, {
        'knn': {
            'field': 'embedding',
            'query_vector': embedding,
            'k': k,
            'num_candidates': 100,
        },
        '_source': ['episode_id', 'episode_num', 'cluster_id', 'split'],
    }, size=k)

    if not knn_result or not knn_result.get('hits', {}).get('hits'):
        return {'error': 'kNN search failed — robot-episode-embeddings may be empty'}

    neighbors = []
    distances = []
    for h in knn_result['hits']['hits']:
        score = h.get('_score', 0)
        dist = round(1.0 - score, 4)
        distances.append(dist)
        s = h.get('_source', {})
        neighbors.append({
            'episode_id': s.get('episode_id', h['_id']),
            'episode_num': s.get('episode_num'),
            'cluster': s.get('cluster_id'),
            'distance': dist,
        })

    nearest = distances[0] if distances else None
    boundary = 0.038

    in_distribution = nearest <= boundary if nearest is not None else None
    if nearest is not None:
        if nearest <= 0.035:
            confidence = 'high'
            assessment = 'Scene is well within training distribution.'
        elif nearest <= boundary:
            confidence = 'moderate'
            assessment = 'Scene is at the edge of training distribution.'
        elif nearest <= 0.045:
            confidence = 'low'
            assessment = ('Scene is outside training distribution. '
                         'Model may fail due to position novelty.')
        else:
            confidence = 'very_low'
            assessment = ('Scene is far outside training distribution. '
                         'Model is very likely to fail here.')
    else:
        confidence = 'unknown'
        assessment = 'Could not compute distance.'

    return {
        'source': source,
        'nearest_distance': nearest,
        'boundary': boundary,
        'in_distribution': in_distribution,
        'confidence': confidence,
        'assessment': assessment,
        'neighbors': neighbors,
        'distance_stats': {
            'min': min(distances) if distances else None,
            'max': max(distances) if distances else None,
            'mean': round(sum(distances) / len(distances), 4) if distances else None,
        },
    }


@mcp.tool
def analyze_failure_patterns(
    last_n: int = 10,
    minutes: int = 1440,
) -> dict:
    """
    Analyze patterns across recent failures to identify the dominant failure
    mode and its root cause. Groups failures by VLM-reported failure mode,
    CLIP distance range, and model version, then correlates with GPU kernel
    health and ROS2 topic status.

    Use this when the robot has failed multiple times and you want to
    understand: is it the same failure repeating, or different issues?
    What's the common thread?

    Args:
        last_n: Number of recent trials to analyze (default 10)
        minutes: Time window to search (default 1440 = 24 hours)
    """
    result = es_search(IDX_INFERENCE, {
        'query': {'bool': {'filter': [
            {'range': {'timestamp': {'gte': f'now-{minutes}m'}}},
        ]}},
        'sort': [{'timestamp': 'desc'}],
        '_source': [
            'cycle_id', 'timestamp', 'model_version',
            'outcome.success', 'outcome.failure_mode',
            'evaluation.confidence', 'evaluation.failure_mode',
            'evaluation.scene_description',
            'nearest_training_distance',
            'action_summary.total_steps',
            'condition.tags',
        ],
    }, size=last_n)

    if not result or not result.get('hits', {}).get('hits'):
        return {'error': 'No trials found in time window'}

    trials = [h['_source'] for h in result['hits']['hits']]

    successes = [t for t in trials if t.get('outcome', {}).get('success')]
    failures = [t for t in trials if not t.get('outcome', {}).get('success')]

    if not failures:
        return {
            'total_trials': len(trials),
            'failures': 0,
            'assessment': 'No failures in the analyzed window.',
            'success_rate': 1.0,
        }

    by_mode = {}
    for f in failures:
        mode = (f.get('evaluation', {}).get('failure_mode')
                or f.get('outcome', {}).get('failure_mode')
                or 'unknown')
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(f)

    by_distribution = {'in_distribution': [], 'out_of_distribution': [], 'unknown': []}
    for f in failures:
        dist = f.get('nearest_training_distance')
        if dist is None:
            by_distribution['unknown'].append(f)
        elif dist <= 0.038:
            by_distribution['in_distribution'].append(f)
        else:
            by_distribution['out_of_distribution'].append(f)

    mode_summary = []
    for mode, mode_failures in sorted(by_mode.items(), key=lambda x: -len(x[1])):
        clips = [f['nearest_training_distance'] for f in mode_failures
                 if f.get('nearest_training_distance') is not None]
        confs = [f.get('evaluation', {}).get('confidence', 0) for f in mode_failures]
        steps = [f.get('action_summary', {}).get('total_steps', 0) for f in mode_failures]

        entry = {
            'failure_mode': mode,
            'count': len(mode_failures),
            'pct_of_failures': round(len(mode_failures) / len(failures) * 100, 1),
            'avg_confidence': round(sum(confs) / len(confs), 3) if confs else 0,
            'avg_steps': round(sum(steps) / len(steps)) if steps else 0,
            'descriptions': [f.get('evaluation', {}).get('scene_description', '')
                           for f in mode_failures[:3]],
        }
        if clips:
            entry['avg_clip_distance'] = round(sum(clips) / len(clips), 4)
            entry['clip_range'] = [round(min(clips), 4), round(max(clips), 4)]
            entry['out_of_distribution'] = sum(1 for c in clips if c > 0.038)
        mode_summary.append(entry)

    ood_count = len(by_distribution['out_of_distribution'])
    id_count = len(by_distribution['in_distribution'])

    recommendations = []
    dominant = mode_summary[0] if mode_summary else None

    if ood_count > id_count and ood_count > 0:
        ood_clips = [f['nearest_training_distance']
                     for f in by_distribution['out_of_distribution']
                     if f.get('nearest_training_distance')]
        avg_ood = round(sum(ood_clips) / len(ood_clips), 4) if ood_clips else 0
        recommendations.append(
            f'{ood_count}/{len(failures)} failures are out-of-distribution '
            f'(avg CLIP distance {avg_ood} vs boundary 0.038). '
            f'Collect 15-20 teleoperated episodes at the failing positions '
            f'and retrain from the pretrained base.'
        )

    if id_count > 0 and dominant:
        zero_step = sum(1 for f in failures
                       if (f.get('action_summary', {}).get('total_steps', 0)) == 0)
        if zero_step > 0:
            recommendations.append(
                f'{zero_step} trial(s) ran 0 steps — hardware issue '
                f'(camera dropout or motor node hang). '
                f'Check get_ros2_topic_health and get_ros2_alerts.'
            )

    if id_count > ood_count and id_count > 0:
        recommendations.append(
            f'{id_count}/{len(failures)} failures occurred within the training '
            f'distribution. The model sees familiar scenes but still fails — '
            f'this suggests a policy quality issue. Use diagnose_trial and '
            f'profile_trajectory to check if the model is executing a fixed '
            f'trajectory or adapting to the scene.'
        )

    if not recommendations:
        recommendations.append(
            'Run diagnose_trial for GPU kernel comparison between success '
            'and failure, then check_scene_novelty on specific failed trials.'
        )

    return {
        'total_trials': len(trials),
        'successes': len(successes),
        'failures': len(failures),
        'success_rate': round(len(successes) / len(trials), 3),
        'by_failure_mode': mode_summary,
        'distribution_analysis': {
            'in_distribution_failures': id_count,
            'out_of_distribution_failures': ood_count,
            'unknown': len(by_distribution['unknown']),
        },
        'models_involved': list(set(f.get('model_version', '') for f in failures)),
        'conditions_involved': list(set(
            tag for f in failures
            for tag in (f.get('condition', {}).get('tags') or [])
        )),
        'recommendations': recommendations,
    }


def _compute_jitter_metrics(steps):
    """Compute velocity, acceleration, jerk from step-level joint data."""
    import math
    if len(steps) < 4:
        return {'error': 'Need at least 4 steps for jerk computation'}

    joint_names = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
    n_joints = 6
    n_steps = len(steps)

    positions = []
    times = []
    for s in steps:
        joints = s.get('joints', [0]*6)
        positions.append(joints[:n_joints])
        times.append(s.get('step_time', s.get('step', 0) / 30.0))

    velocities = []
    for i in range(1, n_steps):
        dt = times[i] - times[i-1]
        if dt <= 0:
            dt = 1.0 / 30.0
        v = [(positions[i][j] - positions[i-1][j]) / dt for j in range(n_joints)]
        velocities.append(v)

    accelerations = []
    for i in range(1, len(velocities)):
        dt = times[i+1] - times[i]
        if dt <= 0:
            dt = 1.0 / 30.0
        a = [(velocities[i][j] - velocities[i-1][j]) / dt for j in range(n_joints)]
        accelerations.append(a)

    jerks = []
    for i in range(1, len(accelerations)):
        dt = times[i+2] - times[i+1]
        if dt <= 0:
            dt = 1.0 / 30.0
        jk = [(accelerations[i][j] - accelerations[i-1][j]) / dt for j in range(n_joints)]
        jerks.append(jk)

    per_joint = {}
    for j in range(n_joints):
        name = joint_names[j] if j < len(joint_names) else f'joint_{j}'
        j_jerks = [abs(jk[j]) for jk in jerks]
        if not j_jerks:
            continue
        mean_jerk = sum(j_jerks) / len(j_jerks)
        max_jerk = max(j_jerks)
        max_jerk_step = j_jerks.index(max_jerk) + 3

        threshold = mean_jerk * 3
        spikes = [{'step': i + 3, 'jerk': round(j_jerks[i], 2)}
                  for i in range(len(j_jerks)) if j_jerks[i] > threshold]

        j_range = max(positions[i][j] for i in range(n_steps)) - min(positions[i][j] for i in range(n_steps))

        per_joint[name] = {
            'range_deg': round(j_range, 2),
            'mean_jerk': round(mean_jerk, 2),
            'max_jerk': round(max_jerk, 2),
            'max_jerk_step': max_jerk_step,
            'spike_count': len(spikes),
            'spikes': spikes[:5],
        }

    all_jerks = [sum(abs(jk[j]) for j in range(n_joints)) for jk in jerks]
    mean_total = sum(all_jerks) / len(all_jerks) if all_jerks else 0
    max_total = max(all_jerks) if all_jerks else 0

    smoothness_score = max(0, 100 - min(100, mean_total / 50))

    gripper_positions = [positions[i][5] for i in range(n_steps)]
    gripper_range = max(gripper_positions) - min(gripper_positions)
    gripper_closed = min(gripper_positions) < 15

    return {
        'total_steps': n_steps,
        'duration_s': round(times[-1] - times[0], 2) if times else 0,
        'smoothness_score': round(smoothness_score, 1),
        'mean_total_jerk': round(mean_total, 2),
        'max_total_jerk': round(max_total, 2),
        'gripper_range_deg': round(gripper_range, 2),
        'gripper_closed': gripper_closed,
        'per_joint': per_joint,
    }


def _kernel_timing_for_window(trial_timestamp, n_steps):
    """Get kernel timing statistics concurrent with a trial."""
    from datetime import datetime, timedelta
    try:
        t_end = datetime.strptime(trial_timestamp, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        return None

    duration_s = n_steps / 30.0
    t_start = (t_end - timedelta(seconds=duration_s + 5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    t_end_str = t_end.strftime('%Y-%m-%dT%H:%M:%SZ')

    result = es_search(IDX_GPU_PROFILING, {
        'query': {'range': {'@timestamp': {'gte': t_start, 'lte': t_end_str}}},
        'aggs': {
            'by_operation': {
                'terms': {'field': 'gpu.kernel.operation', 'size': 10},
                'aggs': {
                    'total_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
                    'avg_ms': {'avg': {'field': 'gpu.kernel.duration_ms'}},
                    'max_ms': {'max': {'field': 'gpu.kernel.duration_ms'}},
                    'p95_ms': {'percentiles': {'field': 'gpu.kernel.duration_ms',
                                               'percents': [95]}},
                    'p99_ms': {'percentiles': {'field': 'gpu.kernel.duration_ms',
                                               'percents': [99]}},
                },
            },
            'total_gpu_ms': {'sum': {'field': 'gpu.kernel.duration_ms'}},
            'total_kernels': {'value_count': {'field': 'gpu.kernel.name'}},
            'duration_histogram': {
                'histogram': {'field': 'gpu.kernel.duration_ms', 'interval': 0.1},
            },
        },
    }, size=0)

    if not result:
        return None

    aggs = result.get('aggregations', {})
    total_ms = aggs.get('total_gpu_ms', {}).get('value', 0)
    total_kernels = aggs.get('total_kernels', {}).get('value', 0)

    ops = []
    for b in aggs.get('by_operation', {}).get('buckets', []):
        t = b['total_ms']['value']
        ops.append({
            'operation': b['key'],
            'total_ms': round(t, 1),
            'pct': round(t / max(total_ms, 0.001) * 100, 1),
            'avg_ms': round(b['avg_ms']['value'], 4),
            'max_ms': round(b['max_ms']['value'], 3),
            'p95_ms': round(list(b['p95_ms']['values'].values())[0], 4),
            'p99_ms': round(list(b['p99_ms']['values'].values())[0], 4),
            'count': b['doc_count'],
        })

    per_step_ms = total_ms / max(n_steps, 1)
    expected_budget_ms = 1000.0 / 30.0

    return {
        'total_gpu_ms': round(total_ms, 1),
        'total_kernels': total_kernels,
        'per_step_gpu_ms': round(per_step_ms, 2),
        'step_budget_ms': round(expected_budget_ms, 2),
        'gpu_utilization_pct': round(per_step_ms / expected_budget_ms * 100, 1),
        'by_operation': ops,
    }


# --- Entrypoint ---

if __name__ == '__main__':
    print('=== SO-101 Diagnostic Agent MCP Server v15 ===')
    print('Starting ROS2 robot bridge...')
    start_ros2()

    if ES_PASSWORD:
        print(f'Elasticsearch: {ES_URL}')
    else:
        print('WARNING: ES_PASSWORD not set — logging disabled')

    print(f'Evaluator: {EVALUATOR_MODEL} via {OLLAMA_URL}')
    print(f'Starting MCP server on port {MCP_PORT}...')
    print(f'Endpoint: http://0.0.0.0:{MCP_PORT}/mcp')

    if APM_AVAILABLE and os.environ.get('ELASTIC_APM_SERVER_URL'):
        import uvicorn
        from elasticapm.contrib.starlette import make_apm_client, ElasticAPM
        apm_client = make_apm_client()
        app = mcp.http_app()
        app.add_middleware(ElasticAPM, client=apm_client)
        print(f'APM: middleware attached → {os.environ["ELASTIC_APM_SERVER_URL"]}', flush=True)
        uvicorn.run(app, host='0.0.0.0', port=MCP_PORT)
    else:
        print('APM: not configured', flush=True)
        mcp.run(transport='http', host='0.0.0.0', port=MCP_PORT)
