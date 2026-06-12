#!/usr/bin/env python3
"""
VLA Inference Node v5 — runs on the GPU server.

Model-agnostic: supports SmolVLA, pi0, pi0.5, X-VLA, or any
LeRobot policy that follows the from_pretrained pattern.

Hot-swap: subscribe to /swap_model to reload a different checkpoint
at runtime without restarting the container.

Uses LeRobot's make_pre_post_processors factory for proper
normalization, tokenization, and unnormalization.

Topics:
  Subscribes:
    /camera/cam0/compressed  (wrist)
    /camera/cam1/compressed  (overhead)
    /joint_states            (current arm positions from motor node)
    /task_instruction        (String — natural language command)
    /inference_control       (String — "start", "stop")
    /swap_model              (String — checkpoint path to load)

  Publishes:
    /joint_commands          (Float32MultiArray — 6 joint positions)
    /inference_status        (String — status updates)
"""

import gc
import os
import threading
import time
import importlib
from collections import deque

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, String


# --- Configuration ---

MODEL_PATH = os.environ.get('MODEL_PATH', '/models/pretrained_model')
DEVICE = os.environ.get('DEVICE', 'cuda')
CAMERA_TOPICS = {
    'cam0': '/camera/cam0/compressed',
    'cam1': '/camera/cam1/compressed',
}
# Map ROS topic keys to dataset camera names
# The preprocessor's rename step handles camera1/camera2 mapping
CAM_NAME_MAP = {
    'cam0': 'wrist',
    'cam1': 'overhead',
}
INFERENCE_FPS = 30
FRAME_TIMEOUT_SEC = 2.0
EXECUTION_STEPS_TARGET = int(os.environ.get('EXECUTION_STEPS_TARGET', '600'))

# --- Trajectory Correction (opt-in via TRAJECTORY_CORRECTION=1) ---
TRAJECTORY_CORRECTION = os.environ.get('TRAJECTORY_CORRECTION', '') == '1'
# Trained-position cube centroid in overhead camera (480x480)
TRAINED_CENTROID = (135, 65)
# Pixels-per-degree mapping (empirical, tunable)
# Joint 1 (base rotation): lateral pixel displacement
# Joint 2 (shoulder): forward/back pixel displacement
CORRECTION_SCALE = {
    'j1_deg_per_px': float(os.environ.get('CORRECTION_J1_SCALE', '0.15')),
    'j2_deg_per_px': float(os.environ.get('CORRECTION_J2_SCALE', '0.10')),
}
# HSV range for blue cube detection
BLUE_HSV_LOW = (100, 100, 60)
BLUE_HSV_HIGH = (130, 255, 255)
# Max correction in degrees — safety clamp
MAX_CORRECTION_DEG = 5.0


def detect_cube_offset(frame, trained_cx=TRAINED_CENTROID[0], trained_cy=TRAINED_CENTROID[1]):
    """Detect blue cube centroid from overhead camera, return pixel offset from trained position.
    Accepts either a BGR numpy image or raw JPEG bytes."""
    if isinstance(frame, np.ndarray):
        img = frame
    else:
        arr = np.frombuffer(frame if isinstance(frame, bytes) else bytes(frame), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:
        return None
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None
    # Filter contours to workspace ROI (cube should be in upper portion of overhead frame)
    workspace_contours = []
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        m = cv2.moments(c)
        if m['m00'] == 0:
            continue
        cy_c = int(m['m01'] / m['m00'])
        if cy_c < trained_cy + 80:
            workspace_contours.append(c)
    if not workspace_contours:
        return None
    largest = max(workspace_contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    dx, dy = cx - trained_cx, cy - trained_cy
    if abs(dx) > 80 or abs(dy) > 80:
        return None
    return (dx, dy)


def compute_joint_correction(pixel_offset):
    """Convert pixel offset to joint angle corrections (degrees), clamped for safety."""
    dx, dy = pixel_offset
    def clamp(v):
        return max(-MAX_CORRECTION_DEG, min(MAX_CORRECTION_DEG, v))
    return [
        clamp(-dx * CORRECTION_SCALE['j1_deg_per_px']),
        clamp(-dy * CORRECTION_SCALE['j2_deg_per_px']),
        clamp(dy * CORRECTION_SCALE['j2_deg_per_px'] * 0.5),
        0.0, 0.0, 0.0,
    ]


# --- Model Loading ---

def detect_policy_type(model_path):
    """Detect the policy type from the checkpoint's config.json."""
    import json
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        # Check explicit 'type' field first (most reliable)
        explicit_type = config.get('type', '').lower()
        if explicit_type in ('xvla', 'smolvla', 'pi0', 'pi05'):
            return explicit_type
        # Fallback: check _target_ / policy_type fields
        # X-VLA before SmolVLA — X-VLA inherits SmolVLA so str(config) matches both
        policy_type = config.get('_target_', config.get('policy_type', ''))
        if 'xvla' in policy_type.lower():
            return 'xvla'
        elif 'smolvla' in policy_type.lower() or 'smolvla' in str(config):
            return 'smolvla'
        elif 'pi05' in policy_type.lower() or 'pi0.5' in str(config):
            return 'pi05'
        elif 'pi0' in policy_type.lower():
            return 'pi0'
    return 'smolvla'


def _patch_xvla_compat():
    """Patch SmolVLAPolicy.__init__ so XVLAPolicy (which inherits it) doesn't crash.

    XVLAConfig lacks rtc_config and vlm_model_name. SmolVLAPolicy.__init__ tries
    to use both, then XVLAPolicy.__init__ overwrites self.model anyway. We skip
    init_rtc_processor and VLAFlowMatching creation when the config is X-VLA.
    """
    try:
        from lerobot.policies.smolvla.modeling_smolvla import (
            SmolVLAPolicy, VLAFlowMatching,
        )
        _orig_init = SmolVLAPolicy.__init__

        def _safe_init(self, config, **kwargs):
            is_xvla = not hasattr(config, 'vlm_model_name')
            if is_xvla:
                # Minimal base init — XVLAPolicy.__init__ will set self.model
                super(SmolVLAPolicy, self).__init__(config)
                config.validate_features()
                self.config = config
                self.rtc_processor = None
                self.model = None
                self.reset()
            else:
                _orig_init(self, config, **kwargs)

        SmolVLAPolicy.__init__ = _safe_init
        print(f'[PATCH] SmolVLAPolicy.__init__ patched for X-VLA compat')
    except Exception as e:
        print(f'[PATCH] WARNING: X-VLA compat patch failed: {e}')

_patch_xvla_compat()


def load_model(model_path, device='cuda'):
    """Load a policy and its pre/post processors from a checkpoint.
    Auto-detects policy type from the checkpoint config."""
    policy_type = detect_policy_type(model_path)
    print(f'Detected policy type: {policy_type}')

    # Import the policy module to register processors
    if policy_type == 'smolvla':
        import lerobot.policies.smolvla
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as PolicyClass
    elif policy_type == 'pi0':
        import lerobot.policies.pi0
        from lerobot.policies.pi0 import PI0Policy as PolicyClass
    elif policy_type == 'pi05':
        import lerobot.policies.pi05
        from lerobot.policies.pi05 import PI05Policy as PolicyClass
    elif policy_type == 'xvla':
        import lerobot.policies.xvla
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy as PolicyClass
    else:
        raise ValueError(f'Unknown policy type: {policy_type}')

    print(f'Loading {policy_type} from {model_path}...')
    policy = PolicyClass.from_pretrained(model_path)
    policy.to(device)
    policy.eval()
    print(f'Policy loaded on {device}')

    # Load pre/post processors using the factory
    try:
        from lerobot.policies.factory import make_pre_post_processors
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, model_path,
            preprocessor_overrides={'device_processor': {'device': device}},
        )
        print(f'Using make_pre_post_processors factory')
    except (ImportError, Exception) as e:
        print(f'Factory not available ({e}), falling back to DataProcessorPipeline')
        from lerobot.processor.pipeline import DataProcessorPipeline
        preprocessor = DataProcessorPipeline.from_pretrained(
            model_path, 'policy_preprocessor.json'
        )
        postprocessor = DataProcessorPipeline.from_pretrained(
            model_path, 'policy_postprocessor.json'
        )

    return policy, preprocessor, postprocessor, policy_type


# --- Frame processing ---

def decode_compressed(data):
    """Decode JPEG bytes to numpy BGR image."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def image_to_tensor(image):
    """Convert BGR numpy image to [C, H, W] float tensor (0-1 range)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


# --- ROS2 Node ---

class InferenceNode(Node):

    def __init__(self, policy, preprocessor, postprocessor,
                 policy_type='smolvla', device='cuda'):
        super().__init__('vla_inference')
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.policy_type = policy_type
        self.device = device
        self.model_lock = threading.Lock()
        self.frame_lock = threading.Lock()

        # State
        self.latest_frames = {}
        self.latest_joint_state = None
        self.task_description = ''
        self.running = False
        self.step_count = 0
        self.model_path = MODEL_PATH
        self._last_frame_warning = 0  # throttle frame warnings to 1 per 5s
        self._correction_offset = None  # computed once per task from overhead camera

        # Camera subscribers
        for topic_key, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                CompressedImage, topic,
                lambda msg, k=topic_key: self._on_frame(k, msg), 10
            )
            self.get_logger().info(f'Camera: {topic}')

        # Joint state subscriber
        self.create_subscription(
            Float32MultiArray, '/joint_states',
            self._on_joint_state, 10
        )

        # Control subscribers
        self.create_subscription(String, '/task_instruction', self._on_task, 10)
        self.create_subscription(String, '/inference_control', self._on_control, 10)
        self.create_subscription(String, '/swap_model', self._on_swap_model, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Float32MultiArray, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/inference_status', 10)

        # Timers
        self.create_timer(1.0 / INFERENCE_FPS, self._inference_step)
        self.create_timer(1.0, self._publish_status)

        self.get_logger().info(
            f'Inference node ready ({policy_type}). Send task on /task_instruction'
        )

    # --- Callbacks ---

    def _on_frame(self, topic_key, msg):
        image = decode_compressed(bytes(msg.data))
        if image is not None:
            with self.frame_lock:
                self.latest_frames[topic_key] = (time.time(), image)

    def _on_joint_state(self, msg):
        with self.frame_lock:
            self.latest_joint_state = (time.time(), list(msg.data[:6]))

    def _on_task(self, msg):
        self.task_description = msg.data
        self.running = True
        self.step_count = 0
        self._correction_offset = None
        self._reset_action_queue()
        with self.frame_lock:
            self.latest_frames.clear()
        gc.disable()
        self.get_logger().info(f'Task received, starting: {self.task_description}')

    def _on_control(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == 'start' and self.task_description:
            self.running = True
            gc.disable()
            self.get_logger().info('Inference started')
        elif cmd == 'stop':
            self.running = False
            gc.enable()
            gc.collect()
            self.get_logger().info('Inference stopped')

    def _on_swap_model(self, msg):
        """Hot-swap the model checkpoint."""
        new_path = msg.data.strip()
        if not new_path or not os.path.isdir(new_path):
            self.get_logger().error(f'Invalid model path: {new_path}')
            return

        self.get_logger().info(f'Swapping model to: {new_path}')
        self.running = False
        time.sleep(0.5)

        try:
            with self.model_lock:
                # Free old model memory
                del self.policy
                del self.preprocessor
                del self.postprocessor
                torch.cuda.empty_cache()

                # Load new model
                policy, preprocessor, postprocessor, policy_type = load_model(
                    new_path, self.device
                )
                self.policy = policy
                self.preprocessor = preprocessor
                self.postprocessor = postprocessor
                self.policy_type = policy_type
                self.model_path = new_path

            self.step_count = 0
            self._reset_action_queue()
            self.get_logger().info(
                f'Model swapped to {new_path} ({policy_type}). Ready.'
            )
        except Exception as e:
            self.get_logger().error(f'Model swap failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    # --- Helpers ---

    def _reset_action_queue(self):
        """Reset the policy's internal action queue."""
        if hasattr(self.policy, '_queues'):
            self.policy._queues = {
                k: deque(maxlen=v.maxlen)
                for k, v in self.policy._queues.items()
            }

    def _publish_status(self):
        msg = String()
        if self.running:
            with self.frame_lock:
                now = time.time()
                cams_ok = all(
                    k in self.latest_frames and now - self.latest_frames[k][0] < FRAME_TIMEOUT_SEC
                    for k in CAMERA_TOPICS
                )
            if cams_ok:
                msg.data = f'running (step {self.step_count})'
            else:
                msg.data = f'running (step {self.step_count}, waiting_for_cameras)'
        else:
            msg.data = 'idle' if self.task_description else 'waiting_for_task'
        self.status_pub.publish(msg)

    # --- Inference ---

    def _inference_step(self):
        if not self.running or not self.task_description:
            return

        with self.frame_lock:
            now = time.time()

            frames = {}
            missing = []
            stale = []
            for topic_key in CAMERA_TOPICS:
                if topic_key not in self.latest_frames:
                    missing.append(topic_key)
                    continue
                ts, image = self.latest_frames[topic_key]
                if now - ts > FRAME_TIMEOUT_SEC:
                    stale.append((topic_key, now - ts))
                    continue
                frames[topic_key] = image

            if missing or stale:
                if self.running and now - self._last_frame_warning > 5.0:
                    self._last_frame_warning = now
                    parts = []
                    if missing:
                        parts.append(f'missing: {missing}')
                    if stale:
                        parts.append(f'stale: {[(k, f"{age:.0f}s") for k, age in stale]}')
                    self.get_logger().warn(f'Inference blocked — {", ".join(parts)}')
                return

            joint_state = None
            if self.latest_joint_state:
                ts, values = self.latest_joint_state
                if now - ts < FRAME_TIMEOUT_SEC:
                    joint_state = values

        try:
            with self.model_lock:
                # Build raw observation dict
                observation = {}

                for topic_key, image in frames.items():
                    cam_name = CAM_NAME_MAP.get(topic_key, topic_key)
                    observation[f'observation.images.{cam_name}'] = image_to_tensor(image)

                if joint_state is not None:
                    observation['observation.state'] = torch.tensor(
                        joint_state, dtype=torch.float32
                    )

                observation['task'] = self.task_description

                if self.step_count == 0:
                    print(f'[DBG] first step, obs keys: {list(observation.keys())}', flush=True)

                # Preprocess → inference → postprocess
                batch = self.preprocessor(observation)

                if self.step_count == 0:
                    bkeys = list(batch.keys()) if isinstance(batch, dict) else str(type(batch))
                    print(f'[DBG] batch keys: {bkeys}', flush=True)
                    print('[DBG] calling select_action...', flush=True)

                with torch.no_grad():
                    action = self.policy.select_action(batch)

                if self.step_count == 0:
                    print(f'[DBG] action shape={action.shape}, vals={action.flatten()[:6].tolist()}', flush=True)

                action_in = action.unsqueeze(0) if action.dim() == 1 else action
                output = self.postprocessor(action_in)
                if isinstance(output, dict):
                    positions = output['action'].squeeze().numpy().tolist()
                else:
                    positions = output.squeeze().numpy().tolist()

            # Trajectory correction: detect cube offset on first step, apply every step
            if TRAJECTORY_CORRECTION and self.step_count == 0:
                overhead_entry = self.latest_frames.get('cam1')
                if overhead_entry:
                    _, overhead_data = overhead_entry
                    pixel_offset = detect_cube_offset(overhead_data)
                    if pixel_offset:
                        self._correction_offset = compute_joint_correction(pixel_offset)
                        print(f'[CORRECTION] Cube offset: px=({pixel_offset[0]}, {pixel_offset[1]}), '
                              f'joint correction: [{", ".join(f"{c:.2f}" for c in self._correction_offset)}]',
                              flush=True)
                    else:
                        self._correction_offset = None
                        print('[CORRECTION] No blue cube detected — no correction applied', flush=True)

            if self._correction_offset:
                positions = [p + c for p, c in zip(positions[:6], self._correction_offset)]

            # Publish joint commands
            msg = Float32MultiArray()
            msg.data = positions[:6]
            self.cmd_pub.publish(msg)

            self.step_count += 1
            if self.step_count % 50 == 0:
                self.get_logger().info(
                    f'Step {self.step_count}, cmd: '
                    f'[{", ".join(f"{p:.1f}" for p in positions[:6])}]'
                )

            if self.step_count >= EXECUTION_STEPS_TARGET:
                self.running = False
                gc.enable()
                gc.collect()
                self.get_logger().info(
                    f'Reached {EXECUTION_STEPS_TARGET} steps, auto-stopping'
                )

        except Exception as e:
            print(f'INFERENCE ERROR: {e}', flush=True)
            import traceback
            traceback.print_exc()
            self.running = False


# --- Main ---

def main():
    print('=== VLA Inference Node v5 ===')

    policy, preprocessor, postprocessor, policy_type = load_model(
        MODEL_PATH, DEVICE
    )

    rclpy.init()
    node = InferenceNode(
        policy, preprocessor, postprocessor,
        policy_type=policy_type, device=DEVICE,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
