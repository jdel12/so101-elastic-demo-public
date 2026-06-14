# Architecture

A deep-dive into the system's components, data flow, and design decisions.

## System overview

The system has three logical layers:

1. **Control layer** — Kibana Agent Builder issues natural language commands
   through an LLM, which calls MCP tools on the robot brain
2. **Execution layer** — VLA inference on GPU produces motor commands,
   ROS 2 DDS carries them to the edge node, servos move the arm
3. **Observability layer** — every signal (GPU kernels, camera frames,
   joint trajectories, model decisions) flows to Elasticsearch for
   analysis and diagnosis

## Data flow

```
User (Kibana Agent Builder)
  │
  │ Natural language: "pick up the cube"
  ▼
LLM (Agent Builder reasoning)
  │
  │ Tool call: execute_task(task="pick up the cube")
  ▼
MCP Server (FastMCP, port 8888)
  │
  │ ROS 2: /task_instruction
  ▼
Inference Node (GPU 0)
  │
  │ Reads: /camera/cam0/compressed, /camera/cam1/compressed, /joint_states
  │ Processes: VLA model (camera frames + task + joints → action)
  │ Publishes: /joint_commands at 30 Hz
  ▼
Motor Node (edge, USB serial)
  │
  │ Feetech protocol → servo positions
  ▼
SO-101 Follower Arm
```

Simultaneously, the observability layer captures:

```
Inference Node ──→ per-step telemetry ──→ robot-steps index
GPU (CUPTI shim) ──→ kernel traces ──→ logs-gpu.kernel-* index
MCP Server ──→ APM traces ──→ traces-apm-* index
Camera Nodes ──→ ROS 2 metrics ──→ metrics-ros2.topics-* index
DCGM Exporter ──→ GPU metrics ──→ Prometheus → Elastic Agent
Ollama ──→ VLM evaluation ──→ robot-evaluations index
Jina CLIP ──→ scene embeddings ──→ robot-scene-embeddings index
```

## Component details

### MCP server

**Runtime:** Python 3.12 on ROS 2 Jazzy, running in a container with
FastMCP as the HTTP server framework.

**37 tools** organized by function:

- **Robot control (4):** execute_task, stop_robot, swap_model,
  get_robot_status
- **Observation (2):** get_workspace_snapshot, evaluate_cycle
- **Conditions (2):** set_condition, get_condition
- **Data & stats (3):** get_flywheel_status, query_hard_examples,
  query_inference_stats
- **GPU / Refinery (7):** get_gpu_kernel_profile, get_inference_breakdown,
  get_gpu_memory_status, get_episode_gpu_quality, get_control_loop_status,
  get_safety_events, profile_inference
- **Diagnostics (6):** diagnose_trial, compare_model_kernels,
  check_scene_novelty, analyze_failure_patterns, profile_trajectory,
  training_coverage_report
- **ROS 2 (3):** get_ros2_topic_health, get_ros2_node_status,
  get_ros2_alerts
- **Observatory (7):** workspace_trajectory, and per-step telemetry tools
- **System (3):** reset_arm, get_system_health, get_condition

Each tool talks to Elasticsearch, Ollama, Jina CLIP, and/or the ROS 2
network as needed. APM traces every call automatically.

### Inference node

**Runtime:** Python 3.12 on ROS 2 Jazzy, PyTorch with CUDA.

The inference node is model-agnostic — it supports X-VLA, SmolVLA, Pi0,
and GR00T through a common interface:

1. Subscribe to camera topics and joint state
2. On task instruction, start inference loop
3. Each step: encode frames → run model → decode action → publish commands
4. After target steps (default 600), stop and report

The node supports hot-swap: it can unload one model and load another
without restarting the container.

**Telemetry:** The inference telemetry module captures per-step data
including z-scores, pipeline scores, and forward kinematics coordinates.
This data feeds the Model Decision Observatory.

### Camera node

**Runtime:** Python on ROS 2 Humble, OpenCV for frame capture.

Features:
- USB device discovery by bus number (no hardcoded `/dev/video*` paths)
- Auto-recovery after 10 consecutive dropped frames
- JPEG compression (quality 80) for bandwidth efficiency
- Publishes CompressedImage messages at target FPS (15-30 Hz)

Camera containers use ROS 2 Humble because the base image is smaller and
camera capture doesn't need Jazzy features. FastDDS SHM is disabled to
allow cross-version DDS communication with the Jazzy inference container.

### Motor node

**Runtime:** Python on ROS 2 Humble, LeRobot Feetech driver.

Subscribes to `/joint_commands` and translates ROS 2 messages into
Feetech servo protocol over USB serial. Publishes `/joint_states` with
current servo positions for feedback.

### Forward kinematics

Converts 6-DOF joint angles to workspace coordinates (XYZ in millimeters)
using the SO-101's URDF-derived kinematic chain. Produces:

- End-effector position (ee_x, ee_y, ee_z)
- Elbow position (elbow_x, elbow_y, elbow_z)

These coordinates appear in every step document, enabling spatial analysis
of trajectories without manual measurement.

## Design decisions

### Why Kubernetes, not Docker Compose?

- Automatic pod restart on crash (cameras disconnect, USB re-enumerates)
- ECK handles Elasticsearch lifecycle
- Node selectors route work to correct hardware
- Enterprise-grade pattern that transfers to production
- DaemonSets for system-wide observability agents

### Why ROS 2, not direct serial?

- Decouples inference from motor control
- DDS handles networking between nodes
- Standard topic/service model for camera frames
- Enables multi-node without custom networking
- Industry standard for robotics middleware

### Why FastMCP, not a custom API?

- MCP is the emerging standard for LLM tool integration
- Kibana Agent Builder has native MCP connector support
- Tools are self-documenting (schema + description)
- Same tools work from Agent Builder UI and direct API calls

### Why host-mounted source files?

The MCP server and inference node are mounted from the host filesystem
into running containers. This means edits take effect on pod restart
without rebuilding images (~15-20 minute builds).

The tradeoff: the code on the host must match what the container expects.
If you change imports or add dependencies, you need an image rebuild.

### Why separate camera containers?

Each camera runs in its own pod because:

- Independent failure domains — one camera crash doesn't take down the other
- Separate USB controller isolation is enforced at the pod level
- Different cameras can have different configurations
- Kubernetes restart policy handles USB re-enumeration automatically
