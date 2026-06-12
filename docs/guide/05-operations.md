# Operations

This page covers the day-to-day operation of the system: running trials,
comparing models, interpreting results, and handling common failures.

## Running trials via Agent Builder

Open Kibana, navigate to Agent Builder, and select the SO-101 agent. Type a
natural language command:

- "Pick up the cube"
- "Move the cube to the left side of the workspace"
- "Run 5 trials and report success rate"

The agent calls MCP tools to execute the task. A typical `execute_task` call
runs inference for ~600 steps at 30 FPS (~20 seconds), then returns a trial
summary with:

- Success/failure classification
- Wrist camera frame at start and end
- Joint trajectory summary
- Execution timing

You can also drive tools directly via the connector API (see
[03-deployment.md](03-deployment.md)) for scripted trial runs.

## Model comparison workflow

To compare two models:

1. Run a batch of trials with the current model (e.g., 10 trials)
2. Ask the agent to swap models: "Swap to model xvla_v2"
3. Run the same batch with the new model
4. Ask: "Compare the last two models"

The `compare_model_kernels` tool provides GPU-level comparison: kernel count,
duration distribution, and component-level time allocation between models. The
`query_inference_stats` tool provides aggregate success rates and timing.

### Model hot-swap procedure

The `swap_model` tool changes the inference model without restarting the pod:

1. The MCP server publishes a model swap command to the inference node via ROS 2
2. The inference node unloads the current model from GPU memory
3. The new checkpoint is loaded from the mounted `${MODEL_PATH}` directory
4. Inference resumes on the next `execute_task` call

The model path in the `swap_model` argument must be a path inside the container
(where checkpoints are mounted), not a host path.

## CLIP confidence scoring

Every trial records CLIP embeddings of the wrist camera frame. The
`check_scene_novelty` tool compares the current scene against the training
distribution using kNN distance:

| Distance | Interpretation |
|----------|---------------|
| <= 0.038 | Within training distribution -- high confidence |
| 0.038 - 0.042 | Boundary -- may succeed but less reliable |
| >= 0.042 | Out of distribution -- expect failure |

CLIP scoring works best for position-dependent tasks. It detects lateral
displacement well but is less sensitive to forward/backward displacement from
the wrist camera's perspective. The overhead camera provides complementary
coverage.

Embed frames before execution, not after -- post-execution frames include the
arm in the scene, which pollutes the distance measurement.

## Reading the SRE dashboard

The main SRE dashboard shows:

- **Trial success rate** over time (target: consistent, not necessarily high)
- **MCP tool latency** from APM traces (p50, p95, p99)
- **GPU utilization** from DCGM (should be 80-95% during inference, near 0% at idle)
- **ROS 2 topic rates** (camera topics should publish at ~30 Hz, motor at ~30 Hz)
- **Control loop jitter** from Refinery (standard deviation of loop period)

If the success rate drops suddenly, check the diagnostic tools before
assuming a model problem -- hardware issues (camera disconnects, motor stalls)
are more common than model regression.

## Common failure modes

| Symptom | Likely cause | Diagnostic tool |
|---------|-------------|-----------------|
| Robot does not move | Motor node not connected, wrong serial port | `get_robot_status` |
| Robot moves but misses target | Position out of training distribution | `check_scene_novelty` |
| Inference timeout (>60s) | GPU memory exhaustion, model loading failure | `get_gpu_memory_status` |
| Camera frames stale | USB disconnect, device path shifted | `get_robot_status` (check frame ages) |
| Jerky motion | Control loop jitter, ROS 2 topic drops | `get_control_loop_status`, `get_ros2_topic_health` |
| Gripper does not close | Gripper mode collapse in model (training issue) | `profile_trajectory` (check gripper channel) |

The `diagnose_trial` tool fuses multiple signals (GPU kernels, CLIP distance,
VLM evaluation, trajectory analysis) to classify failures as planning,
perception, or hardware. Use it as the first step when a trial fails
unexpectedly.

## Alert rules

Configure these alerting rules in Kibana:

- **Camera topic silent** -- `metrics-ros2.topics-*` where topic rate drops to
  0 for camera topics. Threshold: 0 publications in 30 seconds.
- **Inference loop stalled** -- `metrics-gpu.robotics-*` where loop frequency
  drops below 10 Hz (target is 30 Hz).
- **GPU temperature** -- `DCGM_FI_DEV_GPU_TEMP` exceeds 85 C.
- **MCP tool errors** -- APM error rate for `so101-mcp-server` exceeds 10%
  over 5 minutes.
- **Trial failure streak** -- `robot-inference` where 5+ consecutive trials
  have `success: false`.

## Startup procedure

After a reboot or maintenance window:

1. Verify k3s is running on both nodes: `sudo kubectl get nodes`
2. Check all pods: `sudo kubectl get pods -n robot -o wide`
3. Verify camera device paths: `v4l2-ctl --list-devices` (they may shift after reboot)
4. Update `CAM0_DEVICE`/`CAM1_DEVICE` in the manifest if paths changed
5. Verify motor serial port: `ls /dev/ttyACM*` on the edge node
6. Run `get_robot_status` via the connector API to confirm end-to-end connectivity
7. Run a test trial: "pick up the cube"
