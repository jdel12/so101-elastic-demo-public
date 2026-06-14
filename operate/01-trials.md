# Running Trials

This guide covers how to run trials, interpret results, use the AI agent
effectively, and recognize common failure patterns.

## Talking to the robot

Open Kibana Agent Builder and select your agent. You can give natural
language instructions:

- "Pick up the cube"
- "Move the cube to the left side"
- "Run 5 trials and report the success rate"
- "What's the robot status?"
- "How is the GPU doing?"

The agent decides which MCP tools to call based on your instruction. For
a task like "pick up the cube," it calls `execute_task`. For diagnostics,
it calls `diagnose_trial`, `check_scene_novelty`, or other diagnostic tools.

You can also call tools directly via the connector API for scripted runs:

```bash
curl -sk -u elastic:<password> \
  "https://<gpu-server>:31561/api/actions/connector/<connector-uuid>/_execute" \
  -X POST -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{
    "params": {
      "subAction": "callTool",
      "subActionParams": {
        "name": "execute_task",
        "arguments": {"task": "pick up the cube"}
      }
    }
  }'
```

## What happens during a trial

When `execute_task` runs:

1. The MCP server publishes the task instruction to the inference node
2. The VLA model processes camera frames + task instruction + joint state
3. It outputs joint commands at 30 FPS for ~600 steps (~20 seconds)
4. Motor commands flow via ROS 2 DDS to the edge node
5. The follower arm executes the motion
6. After execution, Ollama (Qwen3-VL) evaluates the camera frames and
   judges success or failure
7. CLIP embeddings are computed for scene novelty scoring
8. A trial document is indexed to `robot-inference` with the full result

## Reading trial results

Each trial produces a document in the `robot-inference` index with:

| Field | What it means |
|-------|--------------|
| `success` | Boolean — did the evaluator judge this as successful? |
| `confidence` | 0-1 score from the VLM evaluator |
| `task` | The natural language instruction |
| `model_name` | Which checkpoint was used |
| `steps_executed` | How many inference steps ran |
| `execution_time_s` | Wall-clock execution time |
| `clip_distance` | kNN distance to training embeddings |
| `evaluator_verdict` | Full text from the VLM evaluator |

### CLIP distance interpretation

CLIP distance is the kNN distance from the current camera frame to the
nearest training data embeddings. It measures how "familiar" the scene
looks to the model.

| Distance | Interpretation |
|----------|---------------|
| <= 0.038 | Within training distribution — high confidence |
| 0.038 - 0.042 | Boundary — may succeed but less reliable |
| >= 0.042 | Out of distribution — expect failure |

CLIP scoring is axis-dependent. The wrist camera detects lateral
displacement well but is less sensitive to forward/backward displacement.
The overhead camera provides complementary coverage. The system evaluates
all available cameras and uses the highest-confidence result.

**Embed before execution, not after.** Post-execution frames include the
arm in the scene, which pollutes the distance measurement.

## Common failure modes

| Symptom | Likely cause | First diagnostic |
|---------|-------------|-----------------|
| Robot doesn't move | Motor node disconnected, wrong serial port | `get_robot_status` |
| Robot moves but misses | Position out of training distribution | `check_scene_novelty` |
| Inference timeout (>60s) | GPU OOM, model loading failure | `get_gpu_memory_status` |
| Camera frames stale | USB disconnect, device path shifted | `get_robot_status` (frame ages) |
| Jerky motion | Control loop jitter, ROS 2 drops | `get_control_loop_status` |
| Gripper doesn't close | Gripper mode collapse (training issue) | `profile_trajectory` |
| Success rate drops suddenly | Hardware issue more likely than model regression | Check cameras, motor, then model |

**The most important rule:** When trials fail, check hardware before
blaming the model. Camera disconnects, USB re-enumeration, and motor stalls
are more common than model regression. The `get_robot_status` tool checks
all of this in one call.

## Model comparison

To compare two models:

1. Run a batch of trials with the current model
2. Swap models: ask the agent "Swap to model xvla_v2" or call `swap_model`
3. Run the same batch with the new model
4. Ask: "Compare the last two models"

The `compare_model_kernels` tool provides GPU-level comparison: kernel
count, duration distribution, and component-level time allocation.
`query_inference_stats` provides aggregate success rates and timing.

### Hot-swap procedure

The `swap_model` tool changes the inference model without restarting the
pod:

1. MCP server publishes a model swap command via ROS 2
2. Inference node unloads the current model from GPU memory
3. New checkpoint loads from the mounted model directory
4. Inference resumes on the next `execute_task` call

The model path argument must be a container path (where checkpoints are
mounted), not a host path.

## Startup procedure

After a reboot or maintenance window:

1. Verify k3s: `sudo kubectl get nodes` (both Ready)
2. Check pods: `sudo kubectl get pods -n robot -o wide`
3. Verify cameras: `v4l2-ctl --list-devices` (paths may shift after reboot)
4. Update camera env vars in the manifest if paths changed
5. Verify motor: `ls /dev/ttyACM*` on the edge node
6. Call `get_robot_status` to confirm end-to-end connectivity
7. Run a test trial

## Recommended alert rules

Configure these in Kibana to catch problems before they cascade:

| Alert | Condition | Threshold |
|-------|-----------|-----------|
| Camera topic silent | `metrics-ros2.topics-*` rate drops to 0 | 0 publications in 30s |
| Inference loop stalled | `metrics-gpu.robotics-*` frequency < 10 Hz | Target is 30 Hz |
| GPU temperature | `DCGM_FI_DEV_GPU_TEMP` | > 85 C |
| MCP tool errors | APM error rate for `so101-mcp-server` | > 10% over 5 minutes |
| Trial failure streak | `robot-inference` consecutive failures | 5+ consecutive `success: false` |
