# MCP Tools Catalog

Complete reference for all 37 MCP tools exposed by the robot brain server.

## Robot control

| Tool | Arguments | Returns |
|------|-----------|---------|
| `execute_task` | `task` (string) | Trial result: success, confidence, steps, timing |
| `stop_robot` | — | Confirmation that inference was stopped |
| `swap_model` | `model_path` (string) | Confirmation with new model name |
| `get_robot_status` | — | Camera frame ages, model info, motor connectivity |

## Observation

| Tool | Arguments | Returns |
|------|-----------|---------|
| `get_workspace_snapshot` | — | Current camera frames and joint positions |
| `evaluate_cycle` | `trial_id` (optional) | VLM evaluation of the most recent trial |

## Conditions

| Tool | Arguments | Returns |
|------|-----------|---------|
| `set_condition` | `key`, `value` | Confirmation |
| `get_condition` | `key` (optional) | Current condition value(s) |

## Data and statistics

| Tool | Arguments | Returns |
|------|-----------|---------|
| `get_flywheel_status` | — | Training buffer size, recent trials, model history |
| `query_hard_examples` | `limit` (optional) | Hard examples from the training buffer |
| `query_inference_stats` | `model_name` (optional), `window` (optional) | Aggregate success rate, timing, confidence |

## GPU observability (Refinery)

| Tool | Arguments | Returns |
|------|-----------|---------|
| `get_gpu_kernel_profile` | `trial_id` (optional), `window` (optional) | Kernel count and time by component |
| `get_inference_breakdown` | `trial_id` (optional) | Per-pipeline-stage time allocation |
| `get_gpu_memory_status` | — | Current GPU memory usage, high-water mark |
| `get_episode_gpu_quality` | `trial_id` (optional) | GPU execution quality metrics for a trial |
| `get_control_loop_status` | — | Loop frequency, jitter, duty cycle |
| `get_safety_events` | `window` (optional) | Anomalous kernel durations or memory spikes |
| `profile_inference` | `window` (optional) | Duty cycle summary over time |

## Diagnostics

| Tool | Arguments | Returns |
|------|-----------|---------|
| `diagnose_trial` | `trial_id` (optional) | Multi-signal failure classification with evidence |
| `compare_model_kernels` | `model_a`, `model_b` | Side-by-side GPU kernel comparison |
| `check_scene_novelty` | — | CLIP kNN distance to training distribution |
| `analyze_failure_patterns` | `window` (optional) | Aggregate failure patterns over time |
| `profile_trajectory` | `trial_id` (optional) | Per-step joint commands, z-scores, FK coordinates |
| `training_coverage_report` | — | Dataset gap analysis: joint coverage, position diversity |

## ROS 2 observability

| Tool | Arguments | Returns |
|------|-----------|---------|
| `get_ros2_topic_health` | — | Publication rates, bandwidth per topic |
| `get_ros2_node_status` | — | Active ROS 2 nodes and their status |
| `get_ros2_alerts` | — | Topics or nodes that have gone silent |

## Observatory

| Tool | Arguments | Returns |
|------|-----------|---------|
| `workspace_trajectory` | `trial_id` (optional) | FK-computed workspace path (XYZ coordinates) |

Plus additional per-step telemetry and observatory tools for detailed
model decision analysis.

## System

| Tool | Arguments | Returns |
|------|-----------|---------|
| `reset_arm` | — | Smooth 90-step reset to training start position |
| `get_system_health` | — | Cluster status, pod health, service connectivity |

## Calling tools

### Via Agent Builder

Ask in natural language. The agent selects the appropriate tool:

- "What's the robot status?" → `get_robot_status`
- "Pick up the cube" → `execute_task`
- "Why did that fail?" → `diagnose_trial`
- "How is the GPU doing?" → `get_gpu_kernel_profile`

### Via connector API

```bash
curl -sk -u elastic:<password> \
  "https://<gpu-server>:31561/api/actions/connector/<connector-uuid>/_execute" \
  -X POST -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{
    "params": {
      "subAction": "callTool",
      "subActionParams": {
        "name": "<tool_name>",
        "arguments": {<tool_args>}
      }
    }
  }'
```

Valid subActions: `callTool`, `listTools`, `test`. Not `toolsCall` or `run`.
