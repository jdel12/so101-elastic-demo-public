# Diagnostics and the Model Decision Observatory

This is the most important section of this repository. Everything else —
the hardware, the deployment, the observability stack — exists to support
this capability: understanding *why* a model fails and *what* to change
to fix it.

## The problem

A VLA model runs on your robot. Sometimes it picks up the cube. Sometimes
it misses. Standard tooling tells you almost nothing about why:

- `nvidia-smi` says "85% utilization" — but at what?
- ROS 2 logs say frames were published — but did the model use them?
- The model outputs joint commands — but were they reasonable?
- The evaluator says "failure" — but was it a gripper issue, a position
  issue, or a perception issue?

Watching the arm and guessing is not engineering. The Model Decision
Observatory makes failure diagnosis systematic and quantitative.

## Multi-signal failure diagnosis

The `diagnose_trial` MCP tool is the entry point. It fuses four independent
signal sources and classifies failures:

### 1. GPU kernel traces (via Refinery)

Kernel-level GPU profiling reveals what the model is spending compute on:

- **Vision encoder** (78.6%) — processing camera frames through ResNet
- **Attention** (11.4%) — transformer cross-attention between vision and
  language tokens
- **Action head** (2.3%) — MLP that outputs joint commands
- **Memory ops** (7.7%) — data movement

When kernel profiles are identical between success and failure, the model
is replaying a memorized trajectory — it's not using visual information to
adapt. This is a planning failure, not a perception failure, and it tells
you the model needs more diverse training data, not better cameras.

### 2. CLIP distance (scene novelty)

How similar is the current scene to the training data? Jina CLIP v2
computes 1024-dimensional embeddings of camera frames and compares them
via kNN distance to the training distribution.

- **Distance <= 0.038:** within distribution, high confidence
- **Distance >= 0.042:** out of distribution, expect failure

When a trial fails with low CLIP distance (within distribution), the
problem is in the model, not the position. When it fails with high CLIP
distance, the model has never seen this scenario and you need to record
training data at this position.

### 3. Trajectory analysis

Per-step joint commands reveal the shape of the model's planned motion:

- **Z-score analysis:** How many standard deviations is each step's output
  from the model's typical behavior? Outlier steps indicate uncertainty.
- **Forward kinematics:** Joint angles are converted to XYZ workspace
  coordinates (millimeters) so you can see exactly where the end effector
  went versus where it should have gone.
- **Gripper channel:** Is the gripper opening/closing at the right time?
  Gripper mode collapse — where the gripper never closes or closes too
  early — is a specific and diagnosable training failure.

### 4. VLM evaluation

Qwen3-VL looks at camera frames and provides a natural language verdict.
This is useful for initial classification but unreliable as ground truth.
The observatory uses it as one signal among four, not as the final answer.

## How to diagnose a failed trial

### Step 1: Run diagnose_trial

Ask the agent: "Diagnose the last trial"

Or call the tool directly:

```bash
curl -sk -u elastic:<password> \
  "https://<gpu-server>:31561/api/actions/connector/<connector-uuid>/_execute" \
  -X POST -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{
    "params": {
      "subAction": "callTool",
      "subActionParams": {
        "name": "diagnose_trial",
        "arguments": {}
      }
    }
  }'
```

The response includes a failure classification with evidence:

- **planning_failure** — model outputs a memorized trajectory regardless of
  scene. GPU kernels identical, CLIP distance low. Fix: more diverse training
  data.
- **perception_failure** — model fails to perceive the object or workspace.
  CLIP distance high, vision encoder anomalies. Fix: better camera setup,
  more training data at this position.
- **wrong_trajectory_shape** — model moves in the wrong direction or with
  wrong timing. Trajectory z-scores are elevated, FK shows wrong workspace
  path. Fix: check training data quality, re-record anomalous episodes.
- **gripper_mode_collapse** — gripper never activates or activates at wrong
  time. Gripper channel is flat or inverted. Fix: check `empty_cameras`
  config, verify gripper behavior in training data.
- **hardware_failure** — camera disconnected, motor stalled, USB
  re-enumerated. Non-model issue. Fix: check physical connections.

### Step 2: Drill deeper

Based on the classification, use targeted tools:

| Classification | Next tool | What to look for |
|---------------|-----------|-----------------|
| planning_failure | `get_gpu_kernel_profile` | Compare kernel profile against a successful trial |
| perception_failure | `check_scene_novelty` | CLIP distance and nearest training frames |
| wrong_trajectory_shape | `profile_trajectory` | Per-step joint commands, FK workspace path |
| gripper_mode_collapse | `profile_trajectory` | Gripper channel values over time |
| hardware_failure | `get_robot_status` | Camera frame ages, motor connectivity |

### Step 3: Identify the fix

Every diagnosis should end with a specific, testable action:

- "Record 10 episodes with the cube at position X" (not "add more data")
- "Set `empty_cameras=1` and retrain" (not "fix the gripper")
- "Replace the wrist camera cable — frame drops at step 200+" (not "check hardware")
- "Retrain with shoulder_pan coverage from -30 to +30 degrees" (not "improve generalization")

## Per-step telemetry

The inference telemetry system captures detailed data at every inference
step (30 per second during execution):

| Field | What it captures |
|-------|-----------------|
| Joint commands | 6-DOF position commands sent to motor |
| Z-scores | Per-joint deviation from running mean |
| Pipeline score | 4-component score: vision + attention + action + overall |
| Forward kinematics | End-effector XYZ and elbow XYZ in workspace coordinates (mm) |
| Model confidence | Action head output distribution statistics |
| Timestamp | Precise timing for correlation with GPU and ROS 2 telemetry |

Per-step documents are indexed to `robot-steps` in 30-step chunks for
efficient storage.

## Forward kinematics

The forward kinematics module converts joint angles to workspace
coordinates. This transforms abstract "joint 3 is at 45 degrees" into
concrete "the end effector is at (150, 30, 85) mm relative to the base."

This is critical for spatial analysis:

- Comparing the workspace path of a successful trial vs. a failed trial
- Measuring how far off the end effector was from the target
- Identifying which joint contributed most to a miss

Example: SmolVLA v8's first successful grip showed the end effector
reaching within 5mm of the cube center. Failed trials showed a consistent
34mm lateral offset — the model was aiming at the right height but the
wrong position. Forward kinematics made this obvious in the data; watching
the arm, it just looked like "a miss."

## Training data profiling

The `training_coverage_report` tool analyzes your training dataset for:

- **Joint coverage:** What range of each joint is represented? Gaps in
  coverage predict where the model will fail.
- **Position diversity:** How spread out are the object positions? Clustering
  in one position means the model only works there.
- **Trajectory consistency:** How similar are the demonstrations? High
  variance suggests inconsistent demonstrations that confuse the model.
- **Episode quality:** Any episodes with anomalous joint values, gripper
  behavior, or timing?

## Reproduction test

Before blaming a model for failing trials, verify it can reproduce its own
training data. The reproduction test:

1. Takes frames from training episodes
2. Feeds them through the inference pipeline
3. Compares the model's output to the recorded training actions
4. Reports mean absolute error per joint

A healthy model reproduces its training data within ~3 degrees MAE. If the
reproduction error is high, the model didn't learn effectively — retrain
with different hyperparameters or more data before investigating other
issues.

## The diagnostic mental model

```
Trial fails
    │
    ├─ Is it hardware? (get_robot_status)
    │   └─ Yes → fix hardware, re-run
    │
    ├─ Is the scene novel? (check_scene_novelty)
    │   └─ CLIP distance > 0.042 → record data at this position
    │
    ├─ Can the model reproduce training data? (reproduction test)
    │   └─ MAE > 5° → retrain, don't investigate further
    │
    ├─ What does the GPU show? (get_gpu_kernel_profile)
    │   └─ Kernels identical success/failure → planning failure
    │
    ├─ Where did the arm go? (profile_trajectory + FK)
    │   └─ Lateral offset → position coverage gap
    │   └─ Gripper flat → mode collapse
    │
    └─ What does the evaluator say? (diagnose_trial)
        └─ Corroborate with above signals, don't trust alone
```

Every path ends with a specific action. No path ends with "the model is
bad, try again."
