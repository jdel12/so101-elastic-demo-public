# Training and Iteration

This guide covers recording training data, training VLA models, evaluating
performance, and deploying improved checkpoints. This is the feedback loop
that turns a robot that sometimes works into one that reliably works.

## Recording training data

Training data is collected via teleoperation — you physically move the
leader arm while the follower mirrors the motion and cameras record.
LeRobot handles the recording pipeline.

### Setup

Both arms must be calibrated (see [build/01-robot.md](../build/01-robot.md))
and connected to the same machine. For multi-camera recording, all cameras
should be accessible from that machine.

```bash
lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyUSB0 --robot.id=my_follower --teleop.type=so101_leader --teleop.port=/dev/ttyUSB1 --teleop.id=my_leader --dataset.repo_id=your-hf-username/so101-pick-cube --dataset.single_task="Pick up the blue cube and place it in the center" --dataset.num_episodes=50 --dataset.fps=30
```

### How much data

| Goal | Episodes | Time |
|------|----------|------|
| First working model | 50 | ~30 minutes |
| Reliable at trained position | 100 | ~1 hour |
| Position generalization | 200+ across varied positions | ~2 hours |

Start with 50 episodes at a single object position. Get the model working
there first, then expand.

### Recording pro tips

**Consistency matters more than quantity.** 50 clean, consistent
demonstrations beat 200 sloppy ones. Keep your gripper behavior, approach
angle, and timing consistent across episodes.

**Camera position is part of the training data.** If you move a camera
between recording and inference, the model sees a different perspective
than what it trained on. Mount cameras rigidly and don't touch them.

**Record at 30 FPS.** The inference loop runs at 30 FPS by default. If
your training data is at a different FPS, the model's timing will be off.
This mismatch caused us significant debugging time.

**Plan your position coverage.** A model trained with the cube at one
position will fail if you move the cube 5cm to the side. We found that
our 65 episodes covered only 5.2% of the shoulder pan range — a 57-degree
ball in a space that should cover +/- 30 degrees.

To build position generalization, record episodes in a grid:

```
           -30°  -15°   0°  +15°  +30°
Near:       5     5     10    5     5
Mid:        5     5     10    5     5
Far:        5     5     10    5     5
```

~40 episodes across the grid, plus your 50+ at the center position.

### Multi-camera recording

For three cameras, add camera configuration to the record command. The
exact flags depend on your LeRobot version — check `lerobot-record --help`.

The key constraint is that all cameras must be on separate USB controllers
(see [build/01-robot.md](../build/01-robot.md)).

## Training

LeRobot supports several policy architectures. Here's what we've tested:

### ACT (Action Chunking with Transformers)

The recommended starting point. ~80M parameters, trains in 30 minutes on
a consumer GPU, data-efficient.

```bash
lerobot-train --dataset.repo_id=your-hf-username/so101-pick-cube --policy.type=act --output_dir=outputs/train/act_so101 --job_name=act_so101 --policy.device=cuda
```

### X-VLA

Our primary model. ~879M parameters, better generalization but slower
training. Requires 16GB VRAM.

Training X-VLA requires the `lerobot-train` script with VLA-specific
configuration. See the LeRobot documentation for VLA training details.

### SmolVLA

~200M parameters, faster inference but less robust. We experienced gripper
mode collapse across 7 consecutive versions until discovering the
`empty_cameras` configuration bug.

**Critical gotcha:** If fine-tuning from a pretrained base, the
`empty_cameras` parameter must match the pretrained model's configuration.
SmolVLA's pretrained base uses `empty_cameras=1` (it expects a camera slot
that may be empty). Setting `empty_cameras=0` drops 33% of visual context
and causes gripper mode collapse.

### Pi0

~3B parameters, flow-matching architecture. Produces noisy outputs that
require aggressive smoothing (exponential moving average with alpha=0.3).
We achieved one successful trial in 8 configurations, but the smoothing
cost precision.

### Training time expectations

| Model | GPU | 50 episodes | 200 episodes |
|-------|-----|-------------|--------------|
| ACT | RTX 4060 Ti | ~30 min | ~2 hours |
| X-VLA | RTX 4060 Ti | ~4 hours | ~12 hours |
| SmolVLA | RTX 4060 Ti | ~2 hours | ~6 hours |
| Pi0 | 2x RTX 4060 Ti | ~8 hours | ~24 hours |

## Evaluation

After training, evaluate the model to see if it actually works.

### Automated evaluation via MCP

Deploy the checkpoint (see below), then run trials through Agent Builder.
The system automatically:

1. Runs the model on the physical robot
2. Captures camera frames, joint trajectories, GPU telemetry
3. Evaluates with Qwen3-VL (VLM verdict)
4. Computes CLIP distance to training distribution
5. Indexes everything to Elasticsearch

Ask the agent: "Run 10 trials and report the success rate"

### Manual evaluation

You can also use LeRobot's built-in evaluation:

```bash
lerobot-rollout --robot.type=so101_follower --robot.port=/dev/ttyUSB0 --robot.id=my_follower --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model --inference.type=base
```

This runs the model autonomously without the MCP/Elasticsearch layer. Useful
for quick smoke tests before deploying to the cluster.

### Evaluation gotchas

**VLM evaluators hallucinate.** We found that Qwen3-VL sometimes reports
success when the robot clearly failed, or vice versa. The system mitigates
this by:

- Evaluating frames from all available cameras
- Using the highest-confidence result
- Retrying low-confidence evaluations

Despite these mitigations, don't trust the VLM verdict as ground truth.
Cross-reference with trajectory analysis and CLIP distance.

**Reproduction test:** Before blaming the model, verify it can reproduce
its own training data. The reproduction test feeds training frames through
the inference pipeline and checks that the output matches the training
actions within a tolerance. We consider < 5 degrees mean absolute error
a pass.

## Deploying a new model

### Prepare the checkpoint

```bash
# On the GPU server
mkdir -p ~/models
ln -sf ~/outputs/train/<run-name>/checkpoints/last/pretrained_model \
  ~/models/pretrained_model
```

### Hot-swap via MCP

If the robot-brain pod is running, swap the model without restarting:

```bash
# Via Agent Builder: "Swap to model pretrained_model"
# Or via API:
curl -sk -u elastic:<password> \
  "https://<gpu-server>:31561/api/actions/connector/<connector-uuid>/_execute" \
  -X POST -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{
    "params": {
      "subAction": "callTool",
      "subActionParams": {
        "name": "swap_model",
        "arguments": {"model_path": "/models/pretrained_model"}
      }
    }
  }'
```

### Cold deploy

If you need to change the model path in the manifest:

1. Update `MODEL_PATH` in `deploy/robot/k8s/robot-brain.yaml`
2. Delete and re-apply:

```bash
sudo kubectl delete pod -n robot -l app=robot-brain
sudo kubectl apply -f deploy/robot/k8s/robot-brain.yaml
```

## The training feedback loop

The system supports an iterative improvement cycle:

1. **Run trials** — collect success/failure data with full telemetry
2. **Diagnose failures** — use the observatory to understand why specific
   trials failed (see [03-diagnostics.md](03-diagnostics.md))
3. **Identify gaps** — use `training_coverage_report` to find where your
   training data is sparse
4. **Record more data** — targeted episodes in the identified gap areas
5. **Retrain** — with the expanded dataset
6. **Deploy and evaluate** — hot-swap the new model and run trials
7. **Compare** — use `compare_model_kernels` and `query_inference_stats`
   to verify improvement

Each iteration adds data where the model is weakest, not just more data
everywhere. The telemetry tells you exactly where to focus.
