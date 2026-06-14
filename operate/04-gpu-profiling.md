# GPU Profiling

This guide covers kernel-level GPU analysis using Refinery and CUPTI. This
is the layer that transforms "85% GPU utilization" into a complete picture
of what the model is doing on the GPU, step by step, kernel by kernel.

## What nvidia-smi won't tell you

`nvidia-smi` reports a single utilization percentage. During VLA inference,
that number hovers around 80-90%. It tells you the GPU is busy but nothing
about what it's busy doing.

With CUPTI kernel tracing, one inference step becomes hundreds of
classified kernel executions:

| Component | Time share | What it does |
|-----------|-----------|--------------|
| Vision encoder | 78.6% | ResNet backbone processing camera frames into feature maps |
| Attention | 11.4% | Transformer cross-attention between vision and language tokens |
| Action head | 2.3% | MLP that outputs 6-DOF joint commands |
| Memory ops | 7.7% | Data movement between CPU RAM and GPU VRAM |

This decomposition is the foundation for understanding model behavior at
the compute level.

## Kernel classification

Refinery classifies GPU kernels by matching kernel names to known patterns:

- `volta_*`, `turing_*`, `ampere_*` — architecture-specific GEMM kernels
  (matrix multiply, the core of neural network computation)
- `*conv*`, `*winograd*` — convolution kernels (vision encoder)
- `*attention*`, `*softmax*`, `*flash*` — attention kernels (transformer)
- `*elementwise*`, `*reduce*`, `*layernorm*` — normalization and activation
- `*memcpy*`, `*memset*` — memory operations

Each kernel execution is logged with:

- Kernel name
- Duration (nanoseconds)
- Grid size (how many thread blocks)
- Block size (threads per block)
- Component classification
- Correlation ID (links to the CUDA API call that launched it)

## Using the GPU profiling tools

### get_gpu_kernel_profile

Returns the kernel decomposition for a time window or specific trial:

```
Vision encoder:  78.6%  (avg 0.42ms per kernel, 1,247 kernels)
Attention:       11.4%  (avg 0.18ms per kernel, 312 kernels)
Action head:      2.3%  (avg 0.08ms per kernel, 89 kernels)
Memory:           7.7%  (avg 0.15ms per kernel, 203 kernels)
```

**Key insight:** If the vision encoder dominates time but the model
still fails, the model is processing visual information but not using it
effectively. If the action head is anomalously fast, it may be producing
degenerate outputs (all zeros, clamped values).

### get_inference_breakdown

Time allocation across the full inference pipeline per step:

1. Frame capture (camera → CPU)
2. Preprocessing (resize, normalize, move to GPU)
3. Vision encoding (GPU compute)
4. Transformer inference (GPU compute)
5. Action decoding (GPU compute)
6. Postprocessing (smoothing, clamping, move to CPU)
7. Motor command publication (CPU → ROS 2)

The breakdown reveals where latency hides. In our setup, preprocessing
(step 2) consumed more time than expected because CPU-GPU data transfer
was not pinned-memory optimized. This doesn't affect accuracy but limits
throughput.

### get_control_loop_status

Reports the inference loop's real-time performance:

- **Target frequency:** 30 Hz (one inference step every 33.3ms)
- **Actual frequency:** measured Hz over the time window
- **Jitter:** standard deviation of loop period
- **Duty cycle:** percentage of loop time spent in GPU compute vs. idle

High jitter means inconsistent timing, which causes jerky motion. Common
causes: thermal throttling, other processes competing for GPU, memory
pressure causing swap.

### compare_model_kernels

Side-by-side GPU comparison between two models:

```
                        X-VLA v1     SmolVLA v8
Vision encoder kernels:  1,247         423
Attention kernels:         312         187
Action head kernels:        89          45
Total kernel time:        12.4ms      4.8ms
Steps per second:           30          30
```

This reveals architectural differences at the compute level. X-VLA uses
more kernels (larger model) but both achieve the same inference rate — the
smaller model has idle GPU time between steps.

### profile_inference

Duty cycle summary over a time window:

- GPU compute time vs. total time
- Peak vs. average utilization
- Memory high-water mark
- Thermal throttling events (if any)

## Cross-model analysis

GPU kernel profiles are powerful for comparing models because they're
objective — they measure what the GPU actually did, not what the model
claims to have done.

### Memorized trajectories

Our most important finding: when we compared kernel profiles between
successful and failed X-VLA trials, they were **identical**. Same kernel
count, same durations, same component distribution.

This proved the model was replaying a memorized trajectory — the vision
encoder ran but its output didn't change the action head's behavior. The
model wasn't adapting to the scene; it was executing a fixed sequence
regardless of input.

This insight comes only from kernel traces. Watching the arm, you see
"it moved but missed." Looking at the GPU, you see "it computed the same
thing both times." That's the difference between guessing and knowing.

### Architecture comparison

Different VLA architectures use GPU resources differently:

- **X-VLA** (879M params): fp32 inference, large vision encoder, heavy
  attention, small action head. 12.4ms per step.
- **SmolVLA** (200M params): bf16 inference, efficient vision encoder,
  fewer attention heads. 4.8ms per step. But gripper mode collapse.
- **Pi0** (3B params): flow-matching architecture, iterative denoising
  in action space. 28ms per step on 2 GPUs. Noisier outputs.

Kernel traces show exactly where the compute goes in each architecture,
helping you understand the tradeoffs between model size, speed, and
capability.

## Refinery operations

### Verifying the CUPTI shim

Check that Refinery is tracing the inference container:

```bash
# Look for GPU kernel data
curl -sk -u elastic:<password> \
  "https://<gpu-server>:31920/logs-gpu.kernel-*/_count"
```

If the count is 0, verify:

1. The `CUDA_INJECTION64_PATH` env var in the inference container points
   to the actual shim file
2. The Refinery shim volume is mounted in the inference pod
3. The inference container was restarted after Refinery was installed

### Overhead

CUPTI tracing overhead is approximately 0.5% on inference throughput.
In our setup, this means 30.0 FPS drops to 29.85 FPS — effectively
invisible. The trace data is worth the cost.

### Data volume

A single trial (~600 steps at 30 FPS) produces roughly 50,000-100,000
kernel trace documents. At continuous operation, budget approximately 10GB
per day of kernel data in Elasticsearch. Use ILM (Index Lifecycle
Management) to roll over and delete old kernel data:

```
Hot:    7 days  (current trials, active analysis)
Warm:   30 days (historical comparison)
Delete: 90 days
```
