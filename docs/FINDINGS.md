# Project Findings — Chronological

Everything that happened building and operating an AI robot with Elastic,
in the order it happened. Use this document to understand the full arc before
reading any individual article — it's the raw material the articles draw from.

Last updated: 2026-06-12

---

## How to read this

Events are grouped by session. Each entry has a category tag:

- **[infra]** — cluster, pods, networking, hardware
- **[model]** — training, evaluation, inference behavior
- **[diagnosis]** — root cause analysis, hypothesis testing
- **[obs]** — observability pipeline, metrics, dashboards
- **[decision]** — architecture or narrative choice
- **[gotcha]** — something that cost hours and will cost you hours too

Significance is in *italics* after each entry.

---

## Session 0 — Baseline (2026-05-15 to 2026-05-26)

Goal: get ES 9.4.1 + Agent Builder + robot running end-to-end on two machines.

### Architecture decisions

- **[decision]** ADR-0001: Local embedding via Ollama (nomic-embed-text, 768 dims). *No external API dependency, air-gapped pipeline.*
- **[decision]** ADR-0002: Workflows over agent-driven flywheel. *Deterministic scheduled automation > unpredictable LLM decisions for data curation.*
- **[decision]** ADR-0003: Two-repo strategy. *ebpf-gpu-research (31 exploratory themes) stays separate from so101-elastic-demo (deployable teaching instrument).*
- **[decision]** ADR-0004: GPU observability as layered depth (Refinery). *Rust daemon with 10 solution packs replaces Python prototype.*
- **[decision]** ADR-0005: Remove Painless kernel classifier from ingest pipeline. *Classification at query time in Python, not at index time in Painless.*
- **[decision]** ADR-0006: Four-part article series. *Build → SRE → GPU → Experiments.*

### Infrastructure

- **[infra]** Fresh ES 9.4.1 + Kibana 9.4.1 installed. ECK 3.4.0 operator. k3s cluster green (cortex + NUC). *Baseline cluster established.*
- **[infra]** Trial license started 2026-05-26 (expires 2026-06-25). *30-day clock running.*
- **[infra]** MCP v8 server deployed with 11 robot tools. Connector UUID `e68cab04-1110-48c7-9a38-e0a574e7a17a`. *Agent Builder can call robot.*
- **[infra]** 8 index templates created, 14 saved objects imported. 3 Workflows deployed (Flywheel Curator hourly, Flywheel Stats 30min, Model Comparison manual). *Automation cadence established.*
- **[infra]** NUC USB ports mapped: ACM0=leader arm, ACM1=follower arm. Both arms calibrated. *Hardware ready for teleoperation and inference.*
- **[infra]** Camera node running on NUC at 14 Hz on cam0. Motor node listening for `/joint_commands`. *Edge nodes operational.*
- **[gotcha]** Python quoting bugs in rebuild scripts. *Shell escaping through SSH layers is fragile.*
- **[gotcha]** Connector secrets must be manually re-entered after saved object import. *ES export doesn't include secrets.*

### First verification

- **[model]** Denormalization test successful: joint commands in degree range (6.67, -95.17, 79.52, 37.31, 1.53, 29.0). *Inference pipeline math is correct.*
- **[infra]** MCP v13 deployed: host-mounted source (no Docker rebuild needed), pre-execution CLIP embedding added. *Development velocity dramatically improved.*

---

## Session 1 — First Experiments (2026-05-27 to 2026-06-01)

Goal: create failures, measure everything, understand what the model actually learned.

### The first end-to-end success

- **[model]** Agent Builder E2E working: command → MCP → X-VLA → robot → Qwen3-VL eval → ES. 25.3s cycle. *Full distributed inference pipeline verified on physical robot.*
- **[model]** X-VLA v1 tested at trained position: 80% success (3/4 trials), gripper closes to 6°. *Baseline established.*

### SmolVLA gripper mode collapse

- **[diagnosis]** SmolVLA gripper mode collapse root cause: `chunk_size=50` averages bimodal gripper signal. Open ~40 steps + closed ~10 steps → predicted mean ~20° (neither open nor closed). *Paper uses chunk_size=10. Default config is wrong for this task.*
- **[model]** SmolVLA v5 training launched with chunk_size=10 fix. *Hypothesis test on action chunking parameter.*
- **[diagnosis]** SmolVLA v1-v7: all 7 variants failed. Kernel autopsy showed 12x less compute than X-VLA (388ms vs 4,656ms). *Failure is in the weights, not the compute budget. SmolVLA parked.*

### LED lighting experiments

- **[infra]** LED lighting rig set up with 5 presets: ambient, warm_red, cool_blue, low_light, prism. *Environmental condition variation testing enabled.*
- **[model]** V1 LED trial results: blue 100%, red 60% (hesitation), green 100%, strobe 50%, low_light 100%. *V1 is robust to static color shifts, disrupted by dynamic strobing.*

### Position and generalization

- **[model]** V1 at trained position: 80%. V1 at any shifted position: 0%. *Model learned a trajectory replay, not a spatial policy.*
- **[model]** Multi-object experiments: V1 freezes when blue cube present but not at trained position. *Position+color binding — conflicting signals paralyze the model.*
- **[model]** Language conditioning confirmed dead: V1/V4 ignore text instructions ("pick up green cube" → picks blue). *879M parameters too small to retain language understanding after single-task fine-tuning.*

### CLIP embedding pipeline

- **[infra]** Jina CLIP v2 self-hosted on GPU 1 (port 8900, 1024-dim embeddings). *Replaced Ollama nomic-embed-text for multimodal scene analysis.*
- **[obs]** CLIP distance at trained position: 0.028-0.036. At shifted positions: 0.038-0.040. *20-35% separation — pre-execution confidence signal works.*
- **[obs]** Overhead camera too coarse for position prediction. Wrist camera required. Pre-execution embedding essential (post-execution shows arm, not scene). *Design constraints for confidence scoring.*

### Model comparison baselines

- **[model]** Phase 1 baseline: 80+ trials across 3 models at 3 positions. V1: 80% trained. V2: 40% trained (catastrophic forgetting from fine-tuning V1). V3: 20% trained (curation hypothesis failed — removing data hurt). *More data > better data for this model size.*

### Curation hypothesis — FAILED

- **[model]** Methodology: embed 65 episodes with Jina CLIP v2, UMAP projection, identify 3 clusters (varied-early, focused-early, focused-late), remove high-variance cluster, retrain as V3. *Sound methodology, wrong hypothesis.*
- **[model]** V3 result: worst performer at 20%. Removing the varied-early episodes removed approach diversity. *Data curation at this scale hurts more than it helps. The model needs volume, not purity.*

### GPU observability deployed

- **[infra]** Python gpu_obs prototype deleted (45 files). Refinery solution packs installed: gpu-profiling, gpu-inference, gpu-robotics. *Production Rust daemon replaces prototype.*
- **[infra]** NVIDIA driver upgraded 550→580 (CUDA 13.0) for cuVS support. *GPU driver upgrade for vector search acceleration.*
- **[obs]** cuVS benchmark: 6.1x speedup (GPU 11.6s vs CPU 70.9s, 50K × 1024-dim vectors). *GPU-accelerated HNSW indexing works, but ES 9.4 defaults to bbq_hnsw which cuVS cannot accelerate.*
- **[gotcha]** Must set `type: hnsw` explicitly (not `bbq_hnsw` default). Check `index_build_count` — `available: true` does not mean GPU is actually used. *Cost hours of false-positive benchmarking.*

### SRE story pivot

- **[decision]** Flywheel deferred. SRE + operational observability becomes primary narrative. *The interesting story is "we built a robot, it failed, and we used Elastic to figure out exactly why."*

---

## Session 2 — Failure Decomposition (2026-06-02 to 2026-06-03)

Goal: diagnose root cause of V1 failures using GPU kernel traces, CLIP distance, and VLM evaluations.

### V5/V6 training saga

- **[model]** V5/V6 training attempted with `train_xvla_v2_optimized.py` wrapper: FAILED. Wrapper designed for fine-tuning from checkpoint, not base model adaptation. Action dim didn't adapt (20→6). *Different code paths for base vs. checkpoint training.*
- **[model]** Retrained V5/V6 using `lerobot-train` directly (matching V1's original command). Loss converging. *Always use the same training path that produced your working model.*
- **[model]** V5 produced zero motor output during inference. Hours of false diagnosis. *Actual cause: two stacked infrastructure bugs (see below).*
- **[diagnosis]** V5 root cause #1: `inference_node.py` not host-mounted. Debug prints never deployed. *Critical debugging gotcha — code changes weren't reaching the container.*
- **[diagnosis]** V5 root cause #2: Camera USB re-enumeration after driver reboot (/dev/video0→video1). *Device paths shift on every USB event. Check before every session.*
- **[gotcha]** MCP `swap_model` race condition: swap returned before model loaded, `execute_task` sent queued "stop" command → killed inference. *Fix: swap waits for completion, stop is conditional.*

### V6 evaluation

- **[model]** V6 (SART augmented, 325 episodes): 4/5 at trained position (80%, matches V1), 2/2 small lateral (incidental), 0/1 far lateral. *Augmentation didn't help generalization.*
- **[diagnosis]** V6 failure mode completely different from V1: top-down approach instead of side approach, gripper never closes (28° vs V1's 6°). *SART augmentation corrupted the approach trajectory rather than teaching adaptation.*

### Model deployment ops

- **[infra]** `deploy-model.sh` built: automated config check → offline validation → pre-flight → deploy → smoke test → rollback. *Formal deployment ops to separate infrastructure bugs from model bugs.*
- **[infra]** Model deployment runbook published. *Operational procedures documented.*

### D2: The failure decomposition breakthrough

- **[diagnosis]** GPU kernel profiles IDENTICAL between V1 success and V1 failure: 60K kernels, ~4.8s per 500-step trial. Vision encoder (Florence2 SGEMM): 78.6%. Attention (flash): 11.4%. Action head: 2.3%. *The model computes the exact same thing regardless of whether the cube is reachable. Planning failure confirmed.*
- **[obs]** CLIP distance boundary analysis: success ≤0.038, failure ≥0.040 for lateral shifts. But axis-dependent: forward displacement 0.029 (invisible to wrist camera), rotation 0.034 (CLIP can't see orientation). *CLIP necessary but not sufficient for spatial prediction.*
- **[diagnosis]** V1 failure taxonomy complete: the model replays a memorized trajectory. It does not perceive cube position, adapt its approach, or use spatial reasoning. Success happens when the cube is close enough to the memorized path. *This is the core finding of the project.*

### Jitter root cause

- **[obs]** Jitter in robot arm motion: 5-second periodic dips in kernel count histogram. GPU executes in bursts (3-6 steps in 200-300ms) with 500-700ms gaps. *Visible in Refinery kernel traces.*
- **[diagnosis]** GC hypothesis tested: `gc.disable()` at task start — stutter persisted unchanged. *Python garbage collection ruled out.*
- **[obs]** `robot-steps` index analysis: 30-step periodicity exactly matches X-VLA `chunk_size: 30` config. *Root cause: action chunking. Every 30 steps = 325ms full forward pass. Not a GPU bottleneck — architectural.*
- **[obs]** Per-step GPU budget: 9.7ms (29% of 33.3ms at 30 FPS). *GPU is not the bottleneck. The model is fast enough; the chunking pattern creates the stutter.*

### Observability infrastructure

- **[obs]** CUPTI overhead measured: 0.5% (24.19 Hz with profiling vs 24.32 Hz without). *Negligible — Refinery shim stays on permanently.*
- **[obs]** Refinery CUPTI buffer fix: `GPUCOLLECTOR_ACTIVITY_BUFFER_SIZE=65536`. *Default 4MB buffer fills every ~4.5s, correlator evicts launches in 5s — race condition.*
- **[infra]** MCP v14 deployed: 6 GPU diagnostic tools added (17 total). Refinery index patterns corrected (`gpu-profiling-*` → `logs-gpu.*`). *GPU diagnostic tools now query actual data.*
- **[diagnosis]** Refinery was WORKING all along: 223K kernel traces in `logs-gpu.kernel-default`. MCP tools had wrong index patterns. *Multi-hour false diagnosis from a typo in index patterns.*
- **[infra]** MCP v15 deployed: 22 tools. Diagnostic glossary embedded in Agent Builder system prompt. *Agent can now reason about GPU kernel profiles, CLIP distances, and failure classifications.*
- **[gotcha]** Motor node hung after 38h. Silent type mismatch: subscribes to Float32MultiArray, inference publishes Float64MultiArray. *DDS silently drops mismatched messages.*
- **[gotcha]** Camera USB autosuspend killed availability. Fix: udev rule `ATTR{power/control}="on"` for camera VID 0c45. *Linux power management and USB cameras don't mix.*
- **[infra]** All MCP tools confirmed with real Refinery data. `diagnose_trial` fuses robot-inference + kernels + CLIP + VLM eval. `compare_model_kernels` does side-by-side kernel profiles. *Multi-signal diagnostic tools operational.*

---

## Session 3 — ROS2 Enterprise Integration (2026-06-04 to 2026-06-05)

Goal: build production-grade ROS2 observability for fleet monitoring.

### ROS2 v0.2.0

- **[infra]** ROS2 integration rewritten: C++ collector (rclcpp) + Rust bulk shipper. Multi-node DaemonSets on cortex + NUC. *Enterprise-grade replacement for prototype.*
- **[infra]** Rust shipper: 500-doc batches, gzip compression, exponential backoff, 5000-doc buffer with drop-oldest backpressure, `/healthz` + `/metrics` on port 8090. *Production patterns.*
- **[infra]** Collector hardened: SIGTERM handler, lifecycle events, max_topics cap (100), self-monitoring health every 60s. *Graceful shutdown and self-awareness.*
- **[obs]** Continuous transform `ros2-fleet-topic-health`: 1-min rollups by fleet + topic. 52:1 rollup ratio at observed volumes, 300x projected at 10K robot scale. *Cost-optimized fleet storage.*
- **[obs]** 3 enterprise dashboards deployed: Fleet Status (10 panels), Topic Health (9 panels), Collector Ops (10 panels). *Full ROS2 fleet observability.*
- **[infra]** Stress test: cortex 45m CPU/31Mi, NUC 19m CPU/32Mi, 100K+ docs, zero drops, zero errors in 45 min. 17 shipper unit tests pass. *Production-validated.*
- **[infra]** ILM tuned: 7d raw metrics, 365d rollups. *Optimized retention for scale.*
- **[gotcha]** DDS namespace edge case: cross-namespace node list is participant-scoped. Topics propagate via multicast but node discovery doesn't. *One robot = one machine at this scale.*

### ML jobs and detection rules

- **[infra]** 4 Refinery ML jobs installed (fixed `datafeed` → `datafeed_config` for ES 9.x). 8 detection rules via API. *Anomaly detection infrastructure ready.*
- **[infra]** All 26 MCP tools registered in Agent Builder. Agent ID: `so101-flywheel-controller`, name: "SO-101 Diagnostic Agent". *Full tool surface wired.*

---

## Session 4 — Camera Migration (2026-06-04)

Goal: move both cameras from NUC to cortex to resolve USB bandwidth issues.

- **[infra]** Both cameras migrated from NUC to cortex. Containerized as k8s pods. cam0 (wrist) on Bus 5 (/dev/video2), cam1 (overhead) on Bus 1 (/dev/video0). Separate USB controllers verified. *Resolved NUC Bus 01 bandwidth limitation (two 500mA cameras on one hub exceeded bandwidth).*
- **[diagnosis]** USB diagnosis complete: NUC Bus 01 can't sustain two cameras. BPF warm reset tracking confirmed. *Root cause identified and resolved by migration.*
- **[obs]** Camera auto-recovery: after 10 consecutive dropped frames, release and reopen device. If reopen fails, pod exits for k8s restart. *Self-healing camera containers.*

---

## Session 5 — Pi0 Exploration (2026-06-09 to 2026-06-12)

Goal: test Pi0 (GR00T N1.7) as alternative to X-VLA, close GPU observation gaps.

### Pi0 smoke test

- **[model]** GR00T N1.7 + LoRA smoke test PASSED on cortex GPU 1 (RTX 4060 Ti 16GB, ~10-11GB peak). Qwen3-VL backbone swap (source patches auto-swap gated Cosmos → ungated Qwen3-VL). *Alternative model architecture viable on consumer hardware.*
- **[gotcha]** Config patches alone miss code paths that hard-code Cosmos. Source-level patches in Isaac-GR00T required. *Framework-level intervention needed.*

### Pi0 expert-only training

- **[model]** 5000 steps, loss 0.506→0.261, 578M/4B params, ~2h single GPU. *LoRA adapter training fast and cheap.*
- **[model]** Results: goal-directed motion toward cube (promising), inconsistent (went left on 2nd run), jagged at chunk boundaries. *Expert-only insufficient — model needs full parameter updates.*
- **[obs]** EMA smoothing + `n_action_steps=15` improved motion quality. Must apply postprocessor (z-scores → degrees). *Inference pipeline gotchas for Pi0.*

### Pi0 full fine-tune

- **[model]** 2500 steps, 7h 26m, loss 0.845→0.155, 3B params (full). DeepSpeed ZeRO-3 + CPU offloading across 2× RTX 4060 Ti 16GB. *Large model fine-tuning on consumer hardware — possible but requires deep framework patches.*
- **[gotcha]** 6 monkey-patches required: (1) `_initialize_weights` no-op, (2) skip fp32 cast-back, (3) force bf16 + grad_norm fix + end-save, (4) `zero3_linear_wrap` dtype cast, (5) `param_groups` KeyError fix, (6) standalone DeepSpeed JSON config for `torch_autocast`. *Each one cost 30-60 min to diagnose.*
- **[gotcha]** `save_16bit_model()` writes PyTorch format with `.safetensors` extension. Header reads as garbage (551 TB). Post-convert required: `torch.load()` then `safetensors.torch.save_file()`. *Silent corruption in model saving.*
- **[model]** Checkpoint: 7.5 GB bf16, 778 tensors. Pi0-eval pod deployed on GPU 0, all keys loaded, 4B params, inference ready at 30 FPS. *Awaiting physical evaluation against X-VLA v1 baseline (80%).*

### Search Labs submission

- **[decision]** Issue #2087 filed on `elastic/search-labs-elastic-co`: 3-part series (Build, SRE, GPU). Part 4 (Experiments) shelved — content may fold into Parts 2-3. *Publication venue and structure decided.*

---

## Session 6 — Article Reorientation (2026-06-12)

Goal: step back, review all content, decide what goes where.

- **[decision]** Reorientation under consideration: move build/setup content to public repo, keep Search Labs articles focused on Elastic value. Create chronological findings document (this file) as the single source of truth for what happened. *In progress.*

---

## Key findings summary

These are the findings that matter most for the articles. Each one is a
discrete insight that a reader could take away.

### On the robot

1. **X-VLA v1 is the only working model.** 80% at trained position, 0% everywhere else. 7 SmolVLA variants and 5 X-VLA fine-tune variants all failed or regressed.
2. **The model replays a memorized trajectory.** GPU kernel profiles are identical between success and failure. Vision encoder runs, but the action head outputs the same motor commands regardless of scene. Planning failure, not perception failure.
3. **Fine-tuning from a checkpoint causes catastrophic forgetting.** V2 (40%) < V1 (80%). Always train from pretrained base.
4. **Data curation hurts at this scale.** V3 (curated, 20%) < V1 (all data, 80%). Small datasets need volume, not purity.
5. **SART augmentation corrupts trajectories.** V6 approached from above instead of the side. The augmentation changed the approach vector, not just the start position.
6. **Language conditioning is dead after fine-tuning.** 879M parameters can't retain text understanding through single-task training.
7. **Multi-object scenes paralyze the model.** Position+color binding means conflicting signals (right color, wrong position) cause the arm to freeze.

### On observability

8. **CLIP distance predicts failure before execution.** Wrist camera frame → Jina CLIP v2 → kNN distance to training embeddings. Success ≤0.038, failure ≥0.040 (lateral). But axis-dependent: forward displacement invisible (0.029).
9. **Kernel-level GPU traces reveal what nvidia-smi hides.** One "85% utilization" number becomes thousands of classified kernels per second: vision encoder 78.6%, attention 11.4%, action head 2.3%.
10. **Jitter root cause is action chunking, not GPU contention.** 30-step chunk_size = 325ms forward pass every second. Per-step GPU budget is only 9.7ms (29% of frame time). The model is fast; the architecture stutters.
11. **CUPTI profiling overhead is negligible.** 0.5% (24.19 Hz vs 24.32 Hz). Leave it on in production.
12. **Multi-signal diagnosis works.** Fusing VLM evaluation + kernel traces + CLIP distance + trajectory data classifies failures as planning/perception/hardware with evidence, not guessing.

### On infrastructure

13. **Consumer GPUs run production AI inference.** 2× RTX 4060 Ti 16GB. GPU 0 for inference (30 FPS), GPU 1 for Ollama + Jina CLIP. No A100 needed.
14. **k3s on two machines is a real cluster.** cortex (GPU server) + NUC (motor control). DDS over gigabit LAN. FastDDS SHM disabled for cross-version compatibility (Humble ↔ Jazzy).
15. **Host-mounting source code eliminates the rebuild cycle.** Changes to `mcp_server.py` and `inference_node.py` take effect on pod restart. Development velocity increased 10×.
16. **USB cameras need separate controllers.** Two cameras on one USB root hub exceeds bandwidth. Migrated both to cortex on separate buses.
17. **ROS2 fleet monitoring at scale needs continuous transforms.** 52:1 rollup ratio observed, 300× projected at 10K robots. Raw metrics retained 7 days, rollups 365 days.

### On training

18. **Pi0 full fine-tune is possible on consumer hardware.** 3B params, DeepSpeed ZeRO-3 + CPU offload, 2× RTX 4060 Ti, 7.5h. But 6 monkey-patches required.
19. **cuVS delivers 6.1× speedup for vector indexing.** But ES 9.4 defaults to `bbq_hnsw` which cuVS can't accelerate. Must explicitly set `type: hnsw`.
20. **Every model training run surfaces new framework bugs.** Pi0 needed 6 patches. X-VLA needed 5. The gotchas compound across framework versions.

---

## Gotchas reference

The full gotchas list (90+ items across 19 categories) is maintained in the
article drafts and will be published as a standalone reference in this
repository. Categories: Hardware/USB, LeRobot CLI, SmolVLA, X-VLA, Kubernetes,
Docker, MCP, Motor/Servo, Embeddings, Model Generalization, Episode Curation,
Startup, Evaluation, Model Swap, LED/Lighting, Multi-Object, Training, CLIP,
cuVS.
