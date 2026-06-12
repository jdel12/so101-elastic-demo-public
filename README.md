# Operating AI Robots with Elastic

A distributed AI robotics system where a natural language command in Kibana
Agent Builder drives a physical SO-101 robot arm through MCP tools, ROS 2,
and vision-language-action models on Kubernetes. Everything is observable
through Elasticsearch and Kibana.

**Companion to the Search Labs article series** — the articles cover
what we learned; this repository has everything you need to build it yourself.

> [Part 1: An LLM controls a physical robot through MCP tools](#) (coming soon)
> [Part 2: SRE for robots — operating AI systems with Elastic](#) (coming soon)
> [Part 3: Kernel-level GPU observability for AI inference](#) (coming soon)

## What this does

You type a command in Kibana Agent Builder. An LLM reasons about what to do,
calls MCP tools, a vision-language-action model runs inference on a consumer
GPU, ROS 2 sends motor commands over the wire, and a physical robot arm picks
up a cube. 25 seconds, end to end.

Then Elastic tells you exactly what happened — task success rates, GPU kernel
traces, robot joint trajectories, CLIP-based confidence scoring, and automated
failure diagnosis that fuses multiple signals to classify failures as planning,
perception, or hardware.

## Architecture

```
Cortex (Ubuntu 24.04, k3s control, 2x RTX 4060 Ti 16GB)
├── robot-brain pod (hostNetwork)
│   ├── mcp-server    (FastMCP, 26 tools, host-mounted source)
│   └── inference     (X-VLA on GPU 0, 30 FPS)
├── camera-wrist pod  (cam0, /dev/video2, Bus 5)
├── camera-overhead   (cam1, /dev/video0, Bus 1)
├── Ollama            (Qwen3-VL on GPU 1, auto-evaluation)
├── Jina CLIP v2      (GPU 1, 1024-dim embeddings)
├── Elasticsearch 9.4.1  (ECK 3.4.0)
├── Kibana 9.4.1
└── Refinery          (Rust GPU obs daemon, CUPTI kernel tracing)

        ↕ DDS (ROS2 Humble ↔ Jazzy) over gigabit LAN

NUC (Ubuntu 22.04, k3s agent)
├── motor-node pod    (SO-101 follower arm, Feetech servos)
└── SO-101 arms       (leader + follower)
```

Both cameras are on cortex (separate USB controllers). Camera containers use
ROS 2 Humble; inference uses Jazzy. FastDDS SHM is disabled to force UDP
transport between container versions.

## Repository contents

```
docs/
├── FINDINGS.md          # Chronological project narrative (start here)
├── architecture/        # System diagrams
├── gotchas/             # 90+ categorized operational gotchas
└── articles/            # Search Labs article drafts

src/
├── cortex/brain/        # MCP server (26 tools) + inference node
├── edge/camera/         # Camera node with USB auto-recovery
└── edge/motor/          # Motor control bridge (NUC)

infra/
├── k8s/                 # All Kubernetes manifests
└── docker/              # Container images

scripts/
├── rebuild/             # ES cluster rebuild (00-preflight through 05-verify)
├── reset-arm.sh         # Smooth 90-step arm reset
└── deploy-model.sh      # Safe model deployment with rollback
```

## Key findings

These are the most interesting things we discovered. The full chronological
narrative is in [docs/FINDINGS.md](docs/FINDINGS.md).

1. **The model replays a memorized trajectory.** GPU kernel profiles are
   identical between success and failure — the vision encoder runs but the
   action head outputs the same commands regardless of scene. Planning failure,
   not perception failure.

2. **CLIP distance predicts failure before execution.** Wrist camera frame →
   kNN distance to training embeddings. Success ≤0.038, failure ≥0.040. But
   it's axis-dependent — forward displacement is invisible to the wrist camera.

3. **Kernel-level GPU traces reveal what nvidia-smi hides.** One "85%
   utilization" number becomes thousands of classified kernels: vision encoder
   78.6%, attention 11.4%, action head 2.3%.

4. **Multi-signal diagnosis works.** Fusing VLM evaluation + kernel traces +
   CLIP distance + trajectory data classifies failures with evidence, not
   guessing.

5. **Consumer GPUs run production AI inference.** 2× RTX 4060 Ti 16GB — no
   A100 needed. CUPTI profiling overhead is 0.5%.

## Hardware requirements

| Component | What we used | Minimum |
|-----------|-------------|---------|
| GPU server | 2× RTX 4060 Ti 16GB, 64GB RAM | 1× GPU with 16GB VRAM |
| Edge node | Intel NUC, 16GB RAM | Any Linux box with USB |
| Robot arm | SO-101 (leader + follower) | Any LeRobot-compatible arm |
| Cameras | 2× USB webcams (1080p) | 1 camera minimum |
| Network | Gigabit Ethernet | Required for DDS |

## Software stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Elasticsearch | 9.4.1 | System of record |
| Kibana | 9.4.1 | Dashboards, Agent Builder |
| ECK | 3.4.0 | Kubernetes operator |
| k3s | 1.34.4 | Lightweight Kubernetes |
| ROS 2 | Humble + Jazzy | Robot messaging |
| FastMCP | latest | MCP tool server |
| Refinery | latest | GPU kernel tracing |
| X-VLA | v1 | Vision-language-action model |
| Ollama | latest | Qwen3-VL for auto-evaluation |
| Jina CLIP v2 | latest | Scene embeddings |

## Getting started

Detailed setup instructions are in the article series. The short version:

1. Set up k3s on your GPU server and edge node
2. Deploy Elasticsearch and Kibana via ECK
3. Apply the Kubernetes manifests from `infra/k8s/`
4. Register MCP tools in Kibana Agent Builder
5. Start talking to your robot

For the full build story including 90+ gotchas we discovered along the way,
see [docs/FINDINGS.md](docs/FINDINGS.md).

## Elasticsearch indices

| Index | What it stores |
|-------|---------------|
| `robot-inference` | Trial results, success/failure, confidence, timing |
| `robot-steps` | Per-step joint commands (30-step chunking signature) |
| `robot-episode-embeddings` | Training data CLIP embeddings (65 episodes × 10 frames) |
| `robot-training-coverage` | UMAP-projected embedding space with failure labels |
| `logs-gpu.kernel-*` | CUPTI kernel traces, component decomposition |
| `metrics-gpu.robotics-*` | Control loop latency, frequency, jitter |
| `metrics-ros2.topics-*` | ROS 2 topic rates, bandwidth, node health |
| `traces-apm-*` | MCP tool call traces with latency |

## License

Apache 2.0

## Author

Joe de la Rosa — Elastic
