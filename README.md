# Operating AI Robots with Elastic

A distributed AI robotics system where a natural language command in Kibana
Agent Builder drives a physical SO-101 robot arm through MCP tools, ROS 2,
and vision-language-action models on Kubernetes. Everything is observable
through Elasticsearch and Kibana.

**Companion to the Search Labs article series** — the articles cover
what we learned; this repository has everything you need to build it yourself.

> [Part 1: Operating AI robots with Elastic](#) (coming soon)
> [Part 2: What nvidia-smi won't tell you](#) (coming soon)

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
GPU Server (Ubuntu, k3s control, 1-2× NVIDIA GPUs with 16GB+ VRAM)
├── robot-brain pod (hostNetwork)
│   ├── mcp-server    (FastMCP, 26 tools, host-mounted source)
│   └── inference     (VLA model on GPU 0, 30 FPS)
├── camera-wrist pod  (USB camera, separate controller)
├── camera-overhead   (USB camera, separate controller)
├── Ollama            (Qwen3-VL on GPU 1, auto-evaluation)
├── Jina CLIP v2      (GPU 1, 1024-dim embeddings)
├── Elasticsearch 9.4.1  (ECK 3.4.0)
├── Kibana 9.4.1
└── Refinery          (Rust GPU obs daemon, CUPTI kernel tracing)

        ↕ DDS (ROS 2 Humble ↔ Jazzy) over gigabit LAN

Edge Node (Ubuntu, k3s agent)
├── motor-node pod    (SO-101 follower arm, Feetech servos)
└── SO-101 arms       (leader + follower)
```

Both cameras are on the GPU server on separate USB controllers. Camera
containers use ROS 2 Humble; inference uses Jazzy. FastDDS SHM is disabled
to force UDP transport between container versions.

## Repository contents

```
docs/
├── guide/               # Step-by-step build guide (start here)
│   ├── 01-hardware.md
│   ├── 02-cluster-setup.md
│   ├── 03-deployment.md
│   ├── 04-observability.md
│   └── 05-operations.md
├── CONFIG.md            # Environment variable reference
└── gotchas.md           # 35+ categorized operational gotchas

src/
├── server/brain/        # MCP server (26 tools) + inference node
├── edge/camera/         # Camera node with USB auto-recovery
└── edge/motor/          # Motor control bridge

infra/
├── k8s/                 # Kubernetes manifests (parameterized)
├── docker/              # Container images
└── elastic/             # ES/Kibana manifests + index templates

scripts/
├── rebuild/             # ES cluster rebuild (00-preflight through 05-verify)
├── build-images.sh      # Build and push container images
├── reset-arm.sh         # Smooth arm reset to training position
└── deploy.sh            # Apply k8s manifests
```

## Key findings

These are the most interesting things we discovered building and operating
this system.

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
| GPU server | 2× RTX 4060 Ti 16GB, 64GB RAM | 1× NVIDIA GPU with 16GB VRAM |
| Edge node | Intel NUC, 16GB RAM | Any Linux box with USB serial |
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

Follow the [build guide](docs/guide/01-hardware.md) for detailed instructions.
The short version:

1. Set up k3s on your GPU server and edge node
2. Deploy Elasticsearch and Kibana via ECK
3. Copy `scripts/rebuild/config.env.example` to `config.env` and fill in your values
4. Run the rebuild scripts (`00-preflight.sh` through `05-verify.sh`)
5. Build and push container images with `scripts/build-images.sh`
6. Apply the Kubernetes manifests with `scripts/deploy.sh`
7. Register MCP tools in Kibana Agent Builder
8. Start talking to your robot

See [docs/CONFIG.md](docs/CONFIG.md) for the full environment variable reference
and [docs/gotchas.md](docs/gotchas.md) for operational gotchas.

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
