# Operating AI Robots with Elastic

A reference architecture for running vision-language-action models on physical
robots — with full observability through Elasticsearch and Kibana.

Natural language in, robot motion out, and Elastic tells you exactly what
happened: per-step model decisions, GPU kernel traces, joint trajectories,
scene confidence scoring, and automated failure diagnosis that identifies
what to fix next.

## Why this exists

Training a robot model and getting it to move is the easy part. Knowing
*why* it fails — and exactly what to change — is the hard part. This
project answers that question systematically.

A VLA model runs on a consumer GPU. It picks up a cube sometimes and misses
other times. Standard tooling (`nvidia-smi`, log files, watching the arm)
tells you almost nothing about why. This system captures every signal that
matters — model decisions, GPU execution, visual context, physical motion —
indexes it in Elasticsearch, and makes it queryable through Kibana dashboards
and an AI agent that can diagnose failures on its own.

**What we found building this:**

- The model replays a memorized trajectory — GPU kernel profiles are
  identical between success and failure. Planning failure, not perception.
- CLIP distance to training data predicts failure before execution starts.
  Success <= 0.038, failure >= 0.040.
- One "85% GPU utilization" number becomes thousands of classified kernels:
  vision encoder 78.6%, attention 11.4%, action head 2.3%.
- Fusing VLM evaluation + kernel traces + CLIP distance + trajectory data
  diagnoses failures with evidence, not guessing.
- Consumer GPUs (RTX 4060 Ti 16GB) run production inference at 30 FPS.
  CUPTI profiling overhead is 0.5%.

## Architecture

```
GPU Server (Ubuntu, k3s, 1-2x NVIDIA GPUs 16GB+ VRAM)
├── robot-brain pod (hostNetwork)
│   ├── mcp-server    (FastMCP, 37 tools, host-mounted source)
│   └── inference     (VLA model on GPU 0, 30 FPS)
├── camera pods       (USB cameras, separate controllers)
├── Ollama            (Qwen3-VL, auto-evaluation)
├── Jina CLIP v2      (1024-dim scene embeddings)
├── Elasticsearch     (ECK, system of record)
├── Kibana            (dashboards, Agent Builder)
├── Refinery          (Rust GPU obs daemon, CUPTI kernel tracing)
├── Fleet Server      (Elastic Agent management)
└── DCGM Exporter     (GPU metrics)

        ↕ DDS (ROS 2) over gigabit Ethernet

Edge Node (Ubuntu, k3s agent)
├── motor-node pod    (SO-101 follower arm, Feetech servos)
└── SO-101 arms       (leader + follower)
```

The GPU server runs the brain: MCP tools, model inference, evaluation,
and the full Elastic stack. The edge node drives the physical arm over USB
serial. ROS 2 DDS carries motor commands and camera frames between them.
Both cameras are on the GPU server on separate USB controllers.

## What you can do with this

**As a robotics engineer:** Deploy the full stack and get observability into
your VLA model that doesn't exist anywhere else — per-step decision
telemetry, GPU kernel decomposition, training data coverage analysis, and
automated failure diagnosis.

**As a platform engineer:** See how Elasticsearch, Kibana Agent Builder, APM,
Fleet, and MCP tools work together to operate a real-time AI system — a
pattern that applies to any AI workload, not just robots.

**As an AI/ML engineer:** Understand why your model fails at specific
positions, how to profile GPU execution at the kernel level, and how to
build a training feedback loop that systematically improves performance.

## How this repo is organized

The repository is structured as three stages: **Build**, **Deploy**, and
**Operate**. Each stage has its own directory with guides, scripts, configs,
and everything you need for that stage.

```
build/                     Stage 1: Hardware and infrastructure
├── 01-robot.md              Sourcing parts, assembly, calibration
├── 02-compute.md            GPU server, edge node, networking
└── 03-cluster.md            k3s, container registry, GPU runtime

deploy/                    Stage 2: Software stack
├── 01-elastic.md            Elasticsearch, Kibana, trial license
├── 02-robot-stack.md        Brain, cameras, motor — all containers
├── 03-observability.md      APM, Fleet, DCGM, ROS 2 monitoring, dashboards
├── 04-agent-builder.md      MCP connector, tool registration, first trial
├── infrastructure/          ECK manifests, index templates
├── robot/                   Dockerfiles, k8s manifests
├── observability/           Fleet, DCGM, dashboard exports
└── scripts/                 Build, deploy, rebuild scripts

operate/                   Stage 3: Running the system
├── 01-trials.md             Running trials, Agent Builder, interpreting results
├── 02-training.md           Recording episodes, training models, hot-swap
├── 03-diagnostics.md        Model Decision Observatory, failure diagnosis
├── 04-gpu-profiling.md      Kernel-level GPU analysis with Refinery
└── scripts/                 Operational scripts (reset, diagnose, record)

src/                       Runtime source code
├── brain/                   MCP server (37 tools), inference engine,
│                            telemetry, forward kinematics
├── camera/                  Camera node with USB auto-recovery
└── motor/                   Motor control bridge

reference/                 Look-up material
├── config.md                All environment variables
├── gotchas.md               Operational gotchas by category
├── mcp-tools.md             Complete MCP tool catalog
└── architecture.md          Component deep-dive, data flow, design decisions
```

## Hardware

| Component | What we used | Minimum viable |
|-----------|-------------|----------------|
| GPU server | 2x RTX 4060 Ti 16GB, 64GB RAM | 1x NVIDIA GPU, 16GB VRAM, 32GB RAM |
| Edge node | Intel NUC, 16GB RAM | Any Linux box with USB |
| Robot arm | SO-101 leader + follower | SO-101 follower only |
| Cameras | 3x USB webcams (1080p) | 1 camera |
| Network | Gigabit Ethernet | Required for DDS |

Total hardware cost is roughly $1,500-2,500 depending on GPU choice and
whether you already have a spare Linux machine.

## Software stack

| Component | Purpose |
|-----------|---------|
| Elasticsearch + Kibana | System of record, dashboards, Agent Builder |
| ECK | Kubernetes operator for Elastic |
| k3s | Lightweight Kubernetes |
| ROS 2 (Humble + Jazzy) | Robot messaging (DDS) |
| FastMCP | MCP tool server |
| Refinery | GPU kernel tracing (CUPTI) |
| LeRobot | Training data, model training |
| Ollama + Qwen3-VL | Automated trial evaluation |
| Jina CLIP v2 | Scene embeddings for novelty detection |

## Getting started

Start with [build/01-robot.md](build/01-robot.md) and work through each
stage in order. The build stage gets your hardware ready. The deploy stage
brings up the full software stack. The operate stage is where the value
lives — running trials, training models, and diagnosing failures.

If you already have hardware and a Kubernetes cluster, skip to
[deploy/01-elastic.md](deploy/01-elastic.md).

## Elasticsearch indices

| Index pattern | What it stores |
|--------------|----------------|
| `robot-inference` | Trial results — success/failure, confidence, timing |
| `robot-steps` | Per-step model decisions — joint commands, telemetry, z-scores |
| `robot-episodes` | Training episode metadata |
| `robot-scene-embeddings` | CLIP embeddings of training data (1024-dim) |
| `robot-training-coverage` | Embedding space with failure labels |
| `robot-evaluations` | VLM evaluation verdicts |
| `robot-models` | Model checkpoint metadata |
| `logs-gpu.kernel-*` | GPU kernel traces with component classification |
| `metrics-gpu.robotics-*` | Control loop latency, frequency, jitter |
| `metrics-ros2.topics-*` | ROS 2 topic rates, bandwidth, node health |
| `traces-apm-*` | MCP tool call traces with latency |

## Articles

This repository is the companion to a Search Labs article series:

> [Part 1: Operating AI robots with Elastic](#) (coming soon)
> [Part 2: What nvidia-smi won't tell you](#) (coming soon)

The articles cover what we learned. This repository has everything you need
to build it yourself.

## License

Apache 2.0
