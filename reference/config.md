# Configuration Reference

Every configurable value in one place. Environment variables are referenced in
k8s manifests (`deploy/robot/k8s/`), rebuild scripts (`deploy/scripts/rebuild/`),
and container images (`deploy/robot/docker/`).

## Network

| Variable | Default | Description | Where used |
|----------|---------|-------------|------------|
| `GPU_SERVER` | -- | IP address of the GPU server | k8s manifests, rebuild scripts |
| `EDGE_NODE` | -- | IP address of the edge node | k8s manifests |
| `ES_URL` | `https://${GPU_SERVER}:31920` | Elasticsearch endpoint | rebuild scripts, MCP server |
| `KB_URL` | `https://${GPU_SERVER}:31561` | Kibana endpoint | rebuild scripts |
| `REGISTRY` | `${GPU_SERVER}:5050` | Container registry URL | k8s manifests, build scripts |
| `REGISTRY_PORT` | `5050` | Container registry port | registry setup |

## Elasticsearch

| Variable | Default | Description | Where used |
|----------|---------|-------------|------------|
| `ES_PASSWORD` | -- (required) | Elastic superuser password | rebuild scripts, MCP server, Refinery |
| `ES_USER` | `elastic` | Elasticsearch username | rebuild scripts, MCP server |
| `CURL_FLAGS` | `-sk` | curl flags (`-k` disables TLS verify for self-signed certs) | rebuild scripts |

Retrieve the auto-generated password:

```bash
sudo kubectl get secret elasticsearch-es-elastic-user \
  -o jsonpath='{.data.elastic}' | base64 -d
```

## Services

| Variable | Default | Description | Where used |
|----------|---------|-------------|------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint | MCP server |
| `EVALUATOR_MODEL` | `qwen3-vl:8b` | Ollama model for VLM evaluation | MCP server |
| `JINA_CLIP_URL` | `http://localhost:8900` | Jina CLIP v2 endpoint | MCP server |
| `MCP_PORT` | `8888` | FastMCP server port | MCP server, k8s manifest |
| `MCP_URL` | `http://${GPU_SERVER}:8888/mcp` | MCP endpoint for Agent Builder connector | rebuild scripts |
| `ELASTIC_APM_SERVER_URL` | `http://localhost:8200` | APM server endpoint | MCP server |
| `ELASTIC_APM_SERVICE_NAME` | `so101-mcp-server` | APM service name | MCP server |
| `ELASTIC_APM_SECRET_TOKEN` | `""` | APM auth token (if configured) | MCP server |
| `ELASTIC_APM_ENVIRONMENT` | `production` | APM environment label | MCP server |

## Hardware

| Variable | Default | Description | Where used |
|----------|---------|-------------|------------|
| `CAM0_DEVICE` | -- | Wrist camera device identifier (e.g., `usb5`) | camera-wrist pod |
| `CAM1_DEVICE` | -- | Overhead camera device identifier (e.g., `usb1`) | camera-overhead pod |
| `FOLLOWER_PORT` | `/dev/ttyACM1` | USB serial port for follower arm | motor-node pod |
| `MAX_RELATIVE_TARGET` | `none` | Motor clamping limit (`none` = unclamped) | motor-node pod |
| `GPU_NODE_NAME` | -- | k8s hostname of the GPU server node | k8s nodeSelector |
| `MOTOR_NODE_NAME` | -- | k8s hostname of the edge node | k8s nodeSelector |

Camera device identifiers and serial ports may change after USB re-enumeration
(reboot, unplug/replug). Verify before each session:

```bash
# On the GPU server
v4l2-ctl --list-devices

# On the edge node
ls /dev/ttyACM*
```

## Model

| Variable | Default | Description | Where used |
|----------|---------|-------------|------------|
| `MODEL_PATH` | `/models/pretrained_model` | Path to model checkpoint (container path) | inference container |
| `INFERENCE_FPS` | `30` | Inference loop frequency (must match training dataset FPS) | MCP server |
| `EXECUTION_STEPS_TARGET` | `600` | Steps per trial (600 steps @ 30 FPS = 20s) | MCP server |
| `DEVICE` | `cuda` | PyTorch device for inference | inference container |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Which GPUs are visible to the container | inference container |
| `TRAJECTORY_CORRECTION` | `0` | Trajectory correction offset (0 = disabled) | inference container |

## ROS 2

| Variable | Default | Description | Where used |
|----------|---------|-------------|------------|
| `ROS_DOMAIN_ID` | `0` | ROS 2 domain ID (all nodes must match) | all robot pods |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` | ROS 2 middleware implementation | camera pods |
| `FASTRTPS_DEFAULT_PROFILES_FILE` | `/ros2_ws/fastdds_no_shm.xml` | FastDDS config (disables SHM) | camera pods |

## GPU tracing (Refinery)

| Variable | Default | Description | Where used |
|----------|---------|-------------|------------|
| `CUDA_INJECTION64_PATH` | `/var/lib/refinery/shim/libgpucollector.so` | CUPTI shim library path | inference container |
| `GPUCOLLECTOR_ACTIVITY_BUFFER_SIZE` | `65536` | CUPTI activity buffer size | inference container |

## Paths

| Variable | Default | Description | Where used |
|----------|---------|-------------|------------|
| `REPO_ROOT` | -- | Repository root on the GPU server | rebuild scripts, volume mounts |
| `HF_CACHE_DIR` | `~/.cache/huggingface` | HuggingFace cache directory | motor-node pod |
| `OUTPUTS_DIR` | `~/outputs` | Training outputs directory | inference container (checkpoint symlinks) |

## Config file

The rebuild scripts use `deploy/scripts/rebuild/config.env` for credentials
and endpoints. Copy the example and edit:

```bash
cd deploy/scripts/rebuild
cp config.env.example config.env
```

Contents of `config.env.example`:

```bash
ES_URL="https://${GPU_SERVER}:31920"
KB_URL="https://${GPU_SERVER}:31561"
ES_USER="elastic"
ES_PASSWORD=""  # set me
MCP_URL="http://${GPU_SERVER}:8888/mcp"
CURL_FLAGS="-sk"
```

`config.env` is gitignored. Never commit credentials.
