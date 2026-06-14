# Observability

This guide sets up the full observability stack: APM tracing on the MCP
server, system metrics via Elastic Agent, GPU metrics via DCGM, ROS 2
monitoring, GPU kernel tracing via Refinery, and Kibana dashboards.

This is the layer that makes this project different from a standard robotics
demo. Without it you have a robot that works or doesn't. With it, you know
exactly why.

## Elastic APM

The MCP server is already instrumented with the Elastic APM Python agent.
Every tool call, Elasticsearch query, and HTTP request to Ollama or Jina
CLIP is traced automatically. You just need an APM server to receive the
data.

The simplest path is to enable APM through Fleet:

1. In Kibana, go to **Fleet > Integrations > Elastic APM**
2. Add the APM integration to your Fleet agent policy
3. This deploys an APM server that listens on port 8200

The MCP server's APM environment variables are already set in the k8s
manifest:

```yaml
- name: ELASTIC_APM_SERVICE_NAME
  value: "so101-mcp-server"
- name: ELASTIC_APM_SERVER_URL
  value: "http://localhost:8200"
- name: ELASTIC_APM_ENVIRONMENT
  value: "production"
```

With `hostNetwork: true`, `localhost:8200` works when APM runs on the same
machine.

**What APM gives you:**

- Transaction traces for every MCP tool call with full latency breakdown
- Dependency map: MCP server -> Elasticsearch, Ollama, Jina CLIP
- Error rates and exception details per tool
- P50/P95/P99 latency for each of the 37 tools

## Elastic Agent DaemonSet

Elastic Agent collects system metrics (CPU, memory, disk, network) and logs
from every node. Deploy as a DaemonSet so both the GPU server and edge node
are covered.

1. In Kibana, go to **Fleet > Agent policies** and create a policy
2. Add the **System** integration (CPU, memory, disk, network, processes)
3. Go to **Fleet > Agents > Add agent** to get the enrollment command
4. Deploy as a k8s DaemonSet or install directly on each host

The key insight: system metrics establish the baseline. When inference
throughput drops, is it because the GPU is thermal throttling? Is the edge
node out of memory? Is network bandwidth saturated? System metrics answer
these questions without guessing.

## DCGM Exporter

NVIDIA DCGM (Data Center GPU Manager) Exporter provides GPU-specific
metrics as Prometheus-format data that Elastic Agent can scrape.

Deploy the DCGM exporter:

```bash
sudo kubectl apply -f deploy/observability/dcgm-exporter.yaml
```

This runs as a DaemonSet on GPU nodes and exposes metrics on port 9400.
Configure Elastic Agent's Prometheus integration to scrape
`http://<gpu-server>:9400/metrics`.

| Metric | What it tells you |
|--------|-------------------|
| `DCGM_FI_DEV_GPU_UTIL` | GPU compute utilization (%) |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory bandwidth utilization (%) |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature (C) |
| `DCGM_FI_DEV_POWER_USAGE` | Power draw (W) |
| `DCGM_FI_DEV_FB_USED` | Framebuffer memory used (MB) |

DCGM tells you *that* the GPU is busy. Refinery (below) tells you *what*
it's busy doing.

## ROS 2 monitoring

The ROS 2 observability stack monitors topic publication rates, bandwidth,
and node health across the cluster.

**Collector:** A C++ ROS 2 node that subscribes to DDS topics and measures
rates, bandwidth, and latency. Runs as a pod on each node with ROS 2
workloads.

**Shipper:** A Rust binary that reads the collector's output and ships it to
Elasticsearch as `metrics-ros2.topics-*` data streams.

```bash
sudo kubectl apply -f deploy/observability/ros2-collector.yaml
```

Three MCP tools expose ROS 2 health to Agent Builder:

- `get_ros2_topic_health` — publication rates and bandwidth per topic
- `get_ros2_node_status` — which ROS 2 nodes are alive
- `get_ros2_alerts` — nodes or topics that have gone silent

When the camera stops publishing or the motor node goes unresponsive, these
tools surface the problem immediately — in the dashboard and through the
AI agent.

## Refinery (GPU kernel tracing)

Refinery is a Rust daemon that intercepts CUPTI (CUDA Profiling Tools
Interface) callbacks to trace individual GPU kernel executions. This is
the layer that reveals what `nvidia-smi` hides.

Instead of "85% utilization," you see:

- Vision encoder: 78.6% of compute time (ResNet backbone processing camera frames)
- Attention layers: 11.4% (transformer cross-attention between vision and language)
- Action head: 2.3% (MLP that outputs joint commands)
- Memory operations: 7.7% (data movement between CPU and GPU)

### Install Refinery

Refinery installs via Helm:

```bash
helm repo add refinery https://refinery.dev/charts
helm install refinery refinery/refinery \
  --set elasticsearch.url=https://<gpu-server-ip>:31920 \
  --set elasticsearch.username=elastic \
  --set elasticsearch.password=<password>
```

### CUPTI shim

Refinery uses a shared library that intercepts CUDA calls via the
`CUDA_INJECTION64_PATH` environment variable. The inference container
mounts this shim:

```yaml
- name: CUDA_INJECTION64_PATH
  value: "/var/lib/refinery/shim/libgpucollector.so"
```

The shim path must match the actual file on disk. If Refinery is
reinstalled, restart the inference container to pick up the new shim.

Measured overhead: approximately 0.5% on inference throughput. Effectively
invisible.

### What Refinery produces

| Index | Content |
|-------|---------|
| `logs-gpu.kernel-*` | Every GPU kernel execution: name, duration, grid/block size, component classification |
| `metrics-gpu.robotics-*` | Control loop frequency, jitter, duty cycle |

Seven MCP tools query this data:

- `get_gpu_kernel_profile` — kernel breakdown by component
- `get_inference_breakdown` — time allocation across model stages
- `get_gpu_memory_status` — memory utilization trends
- `get_episode_gpu_quality` — per-trial GPU execution quality
- `get_control_loop_status` — real-time loop frequency and jitter
- `get_safety_events` — anomalous kernel durations or memory spikes
- `profile_inference` — duty cycle summary over a time window

## Dashboards

Import the pre-built dashboards using the rebuild scripts:

```bash
cd deploy/scripts/rebuild
./03-import-saved-objects.sh
```

This imports dashboards for:

- **Robot Operations Overview** — trial results, success rates, timing
- **GPU Kernel Decomposition** — component-level GPU time allocation
- **CLIP Confidence Scoring** — scene novelty vs. training distribution
- **Model Comparison** — side-by-side model metrics
- **ROS 2 Topic Health** — publication rates, bandwidth, node status
- **System Metrics** — CPU, memory, network per node
- **APM Service Map** — MCP tool call traces with latency

After importing, you'll need to re-enter connector secrets in the Kibana UI
(Ollama URL, Jina CLIP URL, MCP connector URL) — saved object exports do
not include secrets.

Dashboards will show empty panels until data flows. Run your first trial
and they'll populate.

## What you should have at the end

- [ ] APM tracing active on MCP server (check APM UI in Kibana)
- [ ] Elastic Agent running on both nodes (check Fleet > Agents)
- [ ] DCGM Exporter serving GPU metrics on port 9400
- [ ] ROS 2 collector running and shipping topic metrics
- [ ] Refinery installed and CUPTI shim active
- [ ] Dashboards imported in Kibana

Next: [04-agent-builder.md](04-agent-builder.md) — wiring up Agent Builder
to talk to your robot.
