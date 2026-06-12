# Observability

This page covers setting up the full observability stack: APM tracing on the
MCP server, system metrics via Elastic Agent, GPU metrics via DCGM, ROS 2 fleet
monitoring, and kernel-level GPU tracing via Refinery.

## Elastic APM on the MCP server

The MCP server is already instrumented with the Elastic APM Python agent. It
traces every tool call, Elasticsearch query, and external HTTP request
automatically. You just need to point it at an APM server.

The following environment variables are set in the `mcp-server` container in
`infra/k8s/robot-brain.yaml`:

```yaml
- name: ELASTIC_APM_SERVICE_NAME
  value: "so101-mcp-server"
- name: ELASTIC_APM_SERVER_URL
  value: "http://localhost:8200"
- name: ELASTIC_APM_SECRET_TOKEN
  value: ""
- name: ELASTIC_APM_VERIFY_SERVER_CERT
  value: "false"
- name: ELASTIC_APM_ENVIRONMENT
  value: "production"
```

If your APM server is on a different host or uses a secret token, update these
values. With `hostNetwork: true`, `localhost:8200` works when the APM server
runs on the same machine.

Deploy an APM server via Fleet (recommended) or as a standalone binary. The
Fleet integration is the simplest path -- enable it in Kibana under
Fleet > Integrations > Elastic APM.

Once APM data flows, you get:

- Per-tool-call transaction traces in the APM UI
- Latency distribution across all 26 tools
- Downstream dependency maps (Elasticsearch, Ollama, Jina CLIP)
- Error rates and exception details

## Elastic Agent DaemonSet

Elastic Agent collects system metrics (CPU, memory, disk, network) and logs
from every node. Deploy it as a DaemonSet so both the GPU server and edge node
are covered.

1. In Kibana, go to Fleet > Agent policies and create a policy
2. Add the System integration (collects system metrics and logs)
3. Get the enrollment command from Fleet > Agents > Add agent
4. Deploy as a k8s DaemonSet using the provided manifest, or install directly
   on each host

The DaemonSet manifest in `infra/k8s/` provides a template. Update the Fleet
URL and enrollment token for your environment.

## DCGM Exporter for GPU metrics

NVIDIA DCGM (Data Center GPU Manager) Exporter provides GPU utilization,
temperature, memory usage, and power draw as Prometheus metrics. Elastic Agent
scrapes these.

Deploy the DCGM exporter:

```bash
sudo kubectl apply -f infra/k8s/dcgm-exporter.yaml
```

This runs as a DaemonSet on GPU nodes and exposes metrics on port 9400.
Configure the Elastic Agent's Prometheus integration to scrape
`http://${GPU_SERVER}:9400/metrics`.

Key metrics:

| Metric | What it tells you |
|--------|-------------------|
| `DCGM_FI_DEV_GPU_UTIL` | GPU compute utilization (%) |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory bandwidth utilization (%) |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature (C) |
| `DCGM_FI_DEV_POWER_USAGE` | Power draw (W) |
| `DCGM_FI_DEV_FB_USED` | Framebuffer memory used (MB) |

## ROS 2 fleet monitoring

The ROS 2 observability stack consists of two components:

1. **Collector** -- a C++ ROS 2 node that subscribes to DDS topics and measures
   publication rates, bandwidth, latency, and node liveness. Runs as a pod on
   each ROS 2 node.

2. **Shipper** -- a Rust binary that reads the collector's output and ships it
   to Elasticsearch as `metrics-ros2.topics-*` data streams.

Deploy the collector:

```bash
sudo kubectl apply -f infra/k8s/ros2-collector.yaml
```

This creates a DaemonSet that runs on every node with ROS 2 workloads. The
shipper can run as a sidecar or as a Fleet-managed integration.

Three MCP tools expose ROS 2 health to the Agent Builder:

- `get_ros2_topic_health` -- publication rates and bandwidth per topic
- `get_ros2_node_status` -- which ROS 2 nodes are alive
- `get_ros2_alerts` -- nodes or topics that have gone silent

## Refinery for kernel-level GPU tracing

Refinery is a Rust daemon that intercepts CUPTI (CUDA Profiling Tools Interface)
callbacks to trace individual GPU kernel executions. It classifies kernels by
component (vision encoder, attention, action head, memory operations) and ships
traces to Elasticsearch.

### Installation

Refinery installs via Helm:

```bash
helm repo add refinery https://refinery.dev/charts
helm install refinery refinery/refinery \
  --set elasticsearch.url=${ES_URL} \
  --set elasticsearch.username=${ES_USER} \
  --set elasticsearch.password=${ES_PASSWORD}
```

### CUPTI shim

Refinery uses a shared library (`libgpucollector.so`) that intercepts CUDA
calls via the `CUDA_INJECTION64_PATH` environment variable. The inference
container mounts this shim:

```yaml
- name: CUDA_INJECTION64_PATH
  value: "/var/lib/refinery/shim/libgpucollector.so"
```

The shim path must match the actual inode on disk. If Refinery is reinstalled,
the shim binary changes and the inference container must be restarted to pick
up the new inode.

Measured overhead is approximately 0.5% on inference throughput.

### What you get

With Refinery active, Elasticsearch receives:

- `logs-gpu.kernel-*` -- every GPU kernel execution with duration, grid size,
  block size, and component classification
- `metrics-gpu.robotics-*` -- control loop frequency, jitter, and duty cycle

Seven MCP tools query this data for the Agent Builder:

- `get_gpu_kernel_profile` -- kernel breakdown by component
- `get_inference_breakdown` -- time allocation across model stages
- `get_gpu_memory_status` -- memory utilization trends
- `get_episode_gpu_quality` -- per-trial GPU execution quality
- `get_control_loop_status` -- real-time loop frequency and jitter
- `get_safety_events` -- anomalous kernel durations or memory spikes
- `profile_inference` -- duty cycle summary over a time window

## Dashboard import

The rebuild script `03-import-saved-objects.sh` imports all dashboards from the
saved objects export in `infra/elastic/`. After running it, you should have
dashboards for:

- Robot trial results and success rates
- GPU kernel decomposition
- CLIP confidence scoring
- ROS 2 topic health
- System metrics (CPU, memory, network per node)
- APM service map and transaction traces

If dashboards reference data views that do not exist yet (because no data has
flowed), they will show empty panels until the first trial runs.

## Next steps

With observability configured, learn how to operate the system day to day:
[05-operations.md](05-operations.md).
