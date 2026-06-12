# Deploying the Robot Stack

This page covers building container images, deploying workloads to k3s,
rebuilding Elasticsearch state, and verifying the system end to end.

## Building container images

There are four container images to build. All Dockerfiles are in `infra/docker/`
and use the repository root as the build context.

```bash
# On the GPU server, from the repository root

# MCP server (FastMCP + ROS 2 Jazzy + APM)
sudo docker build -f infra/docker/Dockerfile.mcp \
  -t ${REGISTRY}/so101-mcp-server:v1 .
sudo docker push ${REGISTRY}/so101-mcp-server:v1

# VLA inference (PyTorch + CUDA + LeRobot + ROS 2 Jazzy)
sudo docker build -f infra/docker/Dockerfile.inference \
  -t ${REGISTRY}/so101-inference:v1 .
sudo docker push ${REGISTRY}/so101-inference:v1

# Camera node (OpenCV + ROS 2 Humble + FastDDS config)
sudo docker build -f infra/docker/Dockerfile.camera \
  -t ${REGISTRY}/so101-camera:v1 .
sudo docker push ${REGISTRY}/so101-camera:v1

# Motor node (LeRobot + Feetech drivers + ROS 2 Humble)
sudo docker build -f infra/docker/Dockerfile.motor \
  -t ${REGISTRY}/so101-motor:v1 .
sudo docker push ${REGISTRY}/so101-motor:v1
```

The inference image is the largest (~12 GB) because it includes PyTorch with
CUDA and the full LeRobot dependency tree. Expect the first build to take
15-20 minutes.

### Host-mounted source files

The MCP server and inference node source files are host-mounted into the
running containers via volume mounts in the k8s manifests. This means you can
edit `src/cortex/brain/mcp_server.py` or `src/cortex/brain/inference_node.py`
on the GPU server and restart the pod to pick up changes -- no image rebuild
required. Only rebuild the image when dependencies change.

## Updating the k8s manifests

Before applying, update `infra/k8s/robot-brain.yaml` with your environment:

- **Image references:** Replace the registry URL with `${REGISTRY}`
- **Node selectors:** Set `kubernetes.io/hostname` to match your actual node
  names (run `sudo kubectl get nodes` to check)
- **Device paths:** Set `CAM0_DEVICE`, `CAM1_DEVICE`, `FOLLOWER_PORT` to match
  your hardware (see [CONFIG.md](../CONFIG.md))
- **Model path:** Set the `MODEL_PATH` env var and the corresponding
  `hostPath` volume to your trained checkpoint location
- **Host paths:** Update all `hostPath` volumes to match your repository
  location on the GPU server

See [CONFIG.md](../CONFIG.md) for the complete list of environment variables.

## Deploying workloads

```bash
# Create the namespace if you haven't already
sudo kubectl create namespace robot

# Apply all workloads
sudo kubectl apply -f infra/k8s/robot-brain.yaml
```

This single manifest deploys:

| Deployment | Node | Containers | Purpose |
|-----------|------|------------|---------|
| `robot-brain` | GPU server | `mcp-server` + `inference` | MCP tools + VLA inference |
| `motor-node` | Edge node | `motor` | Servo control over USB serial |
| `camera-wrist` | GPU server | `camera` | Wrist camera (cam0) via ROS 2 |
| `camera-overhead` | GPU server | `camera` | Overhead camera (cam1) via ROS 2 |

Verify all pods are running:

```bash
sudo kubectl get pods -n robot -o wide
```

You should see `robot-brain 2/2`, `motor-node 1/1`, and both camera pods at
`1/1`. If a pod is in `CrashLoopBackOff`, check logs:

```bash
sudo kubectl logs -n robot deployment/robot-brain -c mcp-server
sudo kubectl logs -n robot deployment/robot-brain -c inference
```

### Important: hostNetwork pods

All pods use `hostNetwork: true` for DDS multicast and direct USB access. This
means they bind directly to host ports. If you need to replace a pod, delete
the old one first -- rolling updates will fail because the new pod cannot bind
the same port while the old one is still running.

```bash
sudo kubectl delete pod -n robot -l app=robot-brain
sudo kubectl apply -f infra/k8s/robot-brain.yaml
```

## Rebuilding Elasticsearch state

The rebuild scripts in `scripts/rebuild/` set up index templates, dashboards,
the Agent Builder agent, and Workflows. Run them in order:

```bash
cd scripts/rebuild
cp config.env.example config.env
# Edit config.env: set ES_URL, KB_URL, ES_PASSWORD

./00-preflight.sh              # Verify cluster is reachable
./01-start-trial.sh            # Start trial license (new clusters only)
./02-apply-index-templates.sh  # Create robot-* index templates
./03-import-saved-objects.sh   # Import dashboards, data views, connectors
./04-create-agent-builder-agent.sh  # Create the Agent Builder agent
./05-verify.sh                 # Verify everything is green
./06-deploy-workflows.sh       # Deploy Elastic Workflows
```

Each script is idempotent and can be re-run safely. After
`03-import-saved-objects.sh`, you need to manually re-enter connector secrets
in the Kibana UI (the Ollama URL, Jina CLIP URL, and MCP connector URL) --
saved object export does not include secrets.

## Registering MCP tools in Agent Builder

The MCP server exposes 26 tools. The rebuild script `04-create-agent-builder-agent.sh`
registers all of them, but if you need to register a tool manually:

```bash
curl -sk -u "${ES_USER}:${ES_PASSWORD}" \
  "${KB_URL}/api/agent_builder/tools" \
  -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{
    "id": "robot.<tool_name>",
    "type": "mcp",
    "configuration": {
      "connector_id": "<your-mcp-connector-uuid>",
      "tool_name": "<tool_name>"
    }
  }'
```

The connector ID is a UUID assigned when you create the MCP connector in
Kibana. Find it in the Kibana UI under Stack Management > Connectors, or in
the output of `03-import-saved-objects.sh`.

## Verifying end to end

Once everything is deployed, test the full pipeline with a single tool call:

```bash
curl -sk -u "${ES_USER}:${ES_PASSWORD}" \
  "${KB_URL}/api/actions/connector/<connector-id>/_execute" \
  -X POST -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{
    "params": {
      "subAction": "callTool",
      "subActionParams": {
        "name": "get_robot_status",
        "arguments": {}
      }
    }
  }'
```

If this returns a JSON response with camera frame ages and model info, the MCP
server is running and connected. Next, try `execute_task`:

```bash
curl -sk -u "${ES_USER}:${ES_PASSWORD}" \
  "${KB_URL}/api/actions/connector/<connector-id>/_execute" \
  -X POST -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{
    "params": {
      "subAction": "callTool",
      "subActionParams": {
        "name": "execute_task",
        "arguments": {"task": "pick up the cube"}
      }
    }
  }'
```

Or, open Kibana Agent Builder, select the SO-101 agent, and type
"pick up the cube" in the chat. The agent will call `execute_task`, run
inference for ~600 steps at 30 FPS (~20 seconds), and return a trial summary.

## Next steps

With the robot stack running, set up observability:
[04-observability.md](04-observability.md).
