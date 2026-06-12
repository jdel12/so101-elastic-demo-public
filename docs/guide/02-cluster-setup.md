# Cluster Setup

This page covers getting k3s, Elasticsearch, Kibana, and supporting services
running on your GPU server and edge node.

## Prerequisites

- Two Linux machines on the same LAN (see [01-hardware.md](01-hardware.md))
- SSH access to both machines
- NVIDIA drivers installed on the GPU server (`nvidia-smi` should work)

All commands below use environment variables defined in
[CONFIG.md](../CONFIG.md). Set those first.

## k3s installation

k3s is a lightweight Kubernetes distribution. Install on the GPU server
(control plane) first, then join the edge node as an agent.

```bash
# GPU server (control plane)
curl -sfL https://get.k3s.io | sh -s - \
  --write-kubeconfig-mode 644

# Get the join token
sudo cat /var/lib/rancher/k3s/server/node-token

# Edge node (agent) — replace TOKEN and GPU_SERVER
curl -sfL https://get.k3s.io | K3S_URL=https://${GPU_SERVER}:6443 \
  K3S_TOKEN=<token-from-above> sh -
```

Verify both nodes are ready:

```bash
sudo kubectl get nodes
```

The k3s docs cover this in detail:
[docs.k3s.io/quick-start](https://docs.k3s.io/quick-start).

### NVIDIA GPU support

k3s needs the NVIDIA container runtime to schedule GPU workloads:

```bash
# On the GPU server
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart k3s
```

Verify GPU access from a pod:

```bash
sudo kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:12.1.0-base-ubuntu22.04 \
  --limits=nvidia.com/gpu=1 -- nvidia-smi
```

## Container registry

The k8s manifests pull images from a local registry. The simplest option is
k3s's built-in registry mirror or a local Docker registry:

```bash
# On the GPU server
sudo docker run -d -p ${REGISTRY_PORT:-5050}:5000 \
  --restart=always --name registry registry:2
```

Configure k3s to trust this registry. Create
`/etc/rancher/k3s/registries.yaml` on both nodes:

```yaml
mirrors:
  "${GPU_SERVER}:${REGISTRY_PORT:-5050}":
    endpoint:
      - "http://${GPU_SERVER}:${REGISTRY_PORT:-5050}"
```

Restart k3s after adding this file.

## ECK operator

Elastic Cloud on Kubernetes (ECK) manages the Elasticsearch and Kibana
lifecycle. Install the operator:

```bash
sudo kubectl create -f https://download.elastic.co/downloads/eck/3.0.0/crds.yaml
sudo kubectl apply -f https://download.elastic.co/downloads/eck/3.0.0/operator.yaml
```

Wait for the operator pod to be ready:

```bash
sudo kubectl get pods -n elastic-system
```

The ECK docs cover advanced configuration:
[elastic.co/guide/en/cloud-on-k8s](https://www.elastic.co/guide/en/cloud-on-k8s/current/index.html).

## Elasticsearch

Deploy a single-node Elasticsearch cluster. The manifest in
`infra/elastic/elasticsearch.yaml` uses ECK custom resources:

```bash
sudo kubectl apply -f infra/elastic/elasticsearch.yaml
```

Wait for the cluster to go green:

```bash
sudo kubectl get elasticsearch
```

Retrieve the auto-generated password:

```bash
sudo kubectl get secret elasticsearch-es-elastic-user \
  -o jsonpath='{.data.elastic}' | base64 -d
```

Save this password in `scripts/rebuild/config.env` (see
[CONFIG.md](../CONFIG.md)).

### Expose via NodePort

The default ECK service is ClusterIP. To access Elasticsearch and Kibana
from outside the cluster (your development machine, the MCP server), expose
them via NodePort. The manifests in this repository use port 31920 for
Elasticsearch and port 31561 for Kibana.

## Kibana

Deploy Kibana:

```bash
sudo kubectl apply -f infra/elastic/kibana.yaml
```

Verify it is reachable (note: ECK uses HTTPS by default):

```bash
curl -sk https://${GPU_SERVER}:31561/api/status | python3 -m json.tool
```

### Critical setting: response timeout

Agent Builder tool calls can take 30+ seconds (inference, evaluation, GPU
profiling). The default Kibana timeout will cut them off. Add this to your
Kibana configuration:

```yaml
xpack.actions.responseTimeout: "120s"
```

In ECK, this goes in the Kibana custom resource under `spec.config`.

## Trial license

Several features (Agent Builder, Workflows, Fleet) require at minimum a trial
license. Start a 30-day trial:

```bash
curl -sk -u "${ES_USER}:${ES_PASSWORD}" \
  -X POST "${ES_URL}/_license/start_trial?acknowledge=true"
```

Or use the rebuild script:

```bash
cd scripts/rebuild && ./01-start-trial.sh
```

The trial is tied to the cluster UUID. If you wipe and recreate the cluster,
you get a fresh 30-day window.

## Fleet Server

Fleet Server manages Elastic Agents across both nodes. Deploy it:

```bash
sudo kubectl apply -f infra/elastic/fleet.yaml
```

After Fleet Server is running, enroll Elastic Agents on both nodes. The Fleet
UI in Kibana (Management > Fleet) provides enrollment tokens and installation
commands. Install the agent as a DaemonSet so every node reports system metrics
and log collection automatically.

## Create the robot namespace

The robot workloads run in a dedicated namespace:

```bash
sudo kubectl create namespace robot
```

The ES password secret needs to be available in this namespace for pods that
talk to Elasticsearch directly:

```bash
sudo kubectl get secret elasticsearch-es-elastic-user -o yaml \
  | sed 's/namespace: default/namespace: robot/' \
  | sudo kubectl apply -f -
```

## Next steps

With the cluster running, proceed to [03-deployment.md](03-deployment.md) to
build container images and deploy the robot stack.
