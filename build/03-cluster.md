# Kubernetes Cluster

This guide sets up k3s on both machines, configures GPU access for
containers, and creates a local container registry. By the end, you'll have
a working Kubernetes cluster ready to deploy the robot stack.

## Why Kubernetes?

You could run everything with `docker run` commands and shell scripts. We
use Kubernetes because:

- Pods restart automatically when they crash (cameras disconnect, USB
  re-enumerates)
- Manifests are declarative — the cluster state is in version control
- ECK (Elastic Cloud on Kubernetes) handles Elasticsearch lifecycle
- Node selectors route workloads to the right hardware
- It's the pattern that matters at enterprise scale

k3s is a lightweight Kubernetes distribution that installs in 30 seconds
and uses ~500MB of RAM. It's production-grade and CNCF-certified.

## Install k3s on the GPU server

The GPU server runs the k3s control plane:

```bash
curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
```

Verify it's running:

```bash
sudo kubectl get nodes
```

You should see one node in `Ready` status.

### Get the join token

The edge node needs a token to join the cluster:

```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

Save this token — you'll use it in the next step.

## Install k3s on the edge node

Join the edge node as an agent (worker):

```bash
curl -sfL https://get.k3s.io | \
  K3S_URL=https://192.168.88.254:6443 \
  K3S_TOKEN=<token-from-above> \
  sh -
```

Replace `192.168.88.254` with your GPU server's IP and paste the token from
the previous step.

Back on the GPU server, verify both nodes are ready:

```bash
sudo kubectl get nodes
```

You should see two nodes, both `Ready`.

**Note:** `kubectl` requires `sudo` on both nodes with k3s's default
configuration. This is normal.

## Configure GPU access

k3s needs the NVIDIA container runtime to schedule GPU workloads:

```bash
# On the GPU server
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart k3s
```

Verify GPU access from inside a container:

```bash
sudo kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:12.1.0-base-ubuntu22.04 \
  --limits=nvidia.com/gpu=1 -- nvidia-smi
```

This should show the same output as running `nvidia-smi` on the host. If it
fails with a device error, the container runtime configuration didn't take
effect — restart k3s again.

## Container registry

The robot containers are custom images that need to live in a registry k3s
can pull from. A local registry is the simplest option:

```bash
# On the GPU server
sudo docker run -d -p 5050:5000 \
  --restart=always --name registry registry:2
```

Tell k3s to trust this registry. Create `/etc/rancher/k3s/registries.yaml`
on **both** the GPU server and the edge node:

```yaml
mirrors:
  "192.168.88.254:5050":
    endpoint:
      - "http://192.168.88.254:5050"
```

Replace `192.168.88.254` with your GPU server's IP.

Restart k3s on both nodes after adding this file:

```bash
# GPU server
sudo systemctl restart k3s

# Edge node
sudo systemctl restart k3s-agent
```

Verify the registry is reachable:

```bash
curl http://192.168.88.254:5050/v2/_catalog
```

Should return `{"repositories":[]}` (empty for now).

## Label the nodes

Label each node so k8s manifests can target workloads to the correct
hardware:

```bash
# Check your node names
sudo kubectl get nodes

# The manifests use kubernetes.io/hostname selectors
# Verify node names match what you'll put in the manifests
sudo kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
```

The k8s manifests in this repo use `nodeSelector` with
`kubernetes.io/hostname` to place GPU workloads on the GPU server and motor
workloads on the edge node. You'll set these values when you deploy.

## What you should have at the end of this stage

- [ ] k3s running on GPU server (control plane)
- [ ] k3s running on edge node (agent)
- [ ] Both nodes show `Ready` in `kubectl get nodes`
- [ ] GPU accessible from containers (gpu-test pod succeeded)
- [ ] Local container registry running on port 5050
- [ ] Registry trusted by k3s on both nodes

You're done with the build stage. Everything physical and infrastructure is
in place.

Next: [deploy/01-elastic.md](../deploy/01-elastic.md) — deploying the
Elastic stack.
