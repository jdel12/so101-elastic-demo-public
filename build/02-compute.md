# Compute and Network

This guide covers setting up the GPU server, the edge node, and the network
between them. By the end, you'll have two Linux machines on a wired LAN
ready for Kubernetes.

## Architecture decision: why two machines?

The robot arm connects to a computer via USB serial. The GPU runs inference.
In theory you could put both on the same machine — but in practice:

- USB serial to the arm needs low-latency, real-time scheduling
- GPU inference saturates memory and compute on its own
- Separating concerns lets you swap either side independently
- It demonstrates the distributed pattern that matters in production

If you only have one machine with a GPU and USB ports, you can run
everything on a single node. The k8s manifests support this — just change
the node selectors. But the two-node setup is what we run and recommend.

## GPU server

This is the machine that runs model inference, the MCP server, cameras,
Elasticsearch, Kibana, and all the observability infrastructure.

### Hardware requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | 1x NVIDIA, 16GB VRAM | 2x NVIDIA, 16GB VRAM each |
| RAM | 32 GB | 64 GB |
| CPU | 8 cores | 16 cores |
| Storage | 256 GB SSD | 512 GB NVMe |
| USB | 2+ separate controllers | 3+ (for multiple cameras) |

**GPU selection:** The critical spec is 16GB VRAM. The VLA model (X-VLA,
~879M parameters) needs ~14GB for inference. Cards that work:

- RTX 4060 Ti 16GB (~$400) — what we use, two of them
- RTX 3090 24GB (~$800 used) — plenty of room
- RTX 4070 Ti Super 16GB (~$750)
- RTX 4090 24GB (~$1,600) — overkill but fast
- Any datacenter GPU with 16GB+ (A4000, A5000, L4, etc.)

Cards with 8GB VRAM will not load the model. Don't try.

A second GPU is useful but not required. We run inference on GPU 0 and
evaluation models (Ollama, Jina CLIP) on GPU 1. With a single GPU, you can
still run everything — just not inference and evaluation simultaneously.

### Operating system

Ubuntu 22.04 LTS or 24.04 LTS. Other Linux distributions work but the
commands in this guide assume Ubuntu/Debian.

### NVIDIA driver installation

```bash
sudo apt update
sudo apt install -y nvidia-driver-560
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
```

You should see your GPU(s) listed with driver version and CUDA version.

**Pro tip:** Don't install CUDA toolkit system-wide. The containers bring
their own CUDA. You only need the host driver.

### Network configuration

The GPU server needs a static IP or a DHCP reservation so the edge node
can always find it. Our setup uses 192.168.88.x but any LAN subnet works.

```bash
# Example: static IP via netplan (/etc/netplan/01-config.yaml)
network:
  version: 2
  ethernets:
    enp3s0:
      addresses: [192.168.88.254/24]
      routes:
        - to: default
          via: 192.168.88.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

```bash
sudo netplan apply
```

### SSH setup

Set up SSH key-based access so you can manage the server remotely:

```bash
# From your development machine
ssh-copy-id user@gpu-server

# Add a convenient alias to ~/.ssh/config
Host gpu-server
    HostName 192.168.88.254
    User your-username
    IdentityFile ~/.ssh/id_ed25519
```

## Edge node

The edge node sits next to the robot arm and drives the servos over USB
serial. It doesn't need a GPU.

### Hardware options

| Option | Cost | Notes |
|--------|------|-------|
| Intel NUC (any generation) | $200-400 | What we use. Compact, reliable. |
| Old laptop | Free | Works fine if it has USB and Ethernet |
| Raspberry Pi 5 (8GB) | ~$80 | Cheapest option, adequate performance |
| Mini PC (Beelink, etc.) | $150-300 | Good value |

The only real requirements are: runs Linux, has USB ports for the arm, has
Gigabit Ethernet.

### OS and setup

Ubuntu 22.04 or 24.04 on the edge node as well. Same SSH key setup:

```bash
ssh-copy-id user@edge-node

Host edge-node
    HostName 192.168.88.231
    User your-username
    IdentityFile ~/.ssh/id_ed25519
```

## Network

Both machines must be on the same LAN segment with Gigabit Ethernet. This
is non-negotiable — ROS 2 DDS uses UDP multicast for discovery and data
transport.

**Do not use Wi-Fi.** DDS multicast over Wi-Fi is unreliable. Motor commands
arrive late or out of order, and the arm stutters. We tried it; it doesn't
work.

### Verify connectivity

```bash
# From GPU server
ping edge-node -c 5

# From edge node
ping gpu-server -c 5

# Check for gigabit link
ethtool enp3s0 | grep Speed
# Should show: Speed: 1000Mb/s
```

### Firewall

k3s and ROS 2 need several ports open between the nodes. The simplest
approach is to disable the firewall between them (they're on a private LAN):

```bash
sudo ufw allow from 192.168.88.0/24
```

Or if you prefer explicit rules, k3s needs 6443 (API), 10250 (kubelet),
and ROS 2 DDS uses ephemeral UDP ports (typically 7400-7500+ and 32000+).

## Ollama and Jina CLIP (GPU server)

These services run directly on the GPU server (not in k8s) and provide
evaluation capabilities.

### Ollama

Ollama runs Qwen3-VL for automated trial evaluation — after each trial,
it looks at the camera frames and judges whether the robot succeeded.

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the evaluation model
ollama pull qwen3-vl:8b

# Verify it's running
curl http://localhost:11434/api/tags
```

Ollama defaults to port 11434 and uses GPU 1 if available.

### Jina CLIP v2

Jina CLIP provides scene embeddings for novelty detection — comparing the
current camera view against the training data distribution.

```bash
# Run as a Docker container on GPU 1
sudo docker run -d --gpus '"device=1"' \
  -p 8900:8080 \
  --name jina-clip \
  --restart=always \
  jinaai/jina-clip-v2
```

If you have only one GPU, omit the `--gpus` flag and it will run on CPU
(slower but functional).

## What you should have at the end of this stage

- [ ] GPU server running Ubuntu with NVIDIA drivers installed
- [ ] Edge node running Ubuntu, connected via Gigabit Ethernet
- [ ] SSH key access configured from your development machine to both
- [ ] Both machines can ping each other
- [ ] Gigabit link verified
- [ ] Ollama running with Qwen3-VL pulled
- [ ] Jina CLIP running (optional for initial setup)

Next: [03-cluster.md](03-cluster.md) — installing k3s and the container
registry.
