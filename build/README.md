# Build

Everything you need to go from parts on a table to a two-node Kubernetes
cluster with a robot arm, cameras, and GPU access.

| Guide | What you'll do |
|-------|---------------|
| [01-robot.md](01-robot.md) | Source parts, assemble the SO-101 arms, mount cameras, calibrate |
| [02-compute.md](02-compute.md) | Set up GPU server, edge node, networking, Ollama, Jina CLIP |
| [03-cluster.md](03-cluster.md) | Install k3s, configure GPU runtime, set up container registry |

**Time estimate:** 1-2 days, depending on whether you order a kit or source
parts individually. The cluster setup takes about an hour once the machines
are ready.

**If you already have hardware and Kubernetes:** Skip to
[deploy/01-elastic.md](../deploy/01-elastic.md).
