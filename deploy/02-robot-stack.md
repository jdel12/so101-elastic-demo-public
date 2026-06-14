# Deploying the Robot Stack

This guide builds the container images and deploys the robot brain, cameras,
and motor node to Kubernetes. By the end, you'll have a running robot that
can receive inference commands.

## Container images

There are four container images to build. All Dockerfiles are in
`deploy/robot/docker/` and use the repository root as the build context.

```bash
# On the GPU server, from the repository root

# MCP server (FastMCP + ROS 2 Jazzy + APM agent)
sudo docker build -f deploy/robot/docker/Dockerfile.mcp -t 192.168.88.254:5050/so101-mcp-server:v1 .
sudo docker push 192.168.88.254:5050/so101-mcp-server:v1

# VLA inference (PyTorch + CUDA + LeRobot + ROS 2 Jazzy)
sudo docker build -f deploy/robot/docker/Dockerfile.inference -t 192.168.88.254:5050/so101-inference:v1 .
sudo docker push 192.168.88.254:5050/so101-inference:v1

# Camera node (OpenCV + ROS 2 Humble + FastDDS config)
sudo docker build -f deploy/robot/docker/Dockerfile.camera -t 192.168.88.254:5050/so101-camera:v1 .
sudo docker push 192.168.88.254:5050/so101-camera:v1

# Motor node (LeRobot + Feetech drivers + ROS 2 Humble)
sudo docker build -f deploy/robot/docker/Dockerfile.motor -t 192.168.88.254:5050/so101-motor:v1 .
sudo docker push 192.168.88.254:5050/so101-motor:v1
```

Replace `192.168.88.254:5050` with your registry address.

The inference image is the largest (~12GB) because it includes PyTorch with
CUDA and the full LeRobot dependency tree. The first build takes 15-20
minutes. Subsequent builds use Docker layer caching.

## Model checkpoint

Before deploying the inference container, you need a trained VLA model
checkpoint on the GPU server. If you trained with LeRobot, your checkpoint
is in `~/outputs/train/<run-name>/checkpoints/last/pretrained_model/`.

Create a symlink at the path the manifest expects:

```bash
mkdir -p ~/models
ln -s ~/outputs/train/<run-name>/checkpoints/last/pretrained_model ~/models/pretrained_model
```

If you don't have a trained model yet, you can deploy the stack and add
the model later — the inference container will fail to start until a valid
checkpoint is present, but the MCP server and cameras will still run.

## Update the k8s manifest

The manifest at `deploy/robot/k8s/robot-brain.yaml` needs your
environment-specific values. Update these before applying:

**Image references:** Replace the registry URL with your registry address.

**Node selectors:** Set `kubernetes.io/hostname` to match your node names.
Check with:

```bash
sudo kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
```

**Camera device paths:** Set `CAM0_DEVICE` and `CAM1_DEVICE` to your camera
identifiers. On the GPU server:

```bash
v4l2-ctl --list-devices
# Look for the USB bus number (e.g., usb5, usb1)
# Or use the device path directly (e.g., /dev/video0)
```

**Motor serial port:** Set `FOLLOWER_PORT` to the arm's serial port on the
edge node:

```bash
# On the edge node
ls /dev/ttyACM*
```

**Host paths:** Update `hostPath` volumes to match where you cloned this
repo and where your model checkpoint lives.

See [reference/config.md](../reference/config.md) for the complete list of
environment variables.

## Deploy

```bash
sudo kubectl apply -f deploy/robot/k8s/robot-brain.yaml
```

This single manifest deploys four workloads:

| Deployment | Node | Containers | Purpose |
|-----------|------|------------|---------|
| `robot-brain` | GPU server | `mcp-server` + `inference` | MCP tools + VLA inference |
| `camera-wrist` | GPU server | `camera` | Wrist camera (cam0) via ROS 2 |
| `camera-overhead` | GPU server | `camera` | Overhead camera (cam1) via ROS 2 |
| `motor-node` | Edge node | `motor` | Servo control over USB serial |

Verify all pods are running:

```bash
sudo kubectl get pods -n robot -o wide
```

You should see `robot-brain 2/2`, `motor-node 1/1`, and both camera pods
at `1/1`.

### Troubleshooting pods

If a pod is in `CrashLoopBackOff`:

```bash
# Check the MCP server logs
sudo kubectl logs -n robot deployment/robot-brain -c mcp-server

# Check inference logs
sudo kubectl logs -n robot deployment/robot-brain -c inference

# Check camera logs
sudo kubectl logs -n robot deployment/camera-wrist

# Check motor logs
sudo kubectl logs -n robot deployment/motor-node
```

Common issues:

- **Inference crashes on startup:** Model checkpoint not found or wrong
  format. Verify the `hostPath` volume and `MODEL_PATH` env var.
- **Camera pod restarts:** Wrong device path or camera not connected. Check
  `v4l2-ctl --list-devices` and update `CAM0_DEVICE`/`CAM1_DEVICE`.
- **Motor node can't open serial port:** Wrong port or permissions. Check
  `ls /dev/ttyACM*` on the edge node and verify the pod is privileged.

### Host-mounted source files

The MCP server and inference node source files are host-mounted into the
running containers via volume mounts. This means you can edit
`src/brain/mcp_server.py` or `src/brain/inference_node.py` on the GPU
server and restart the pod to pick up changes — no image rebuild required.

```bash
# After editing source files
sudo kubectl delete pod -n robot -l app=robot-brain
# The deployment controller recreates it automatically
```

Only rebuild the images when dependencies (Python packages, ROS 2 packages)
change.

### hostNetwork pods

All pods use `hostNetwork: true` for DDS multicast and direct USB access.
This means they bind directly to host ports. If you need to update a pod,
**delete the old one first** — Kubernetes cannot do a rolling update because
the new pod cannot bind the same port while the old one is still running:

```bash
sudo kubectl delete pod -n robot -l app=robot-brain
sudo kubectl apply -f deploy/robot/k8s/robot-brain.yaml
```

## ROS 2 DDS verification

With all pods running, verify that ROS 2 topics are flowing:

```bash
# Exec into the MCP server container (which has ROS 2 Jazzy)
sudo kubectl exec -n robot deployment/robot-brain -c mcp-server -- \
  ros2 topic list
```

You should see topics like `/camera/cam0/compressed`,
`/camera/cam1/compressed`, `/joint_states`, and others.

Check camera publishing rate:

```bash
sudo kubectl exec -n robot deployment/robot-brain -c mcp-server -- \
  ros2 topic hz /camera/cam0/compressed
```

Should show ~15-30 Hz.

### FastDDS SHM

The camera containers use ROS 2 Humble and the inference container uses
ROS 2 Jazzy. These use different FastDDS versions that are not compatible
over shared memory. The containers include a FastDDS configuration that
forces UDP transport:

```xml
<!-- deploy/robot/docker/fastdds_no_shm.xml -->
<dds>
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>udp_transport</transport_id>
      <type>UDPv4</type>
    </transport_descriptor>
  </transport_descriptors>
</dds>
```

This is already configured in the Dockerfiles and k8s manifests. If you
add new ROS 2 containers, make sure they include this configuration via
the `FASTRTPS_DEFAULT_PROFILES_FILE` environment variable.

## What you should have at the end

- [ ] All 4 container images built and pushed to registry
- [ ] `robot-brain` pod running 2/2 (MCP + inference)
- [ ] Both camera pods running 1/1
- [ ] `motor-node` pod running 1/1 on edge node
- [ ] ROS 2 topics flowing (camera, joint_states)
- [ ] Model checkpoint mounted and loaded

Next: [03-observability.md](03-observability.md) — adding APM, Fleet,
GPU metrics, and dashboards.
