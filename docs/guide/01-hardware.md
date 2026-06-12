# Hardware

This page covers what you need to build the system. It is not a full assembly
guide -- the SO-101 arm has excellent documentation upstream, and there is no
reason to duplicate it here.

## What you need

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU server | 1x NVIDIA GPU, 16 GB VRAM, 32 GB RAM, Ubuntu 22.04+ | 2x GPUs (16 GB each), 64 GB RAM |
| Edge node | Any Linux box with USB ports | Intel NUC or Raspberry Pi 5 |
| Robot arm | 1x SO-101 follower arm | Leader + follower pair |
| Cameras | 1x USB webcam (1080p) | 2x webcams (wrist + overhead) |
| Network | Gigabit Ethernet between GPU server and edge node | Required -- Wi-Fi adds too much jitter for DDS |

**Total cost** is roughly $1,500-2,500 depending on GPU choice and whether you
already have a spare Linux machine for the edge node.

## What matters vs. what is flexible

**Must get right:**

- **VRAM.** VLA models need 14-16 GB for inference. A single RTX 4060 Ti 16 GB
  works. An RTX 3090, 4070 Ti Super, or 4090 all work. Cards with 8 GB VRAM
  will not load the model.
- **USB controller separation.** If you run two cameras on the same USB root
  hub, bandwidth contention causes frame drops. Use `lsusb -t` to verify each
  camera is on a separate controller.
- **Wired network.** ROS 2 DDS relies on multicast and low-latency UDP. Gigabit
  Ethernet is the baseline.

**Flexible:**

- GPU brand and generation (any NVIDIA with 16 GB+ VRAM and CUDA support)
- Edge node hardware (NUC, old laptop, Raspberry Pi -- anything that runs
  Linux and has USB)
- Camera models (any V4L2-compatible USB webcam at 640x480 or higher)
- Number of GPUs (1 is enough for inference; 2 lets you run evaluation models
  in parallel)

## Robot arm

The SO-101 is a 6-DOF robot arm designed by HuggingFace for the LeRobot
project. It uses Feetech STS3215 servos and costs roughly $150 in parts.

- **Assembly guide:** [HuggingFace LeRobot SO-101](https://github.com/huggingface/lerobot/blob/main/examples/robots/so101/README.md)
- **LeRobot project:** [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- You need at minimum a follower arm (the one the model controls). A leader arm
  is required only for teleoperation during training data collection.

## Cameras

Two USB webcams provide the visual input for inference:

- **Wrist camera (cam0):** Mounted on or near the end effector. This is the
  primary input to the vision-language-action model.
- **Overhead camera (cam1):** Fixed above the workspace. Used for evaluation
  and CLIP confidence scoring.

Both cameras must be on separate USB controllers. Check with `lsusb -t` on
the machine where they are plugged in.

## Training data collection

Before the model can do anything useful, you need training episodes collected
via teleoperation (physically guiding the leader arm while the follower mirrors
and cameras record). LeRobot handles recording, dataset formatting, and upload.

- [LeRobot teleoperation guide](https://github.com/huggingface/lerobot/blob/main/examples/robots/so101/README.md)
- Plan for 50-100 episodes of the target task (pick up cube, move object, etc.)
- Each episode is roughly 20-30 seconds of demonstration

Training the VLA model itself is covered in the LeRobot training documentation.
This repository assumes you have a trained checkpoint ready to deploy.
