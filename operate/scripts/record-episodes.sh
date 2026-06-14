#!/bin/bash
# Record episodes with LeRobot.
# Wraps lerobot-record with best-practice defaults.
#
# Usage:
#   ./record-episodes.sh                           # 10 episodes, default task
#   ./record-episodes.sh --episodes 50             # 50 episodes
#   ./record-episodes.sh --task "Sort red and blue cubes"
#   ./record-episodes.sh --resume                  # resume existing dataset
#   ./record-episodes.sh --dataset my_new_task     # custom dataset name

set -euo pipefail

# Defaults
EPISODES=10
TASK="Pick up blue cube and place in bin"
DATASET="blue_cube_2cam"
RESUME=""
FOLLOWER_PORT="/dev/ttyACM0"
LEADER_PORT="/dev/ttyACM1"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)   EPISODES="$2"; shift 2 ;;
        --task)       TASK="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --resume)     RESUME="--resume=true"; shift ;;
        --follower)   FOLLOWER_PORT="$2"; shift 2 ;;
        --leader)     LEADER_PORT="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Recording ${EPISODES} episodes ==="
echo "Task: ${TASK}"
echo "Dataset: local/${DATASET}"
echo "Follower: ${FOLLOWER_PORT}"
echo "Leader: ${LEADER_PORT}"
echo ""
echo "Controls: → end episode | ← redo | ESC stop & save"
echo ""

# IMPORTANT: Stop the camera node before recording — lerobot-record
# opens its own camera handles and they'll conflict.
echo "WARNING: Make sure edge_camera_node.py is NOT running!"
echo "Press Enter to continue or Ctrl+C to abort..."
read -r

conda activate lerobot

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=follower_arm \
    --robot.cameras="{ \
        wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
        overhead: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} \
    }" \
    --teleop.type=so101_leader \
    --teleop.port=${LEADER_PORT} \
    --teleop.id=leader_arm \
    --display_data=false \
    --dataset.repo_id=local/${DATASET} \
    --dataset.num_episodes=${EPISODES} \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=120 \
    --dataset.single_task="${TASK}" \
    --dataset.push_to_hub=false \
    ${RESUME}

echo ""
echo "=== Recording complete ==="
echo "Dataset at: ~/.cache/huggingface/lerobot/local/${DATASET}"
echo ""
echo "Next steps:"
echo "  1. Transfer dataset for training (if recording on a different node):"
echo "     tar -czf ${DATASET}.tar.gz -C ~/.cache/huggingface/lerobot/local ${DATASET}"
echo "  2. Train: see operate/02-training.md for model-specific commands"
