#!/usr/bin/env bash
# Reset the follower arm to training start position with smooth interpolation.
# Runs inside the robot-brain inference container via kubectl exec.
#
# Usage: ./scripts/reset-arm.sh
#
# Set GPU_NODE_NAME to the hostname/SSH alias of your GPU server.

set -euo pipefail

: "${GPU_NODE_NAME:?Set GPU_NODE_NAME to the hostname or SSH alias of your GPU server}"
GPU_HOST="$GPU_NODE_NAME"
NAMESPACE="robot"
DEPLOYMENT="robot-brain"
CONTAINER="inference"

# Training start position
TARGET="[-2.0, -104.0, 78.0, 68.0, 4.0, 24.0]"
STEPS=90
FPS=30

echo "Resetting arm to training start position..."

ssh "$GPU_HOST" "sudo kubectl exec -n $NAMESPACE deployment/$DEPLOYMENT -c $CONTAINER -- bash -c 'source /opt/ros/jazzy/setup.bash && python3 -c \"
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import time

rclpy.init()
node = Node(\\\"reset_arm\\\")
pub = node.create_publisher(Float32MultiArray, \\\"/joint_commands\\\", 10)
ctrl = node.create_publisher(String, \\\"/inference_control\\\", 10)

# Stop inference first
stop = String()
stop.data = \\\"stop\\\"
ctrl.publish(stop)
time.sleep(0.3)

# Read current position
current = None
def cb(msg):
    global current
    current = list(msg.data[:6])
sub = node.create_subscription(Float32MultiArray, \\\"/joint_states\\\", cb, 10)

for _ in range(50):
    rclpy.spin_once(node, timeout_sec=0.1)
    if current:
        break

if not current:
    print(\\\"ERROR: no joint states received\\\")
else:
    target = $TARGET
    steps = $STEPS
    print(f\\\"Moving from {[round(c,1) for c in current]} to {target}\\\")
    for i in range(steps):
        t = i / (steps - 1)
        t = t * t * (3 - 2 * t)  # ease-in-out
        pos = [current[j] + (target[j] - current[j]) * t for j in range(6)]
        msg = Float32MultiArray()
        msg.data = pos
        pub.publish(msg)
        time.sleep(1.0 / $FPS)
    print(\\\"Reset complete\\\")

node.destroy_node()
rclpy.shutdown()
\"'"

echo "Done."
