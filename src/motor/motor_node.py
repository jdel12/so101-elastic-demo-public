#!/usr/bin/env python3
"""
Motor Node v3 — runs on the NUC, receives joint commands over ROS2
and drives the SO-101 follower arm using LeRobot's API.

Graceful retry: if the arm isn't connected on startup, retries
every 5 seconds until it succeeds. Also recovers from runtime
disconnections.

Topics:
  Subscribes:
    /joint_commands (Float32MultiArray — 6 joint positions)

  Publishes:
    /joint_states (Float32MultiArray — 6 joint positions, ~30Hz)
"""

import os
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


# --- Configuration ---

FOLLOWER_PORT = os.environ.get('FOLLOWER_PORT', '/dev/ttyACM0')
FOLLOWER_ID = os.environ.get('FOLLOWER_ID', 'follower_arm')
_mrt_raw = os.environ.get('MAX_RELATIVE_TARGET', '')
MAX_RELATIVE_TARGET = None if _mrt_raw.lower() in ('', 'none') else float(_mrt_raw)
STATE_PUBLISH_HZ = 30
RETRY_INTERVAL_SEC = 5

JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]


# --- Helpers ---

def values_to_action_dict(values):
    """Convert a list of 6 floats to the action dict LeRobot expects."""
    return {f'{name}.pos': float(val) for name, val in zip(JOINT_NAMES, values)}


def observation_to_values(obs_dict):
    """Extract joint position values from LeRobot observation dict."""
    return [float(obs_dict.get(f'{name}.pos', 0.0)) for name in JOINT_NAMES]


def connect_with_retry(port, robot_id, logger=None):
    """Attempt to connect to the follower arm, retrying on failure."""
    while True:
        try:
            config = SOFollowerRobotConfig(
                port=port, id=robot_id,
                max_relative_target=MAX_RELATIVE_TARGET,
            )
            robot = SOFollower(config)
            robot.connect(calibrate=False)
            if logger:
                logger.info(f'Follower arm connected on {port}')
            else:
                print(f'Follower arm connected on {port}')
            return robot
        except Exception as e:
            msg = f'Connection failed ({e}), retrying in {RETRY_INTERVAL_SEC}s...'
            if logger:
                logger.warn(msg)
            else:
                print(msg)
            time.sleep(RETRY_INTERVAL_SEC)


# --- ROS2 Node ---

class MotorNode(Node):

    def __init__(self, robot):
        super().__init__('motor_node')
        self.robot = robot
        self.lock = threading.Lock()
        self.connected = True

        self.create_subscription(
            Float32MultiArray, '/joint_commands',
            self._on_command, 10
        )

        self.state_pub = self.create_publisher(Float32MultiArray, '/joint_states', 10)
        self.create_timer(1.0 / STATE_PUBLISH_HZ, self._publish_state)

        # Reconnection timer — checks every 5 seconds if disconnected
        self.create_timer(RETRY_INTERVAL_SEC, self._check_reconnect)

        self.get_logger().info('Motor node ready. Listening for /joint_commands')

    def _on_command(self, msg):
        if not self.connected:
            return
        with self.lock:
            try:
                action = values_to_action_dict(msg.data[:6])
                self.robot.send_action(action)
            except Exception as e:
                self.get_logger().error(f'Command error: {e}')
                self.connected = False

    def _publish_state(self):
        if not self.connected:
            return
        try:
            obs = self.robot.get_observation()
            values = observation_to_values(obs)
            msg = Float32MultiArray()
            msg.data = values
            self.state_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'State read error: {e}')
            self.connected = False

    def _check_reconnect(self):
        if self.connected:
            return
        self.get_logger().info('Attempting reconnection...')
        try:
            with self.lock:
                try:
                    self.robot.disconnect()
                except Exception:
                    pass
                config = SOFollowerRobotConfig(
                    port=FOLLOWER_PORT, id=FOLLOWER_ID,
                    max_relative_target=MAX_RELATIVE_TARGET,
                )
                self.robot = SOFollower(config)
                self.robot.connect(calibrate=False)
                self.connected = True
                self.get_logger().info('Reconnected successfully')
        except Exception as e:
            self.get_logger().warn(f'Reconnection failed: {e}')


# --- Main ---

def main():
    print('=== SO-101 Motor Node v3 ===')
    print(f'Connecting to follower arm on {FOLLOWER_PORT}...')
    print(f'max_relative_target={MAX_RELATIVE_TARGET} deg/step')

    robot = connect_with_retry(FOLLOWER_PORT, FOLLOWER_ID)

    rclpy.init()
    node = MotorNode(robot)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        try:
            robot.disconnect()
        except Exception:
            pass
        print('Motor node stopped.')


if __name__ == '__main__':
    main()
