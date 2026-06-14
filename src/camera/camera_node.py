#!/usr/bin/env python3
"""
Edge Camera Node — publishes compressed camera frames over ROS2.

Functional style. The only class is the ROS2 Node itself (required by rclpy).

Usage:
  python3 edge_camera_node.py                # single camera (cam0)
  python3 edge_camera_node.py --all          # all three cameras
  python3 edge_camera_node.py --camera cam1  # specific camera
"""

import argparse
import sys
import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import CompressedImage


# --- Configuration ---

import os
import glob

CAMERAS = {
    'cam0': os.environ.get('CAM0_DEVICE', '/dev/video0'),
    'cam1': os.environ.get('CAM1_DEVICE', '/dev/video2'),
    'cam2': os.environ.get('CAM2_DEVICE', '/dev/video4'),
}

TARGET_FPS = 15
JPEG_QUALITY = 80
FRAME_SIZE = (480, 480)


# --- Device discovery ---

def find_capture_device_on_bus(usb_bus):
    """Find the capture-capable video device (index=0) on a given USB bus (e.g. 'usb5')."""
    for vdev in sorted(glob.glob('/sys/class/video4linux/video*')):
        try:
            idx = open(os.path.join(vdev, 'index')).read().strip()
            if idx != '0':
                continue
            devpath = os.path.realpath(os.path.join(vdev, 'device', '..'))
            if usb_bus in devpath:
                return '/dev/' + os.path.basename(vdev)
        except (OSError, IOError):
            continue
    return None


def resolve_device(device_or_bus):
    """Resolve a device path or USB bus specifier (e.g. 'usb5') to a /dev/video* path."""
    if device_or_bus.startswith('usb'):
        return find_capture_device_on_bus(device_or_bus)
    return device_or_bus


# --- Pure functions ---

def compress_frame(frame, size=FRAME_SIZE, quality=JPEG_QUALITY):
    """Resize and JPEG-compress a raw frame. Returns bytes or None."""
    resized = cv2.resize(frame, size)
    success, encoded = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return None
    return np.array(encoded).tobytes()


def build_message(image_bytes, frame_id, clock):
    """Build a CompressedImage message from JPEG bytes."""
    msg = CompressedImage()
    msg.header.stamp = clock.now().to_msg()
    msg.header.frame_id = frame_id
    msg.format = 'jpeg'
    msg.data = image_bytes
    return msg


def open_camera(device_path, width=640, height=480, fps=30):
    """Open a camera device. Returns capture object or None."""
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def open_cameras(camera_configs, logger):
    """Open all requested cameras. Resolves USB bus specifiers to device paths."""
    captures = {}
    for name, device_or_bus in camera_configs.items():
        device = resolve_device(device_or_bus)
        if device is None:
            logger.error(f'Failed to find device for {name} ({device_or_bus})')
            continue
        cap = open_camera(device)
        if cap is None:
            logger.error(f'Failed to open {name} at {device}')
            continue
        captures[name] = cap
        camera_configs[name] = device
        logger.info(f'{name}: opened {device} (from {device_or_bus})')
    return captures


def release_cameras(captures):
    """Release all open camera captures."""
    for cap in captures.values():
        cap.release()


def parse_args():
    """Parse command line arguments. Returns selected camera configs."""
    parser = argparse.ArgumentParser(description='Edge Camera Node')
    parser.add_argument('--all', action='store_true', help='Publish all 3 cameras')
    parser.add_argument('--camera', type=str, default='cam0', help='Camera name (cam0, cam1, cam2)')
    parsed = parser.parse_args()

    if parsed.all:
        return CAMERAS

    if parsed.camera not in CAMERAS:
        print(f'Unknown camera: {parsed.camera}. Options: {list(CAMERAS.keys())}')
        sys.exit(1)

    return {parsed.camera: CAMERAS[parsed.camera]}


# --- ROS2 wiring ---

def create_publishers(node, camera_names):
    """Create a CompressedImage publisher for each camera."""
    return {
        name: node.create_publisher(CompressedImage, f'/camera/{name}/compressed', 10)
        for name in camera_names
    }


MAX_CONSECUTIVE_DROPS = 10

drop_counts = {}


def capture_and_publish(captures, publishers, node, camera_configs):
    """Read one frame from each camera, compress, and publish. Auto-recovers on sustained drops."""
    dead = []
    for name, cap in list(captures.items()):
        ret, frame = cap.read()
        if not ret:
            drop_counts[name] = drop_counts.get(name, 0) + 1
            if drop_counts[name] >= MAX_CONSECUTIVE_DROPS:
                old_device = camera_configs[name]
                node.get_logger().warning(f'{name}: {drop_counts[name]} consecutive drops, recovering...')
                cap.release()
                env_key = name.upper() + '_DEVICE'
                bus_or_path = os.environ.get(env_key, old_device)
                device = resolve_device(bus_or_path) or old_device
                if device != old_device:
                    node.get_logger().info(f'{name}: device moved {old_device} → {device}')
                new_cap = open_camera(device)
                if new_cap is not None:
                    captures[name] = new_cap
                    camera_configs[name] = device
                    drop_counts[name] = 0
                    node.get_logger().info(f'{name}: recovered on {device}')
                else:
                    node.get_logger().error(f'{name}: reopen failed at {device}, removing')
                    dead.append(name)
            continue

        drop_counts[name] = 0
        image_bytes = compress_frame(frame)
        if image_bytes is None:
            continue

        msg = build_message(image_bytes, name, node.get_clock())
        publishers[name].publish(msg)

    for name in dead:
        captures.pop(name, None)
    if not captures:
        node.get_logger().fatal('All cameras lost. Exiting for restart.')
        raise SystemExit(1)


# --- Main ---

def main():
    camera_configs = parse_args()

    rclpy.init()
    node = rclpy.create_node('edge_camera_node')
    logger = node.get_logger()

    captures = open_cameras(camera_configs, logger)
    if not captures:
        logger.error('No cameras opened. Exiting.')
        node.destroy_node()
        rclpy.shutdown()
        return

    publishers = create_publishers(node, captures.keys())

    node.create_timer(
        1.0 / TARGET_FPS,
        lambda: capture_and_publish(captures, publishers, node, camera_configs)
    )

    logger.info(f'Publishing {len(captures)} camera(s) at {TARGET_FPS} FPS')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        release_cameras(captures)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
