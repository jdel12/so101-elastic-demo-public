"""SO-101 forward kinematics: joint angles (degrees) -> end-effector XYZ (meters).

Geometry extracted from TheRobotStudio/SO-ARM100 URDF (so101_new_calib.urdf).
Pure numpy, no external dependencies beyond what inference already uses.
"""

import numpy as np


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]])


def _translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])


def _rpy_to_matrix(roll, pitch, yaw):
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _joint_transform(xyz, rpy, angle_rad):
    """Build the transform for a revolute joint: fixed offset (xyz+rpy) then rotation around Z."""
    T = _translate(*xyz) @ _rpy_to_matrix(*rpy)
    return T @ _rot_z(angle_rad)


# URDF joint parameters: (xyz, rpy) for each joint in kinematic chain order
# Source: TheRobotStudio/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
_JOINTS = [
    # shoulder_pan: base_link -> shoulder_link
    {'xyz': (0.0388353, 0.0, 0.0624), 'rpy': (3.14159, 0.0, -3.14159)},
    # shoulder_lift: shoulder_link -> upper_arm_link
    {'xyz': (-0.0303992, -0.0182778, -0.0542), 'rpy': (-1.5708, -1.5708, 0.0)},
    # elbow_flex: upper_arm_link -> lower_arm_link
    {'xyz': (-0.11257, -0.028, 0.0), 'rpy': (0.0, 0.0, 1.5708)},
    # wrist_flex: lower_arm_link -> wrist_link
    {'xyz': (-0.1349, 0.0052, 0.0), 'rpy': (0.0, 0.0, -1.5708)},
    # wrist_roll: wrist_link -> gripper_link
    {'xyz': (0.0, -0.0611, 0.0181), 'rpy': (1.5708, 0.0486795, 3.14159)},
]

# Fixed transform from gripper_link to gripper tip (gripper_frame_joint)
_GRIPPER_TIP = _translate(-0.0079, -0.000218121, -0.0981274) @ _rpy_to_matrix(0, 3.14159, 0)


def forward_kinematics(joint_angles_deg):
    """Compute end-effector position from 6 joint angles in degrees.

    Args:
        joint_angles_deg: [shoulder_pan, shoulder_lift, elbow_flex,
                           wrist_flex, wrist_roll, gripper] in degrees.
                           Only first 5 are used (gripper doesn't affect position).

    Returns:
        dict with 'x', 'y', 'z' in meters (workspace coordinates),
        and 'elbow_xyz' for the elbow position.
    """
    angles_rad = np.radians(joint_angles_deg[:5])

    T = np.eye(4)
    positions = {}

    for i, (joint, angle) in enumerate(zip(_JOINTS, angles_rad)):
        T = T @ _joint_transform(joint['xyz'], joint['rpy'], angle)
        if i == 2:
            positions['elbow'] = T[:3, 3].copy()

    # Apply fixed gripper tip transform
    T_tip = T @ _GRIPPER_TIP

    return {
        'x': float(T_tip[0, 3]),
        'y': float(T_tip[1, 3]),
        'z': float(T_tip[2, 3]),
        'elbow_x': float(positions['elbow'][0]),
        'elbow_y': float(positions['elbow'][1]),
        'elbow_z': float(positions['elbow'][2]),
    }


def forward_kinematics_batch(joint_trajectories):
    """Compute FK for a list of joint angle sets. Returns list of position dicts."""
    return [forward_kinematics(angles) for angles in joint_trajectories]


if __name__ == '__main__':
    # Sanity check: home position (all zeros) and a known pose
    home = forward_kinematics([0, 0, 0, 0, 0, 0])
    print(f"Home position: x={home['x']*1000:.1f}mm y={home['y']*1000:.1f}mm z={home['z']*1000:.1f}mm")

    # Typical V1 success start position
    start = forward_kinematics([-2.7, -105.2, 85.3, 59.8, 6.0, 18.9])
    print(f"V1 start:      x={start['x']*1000:.1f}mm y={start['y']*1000:.1f}mm z={start['z']*1000:.1f}mm")

    # Typical V1 approach position (reaching for cube)
    approach = forward_kinematics([-3.0, -10.0, 5.0, 75.0, 6.5, 28.0])
    print(f"V1 approach:   x={approach['x']*1000:.1f}mm y={approach['y']*1000:.1f}mm z={approach['z']*1000:.1f}mm")

    # V1 placement position (above bin)
    place = forward_kinematics([40.0, 20.0, -15.0, 80.0, 3.0, 5.0])
    print(f"V1 placement:  x={place['x']*1000:.1f}mm y={place['y']*1000:.1f}mm z={place['z']*1000:.1f}mm")
