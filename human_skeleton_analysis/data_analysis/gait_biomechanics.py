import numpy as np

# SMPL Joint Indices based on standard SMPL 24-joint format
PELVIS = 0
L_HIP = 1
R_HIP = 2
L_KNEE = 4
R_KNEE = 5
L_FOOT = 10
R_FOOT = 11
NECK = 12
HEAD = 15


def compute_leg_symmetry_simple(joints_3d):
    """
    Inspired by Bio-LSTM. Calculates the absolute angle of each leg
    relative to the spine, and the raw symmetry variance between them.
    """
    spine_vec = joints_3d[PELVIS] - joints_3d[NECK]
    spine_dir = spine_vec / (np.linalg.norm(spine_vec) + 1e-9)

    l_leg_vec = joints_3d[L_KNEE] - joints_3d[L_HIP]
    r_leg_vec = joints_3d[R_KNEE] - joints_3d[R_HIP]

    l_leg_dir = l_leg_vec / (np.linalg.norm(l_leg_vec) + 1e-9)
    r_leg_dir = r_leg_vec / (np.linalg.norm(r_leg_vec) + 1e-9)

    theta_l_deg = np.degrees(np.arccos(np.clip(np.dot(l_leg_dir, spine_dir), -1.0, 1.0)))
    theta_r_deg = np.degrees(np.arccos(np.clip(np.dot(r_leg_dir, spine_dir), -1.0, 1.0)))

    symmetry_variance = abs(theta_l_deg) - abs(theta_r_deg)

    return theta_l_deg, theta_r_deg, symmetry_variance


def compute_foot_contact(joints_3d, lidar_max_z):
    """
    Bio-LSTM Foot-Ground Reaction Consistency.
    Inverts the +Z down coordinate system so that +Height is intuitively "Up"
    for the thesis visualizations.
    """
    logical_ground = -lidar_max_z
    logical_l_foot = -joints_3d[L_FOOT, 2]
    logical_r_foot = -joints_3d[R_FOOT, 2]

    dist_l = logical_l_foot - logical_ground
    dist_r = logical_r_foot - logical_ground

    lowest_foot_dist = min(dist_l, dist_r)

    return logical_l_foot, logical_r_foot, logical_ground, lowest_foot_dist


def compute_skeleton_height(joints_3d):
    """
    Calculates the anatomical height of the skeleton as the 3D Euclidean distance
    from the Head joint (15) to the midpoint between the two Foot joints (10, 11).
    """
    head_pos = joints_3d[HEAD]
    left_foot = joints_3d[L_FOOT]
    right_foot = joints_3d[R_FOOT]

    mid_foot_pos = (left_foot + right_foot) / 2.0

    # 3D Euclidean distance formula
    height_m = np.linalg.norm(head_pos - mid_foot_pos)

    return height_m, head_pos, mid_foot_pos