import numpy as np

# SMPL Joint Indices based on standard SMPL 24-joint format
PELVIS = 0
L_HIP = 1
R_HIP = 2
L_KNEE = 4
R_KNEE = 5
NECK = 12

def compute_leg_symmetry_simple(joints_3d):
    """
    Inspired by Bio-LSTM. Calculates the absolute angle of each leg
    relative to the spine, and the raw symmetry variance between them.
    """
    # 1. Define the center line (Spine vector: Neck down to Pelvis)
    spine_vec = joints_3d[PELVIS] - joints_3d[NECK]
    spine_dir = spine_vec / (np.linalg.norm(spine_vec) + 1e-9)

    # 2. Define the leg vectors (Hip down to Knee)
    l_leg_vec = joints_3d[L_KNEE] - joints_3d[L_HIP]
    r_leg_vec = joints_3d[R_KNEE] - joints_3d[R_HIP]

    l_leg_dir = l_leg_vec / (np.linalg.norm(l_leg_vec) + 1e-9)
    r_leg_dir = r_leg_vec / (np.linalg.norm(r_leg_vec) + 1e-9)

    # 3. Calculate absolute angles (in degrees) between legs and the center line
    theta_l_deg = np.degrees(np.arccos(np.clip(np.dot(l_leg_dir, spine_dir), -1.0, 1.0)))
    theta_r_deg = np.degrees(np.arccos(np.clip(np.dot(r_leg_dir, spine_dir), -1.0, 1.0)))

    # 4. Simple Symmetry Variance
    symmetry_variance = abs(theta_l_deg) - abs(theta_r_deg)

    return theta_l_deg, theta_r_deg, symmetry_variance