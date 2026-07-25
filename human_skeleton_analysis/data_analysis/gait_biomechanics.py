import numpy as np

PELVIS = 0
L_HIP = 1
R_HIP = 2
L_KNEE = 4
R_KNEE = 5
NECK = 12

def compute_leg_symmetry_bio_lstm(joints_3d):
    """
    Extracts the symmetry metric using the exact math from Bio-LSTM Equation (6).
    The paper utilizes the cosine of the angle (the dot product of the normalized vectors).
    """
    spine_vec = joints_3d[PELVIS] - joints_3d[NECK]
    spine_dir = spine_vec / (np.linalg.norm(spine_vec) + 1e-9)

    l_leg_vec = joints_3d[L_KNEE] - joints_3d[L_HIP]
    r_leg_vec = joints_3d[R_KNEE] - joints_3d[R_HIP]

    l_leg_dir = l_leg_vec / (np.linalg.norm(l_leg_vec) + 1e-9)
    r_leg_dir = r_leg_vec / (np.linalg.norm(r_leg_vec) + 1e-9)

    # Bio-LSTM Math: Cosine of the angle via dot product
    cos_theta_l = np.clip(np.dot(l_leg_dir, spine_dir), -1.0, 1.0)
    cos_theta_r = np.clip(np.dot(r_leg_dir, spine_dir), -1.0, 1.0)

    # Calculate raw degrees strictly for Constraint A (Maximum Extension Limit)
    theta_l_deg = np.degrees(np.arccos(cos_theta_l))
    theta_r_deg = np.degrees(np.arccos(cos_theta_r))

    # Bio-LSTM Delta: cos(theta_L) - cos(theta_R)
    bio_lstm_sym_delta = cos_theta_l - cos_theta_r

    return theta_l_deg, theta_r_deg, cos_theta_l, cos_theta_r, bio_lstm_sym_delta