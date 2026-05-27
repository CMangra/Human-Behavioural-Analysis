import os
import json
import cv2
import numpy as np
import torch

from visualizations.smpl_video_annotator import load_smpl_params


# =============================================================================
# STEP 5 DATA ANALYSIS: 3D SMPL ORIENTATION METRICS
# =============================================================================

SMPL_PARENT = np.array([
    -1, 0, 0, 0,
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    9, 9, 12,
    13, 14,
    16, 17,
    18, 19,
    20, 21
], dtype=int)

SMPL_HEAD_INDEX = 15
SMPL_LEFT_SHOULDER_INDEX = 16
SMPL_RIGHT_SHOULDER_INDEX = 17

HEAD_AXIS_CANDIDATES = [
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
]


def normalize_2d(v, eps=1e-9):
    v = np.asarray(v, dtype=float).reshape(2)
    n = np.linalg.norm(v)

    if n < eps or not np.isfinite(n):
        return None

    return v / n


def signed_deviation_deg(reference_dir, target_dir):
    """
    Signed yaw deviation in degrees.

    Convention:
        left relative to walking direction  -> negative
        right relative to walking direction -> positive
    """
    ref = normalize_2d(reference_dir)
    tgt = normalize_2d(target_dir)

    if ref is None or tgt is None:
        return np.nan

    cross_z = ref[0] * tgt[1] - ref[1] * tgt[0]
    dot = np.clip(np.dot(ref, tgt), -1.0, 1.0)

    angle_rad = np.arctan2(cross_z, dot)

    return -np.degrees(angle_rad)


def axis_angle_to_rotmat(axis_angle):
    rotmat, _ = cv2.Rodrigues(np.asarray(axis_angle, dtype=np.float64).reshape(3, 1))
    return rotmat.astype(np.float64)


def compute_global_rotations_from_pose(pose_72):
    """
    Converts SMPL local axis-angle pose parameters into global joint rotations.
    """
    local_rotations = []

    for i in range(24):
        aa = pose_72[i * 3:(i + 1) * 3]
        local_rotations.append(axis_angle_to_rotmat(aa))

    global_rotations = [None] * 24

    for i in range(24):
        parent = SMPL_PARENT[i]

        if parent < 0:
            global_rotations[i] = local_rotations[i]
        else:
            global_rotations[i] = global_rotations[parent] @ local_rotations[i]

    return global_rotations


def smpl_forward_joints(model, betas, pose, trans, device):
    """
    Runs SMPL forward pass and returns joints only.
    """
    betas_t = torch.tensor(betas.reshape(1, 10), dtype=torch.float32, device=device)
    global_orient_t = torch.tensor(pose[:3].reshape(1, 3), dtype=torch.float32, device=device)
    body_pose_t = torch.tensor(pose[3:].reshape(1, 69), dtype=torch.float32, device=device)
    transl_t = torch.tensor(trans.reshape(1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        output = model(
            betas=betas_t,
            global_orient=global_orient_t,
            body_pose=body_pose_t,
            transl=transl_t,
            return_verts=False,
        )

    return output.joints.detach().cpu().numpy()[0]


def compute_walking_direction_from_kinematics(kinematics, frame_id, past_frames):
    """
    Uses the existing centralized Step 2 smoothed trajectory.

    Direction at frame f:
        vector from smoothed position at f - past_frames to smoothed position at f.
    """
    sorted_frames = kinematics["sorted_frames"]
    xs = kinematics["xs"]
    ys = kinematics["ys"]

    if frame_id not in sorted_frames:
        return None

    idx = sorted_frames.index(frame_id)
    past_idx = idx - past_frames

    if past_idx < 0:
        return None

    p_now = np.array([xs[idx], ys[idx]], dtype=float)
    p_past = np.array([xs[past_idx], ys[past_idx]], dtype=float)

    return normalize_2d(p_now - p_past)


def choose_head_forward_axis(rows):
    """
    Calibrates which SMPL head local axis is most likely face-forward.

    We choose the candidate axis that most often aligns with the person's walking
    direction across the available track.
    """
    best_axis = HEAD_AXIS_CANDIDATES[0]
    best_score = -np.inf

    for axis in HEAD_AXIS_CANDIDATES:
        scores = []

        for row in rows:
            walking_dir = row.get("walking_dir")
            head_global_rot = row.get("head_global_rot")

            if walking_dir is None or head_global_rot is None:
                continue

            candidate_3d = head_global_rot @ axis
            candidate_2d = normalize_2d(candidate_3d[:2])

            if candidate_2d is None:
                continue

            scores.append(np.dot(walking_dir, candidate_2d))

        if not scores:
            continue

        score = float(np.nanmean(scores))

        if score > best_score:
            best_score = score
            best_axis = axis

    return best_axis, best_score


def build_smpl_json_path(data_dir, sequence, frame_id, tid):
    return os.path.join(
        data_dir,
        "labels",
        "3d",
        "smpl",
        sequence,
        f"{sequence}_{frame_id:07d}_{tid}.json"
    )


def compute_smpl_orientation_track(
    data_dir,
    sequence,
    tid,
    kinematics,
    smpl_model,
    device,
    past_frames,
):
    """
    Computes head and shoulder deviation for all frames where both:
        - Step 2 trajectory exists
        - official PedX SMPL JSON exists

    It deliberately reuses Step 2 kinematics for walking direction.
    """
    sorted_frames = kinematics["sorted_frames"]

    rows = []

    for frame_id in sorted_frames:
        json_path = build_smpl_json_path(data_dir, sequence, frame_id, tid)

        if not os.path.exists(json_path):
            continue

        betas, pose, trans = load_smpl_params(json_path)
        joints = smpl_forward_joints(
            model=smpl_model,
            betas=betas,
            pose=pose,
            trans=trans,
            device=device,
        )

        walking_dir = compute_walking_direction_from_kinematics(
            kinematics=kinematics,
            frame_id=frame_id,
            past_frames=past_frames,
        )

        global_rots = compute_global_rotations_from_pose(pose)
        head_global_rot = global_rots[SMPL_HEAD_INDEX]

        left_shoulder = joints[SMPL_LEFT_SHOULDER_INDEX]
        right_shoulder = joints[SMPL_RIGHT_SHOULDER_INDEX]

        shoulder_line_2d = normalize_2d((right_shoulder - left_shoulder)[:2])

        shoulder_forward = None

        if shoulder_line_2d is not None and walking_dir is not None:
            normal_1 = np.array([-shoulder_line_2d[1], shoulder_line_2d[0]])
            normal_2 = -normal_1

            # Pick the normal that generally faces the walking direction.
            if np.dot(normal_1, walking_dir) >= np.dot(normal_2, walking_dir):
                shoulder_forward = normal_1
            else:
                shoulder_forward = normal_2

        rows.append({
            "frame": frame_id,
            "walking_dir": walking_dir,
            "head_global_rot": head_global_rot,
            "shoulder_forward": shoulder_forward,
        })

    head_axis, head_axis_score = choose_head_forward_axis(rows)

    final_rows = []

    for row in rows:
        walking_dir = row["walking_dir"]

        if walking_dir is None:
            head_dev = np.nan
            shoulder_dev = np.nan
        else:
            head_vec_3d = row["head_global_rot"] @ head_axis
            head_vec_2d = normalize_2d(head_vec_3d[:2])

            if head_vec_2d is None:
                head_dev = np.nan
            else:
                head_dev = signed_deviation_deg(walking_dir, head_vec_2d)

            shoulder_forward = row["shoulder_forward"]

            if shoulder_forward is None:
                shoulder_dev = np.nan
            else:
                shoulder_dev = signed_deviation_deg(walking_dir, shoulder_forward)

        final_rows.append({
            "frame": row["frame"],
            "head_deviation_deg": head_dev,
            "shoulder_deviation_deg": shoulder_dev,
        })

    return final_rows, head_axis, head_axis_score


def get_angular_velocity_frame_series_from_kinematics(kinematics):
    """
    Reuses the exact angular velocity sequence from Step 2.

    In turn_detection.compute_kinematics:
        smoothed_ang_vel index i corresponds visually to sorted_frames[i + 1]
        as used in turn_math_debugger.py.
    """
    sorted_frames = kinematics["sorted_frames"]
    smoothed_ang_vel = kinematics["smoothed_ang_vel"]

    frames = np.array(sorted_frames[1:-1], dtype=int)
    values = np.asarray(smoothed_ang_vel, dtype=float)

    n = min(len(frames), len(values))

    return frames[:n], values[:n]


def get_peak_frame_for_onset(kinematics, onset_frame):
    """
    Reuses Step 2 detected peaks.

    Selects the first peak after the onset frame. If no peak after onset exists,
    returns the nearest detected peak.
    """
    sorted_frames = kinematics["sorted_frames"]
    peaks = kinematics["peaks"]

    if peaks is None or len(peaks) == 0:
        return None

    peak_frames = []

    for peak_idx in peaks:
        candidate_idx = int(peak_idx) + 1

        if 0 <= candidate_idx < len(sorted_frames):
            peak_frames.append(sorted_frames[candidate_idx])

    if not peak_frames:
        return None

    after = [f for f in peak_frames if f >= onset_frame]

    if after:
        return min(after)

    return min(peak_frames, key=lambda f: abs(f - onset_frame))