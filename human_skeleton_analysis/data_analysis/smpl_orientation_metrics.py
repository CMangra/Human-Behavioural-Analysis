import os
import cv2
import numpy as np
import torch

from visualizations.smpl_video_annotator import load_smpl_params
from data_analysis.timestamp_utils import (
    relative_time_seconds,
    frame_time_seconds,
)


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

HEAD_AXES = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}


def normalize_2d(v, eps=1e-9):
    v = np.asarray(v, dtype=float).reshape(2)
    n = np.linalg.norm(v)

    if n < eps or not np.isfinite(n):
        return None

    return v / n


def signed_deviation_deg(reference_dir, target_dir):
    """
    Signed yaw deviation.

    Convention:
        left relative to reference direction  -> negative
        right relative to reference direction -> positive
    """
    ref = normalize_2d(reference_dir)
    tgt = normalize_2d(target_dir)

    if ref is None or tgt is None:
        return np.nan

    cross_z = ref[0] * tgt[1] - ref[1] * tgt[0]
    dot = np.clip(np.dot(ref, tgt), -1.0, 1.0)

    angle = np.degrees(np.arctan2(cross_z, dot))
    return -angle


def axis_angle_to_rotmat(axis_angle):
    rotmat, _ = cv2.Rodrigues(np.asarray(axis_angle, dtype=np.float64).reshape(3, 1))
    return rotmat.astype(np.float64)


def compute_global_rotations_from_pose(pose_72):
    local_rots = []

    for i in range(24):
        aa = pose_72[i * 3:(i + 1) * 3]
        local_rots.append(axis_angle_to_rotmat(aa))

    global_rots = [None] * 24

    for i in range(24):
        parent = SMPL_PARENT[i]

        if parent < 0:
            global_rots[i] = local_rots[i]
        else:
            global_rots[i] = global_rots[parent] @ local_rots[i]

    return global_rots


def smpl_forward_joints(model, betas, pose, trans, device):
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


def build_smpl_json_path(data_dir, sequence, frame_id, tid):
    return os.path.join(
        data_dir,
        "labels",
        "3d",
        "smpl",
        sequence,
        f"{sequence}_{frame_id:07d}_{tid}.json",
    )


def get_kinematic_index(kinematics, frame_id):
    sorted_frames = kinematics["sorted_frames"]

    if frame_id not in sorted_frames:
        return None

    return sorted_frames.index(frame_id)


def walking_dir_past_seconds(kinematics, frame_id, past_seconds, frame_to_time):
    """
    Walking direction using real timestamp duration.

    Finds the frame closest to current_time - past_seconds, then computes
    displacement from that frame to the current frame.
    """
    sorted_frames = np.asarray(kinematics["sorted_frames"], dtype=int)
    xs = np.asarray(kinematics["xs"], dtype=float)
    ys = np.asarray(kinematics["ys"], dtype=float)

    if frame_id not in sorted_frames:
        return None

    idx = list(sorted_frames).index(frame_id)

    current_t = frame_time_seconds(frame_id, frame_to_time)
    target_t = current_t - past_seconds

    candidate_indices = [
        i for i, f in enumerate(sorted_frames)
        if frame_time_seconds(int(f), frame_to_time) <= target_t
    ]

    if not candidate_indices:
        return None

    past_idx = candidate_indices[-1]

    p_now = np.array([xs[idx], ys[idx]], dtype=float)
    p_past = np.array([xs[past_idx], ys[past_idx]], dtype=float)

    return normalize_2d(p_now - p_past)


def walking_dir_smoothed_tangent(kinematics, frame_id):
    idx = get_kinematic_index(kinematics, frame_id)

    if idx is None:
        return None

    xs = kinematics["xs"]
    ys = kinematics["ys"]

    if idx <= 0 or idx >= len(xs) - 1:
        return None

    p_prev = np.array([xs[idx - 1], ys[idx - 1]], dtype=float)
    p_next = np.array([xs[idx + 1], ys[idx + 1]], dtype=float)

    return normalize_2d(p_next - p_prev)


def stable_pre_turn_heading(
    kinematics,
    onset_frame,
    baseline_start_s,
    baseline_end_s,
    frame_to_time,
):
    """
    Fixed event-level walking reference using timestamps.

    Uses displacement between average position near baseline start and average
    position near baseline end.
    """
    sorted_frames = np.asarray(kinematics["sorted_frames"], dtype=int)
    xs = np.asarray(kinematics["xs"], dtype=float)
    ys = np.asarray(kinematics["ys"], dtype=float)

    times = np.array([
        relative_time_seconds(int(f), onset_frame, frame_to_time)
        for f in sorted_frames
    ], dtype=float)

    start_mask = (times >= baseline_start_s) & (times <= baseline_start_s + 0.5)
    end_mask = (times >= baseline_end_s - 0.5) & (times <= baseline_end_s)

    if np.sum(start_mask) < 2 or np.sum(end_mask) < 2:
        return None

    p_start = np.array([np.nanmean(xs[start_mask]), np.nanmean(ys[start_mask])])
    p_end = np.array([np.nanmean(xs[end_mask]), np.nanmean(ys[end_mask])])

    return normalize_2d(p_end - p_start)


def baseline_correct(values, times, baseline_start_s, baseline_end_s):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)

    mask = (
        np.isfinite(values)
        & (times >= baseline_start_s)
        & (times <= baseline_end_s)
    )

    if np.sum(mask) < 3:
        return values, np.nan

    baseline = np.nanmean(values[mask])
    return values - baseline, baseline


def choose_best_head_axis_by_stability(rows, times, baseline_start_s, baseline_end_s):
    """
    Chooses head axis with lowest baseline standard deviation.

    This is diagnostic only. It does not prove anatomical gaze direction.
    """
    best_axis = None
    best_score = np.inf

    baseline_mask = (times >= baseline_start_s) & (times <= baseline_end_s)

    for axis_name in HEAD_AXES.keys():
        values = np.array([r[f"head_{axis_name}_stable"] for r in rows], dtype=float)
        vals = values[baseline_mask & np.isfinite(values)]

        if len(vals) < 3:
            continue

        score = np.nanstd(vals)

        if score < best_score:
            best_score = score
            best_axis = axis_name

    return best_axis, best_score


def angular_velocity_frame_series_from_kinematics(kinematics):
    """
    Reuses Step 2 angular velocity values.

    Note:
        Values are degrees/frame because compute_kinematics() computes heading
        change per frame, not per timestamp-second.
    """
    frames = np.asarray(kinematics["sorted_frames"][1:-1], dtype=int)
    values = np.asarray(kinematics["smoothed_ang_vel"], dtype=float)

    n = min(len(frames), len(values))
    return frames[:n], values[:n]


def peak_frame_after_onset(kinematics, onset_frame):
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


def compute_event_orientation_sensitivity(
    data_dir,
    sequence,
    tid,
    onset_frame,
    kinematics,
    smpl_model,
    device,
    frame_to_time,
    pre_seconds=4.0,
    post_seconds=3.0,
    baseline_start_s=-3.0,
    baseline_end_s=-1.5,
):
    """
    Computes timestamp-aware Step 5 orientation sensitivity for one turn event.

    Output contains:
        - event dataframe
        - selected head axis
        - peak frame/time
        - angular velocity arrays for plotting
    """
    peak_frame = peak_frame_after_onset(kinematics, onset_frame)

    stable_heading = stable_pre_turn_heading(
        kinematics=kinematics,
        onset_frame=onset_frame,
        baseline_start_s=baseline_start_s,
        baseline_end_s=baseline_end_s,
        frame_to_time=frame_to_time,
    )

    if stable_heading is None:
        return None

    frames = [
        f for f in kinematics["sorted_frames"]
        if -pre_seconds <= relative_time_seconds(f, onset_frame, frame_to_time) <= post_seconds
    ]

    if not frames:
        return None

    start_frame = min(frames)
    end_frame = max(frames)

    rows = []

    for frame_id in frames:
        json_path = build_smpl_json_path(data_dir, sequence, frame_id, tid)

        if not os.path.exists(json_path):
            continue

        betas, pose, trans = load_smpl_params(json_path)
        joints = smpl_forward_joints(smpl_model, betas, pose, trans, device)

        global_rots = compute_global_rotations_from_pose(pose)
        head_rot = global_rots[SMPL_HEAD_INDEX]

        left_shoulder = joints[SMPL_LEFT_SHOULDER_INDEX]
        right_shoulder = joints[SMPL_RIGHT_SHOULDER_INDEX]
        shoulder_line = normalize_2d((right_shoulder - left_shoulder)[:2])

        time_s = relative_time_seconds(frame_id, onset_frame, frame_to_time)

        row = {
            "frame": frame_id,
            "time_s": time_s,
        }

        walk_refs = {
            "stable": stable_heading,
            "past_05s": walking_dir_past_seconds(kinematics, frame_id, 0.5, frame_to_time),
            "past_10s": walking_dir_past_seconds(kinematics, frame_id, 1.0, frame_to_time),
            "past_15s": walking_dir_past_seconds(kinematics, frame_id, 1.5, frame_to_time),
            "tangent": walking_dir_smoothed_tangent(kinematics, frame_id),
        }

        for ref_name, ref_dir in walk_refs.items():
            if shoulder_line is None or ref_dir is None:
                row[f"shoulder_{ref_name}"] = np.nan
            else:
                normal_1 = np.array([-shoulder_line[1], shoulder_line[0]])
                normal_2 = -normal_1

                shoulder_forward = (
                    normal_1
                    if np.dot(normal_1, ref_dir) >= np.dot(normal_2, ref_dir)
                    else normal_2
                )

                row[f"shoulder_{ref_name}"] = signed_deviation_deg(ref_dir, shoulder_forward)

        for axis_name, axis_vec in HEAD_AXES.items():
            head_vec_3d = head_rot @ axis_vec
            head_vec_2d = normalize_2d(head_vec_3d[:2])

            for ref_name, ref_dir in walk_refs.items():
                key = f"head_{axis_name}_{ref_name}"

                if head_vec_2d is None or ref_dir is None:
                    row[key] = np.nan
                else:
                    row[key] = signed_deviation_deg(ref_dir, head_vec_2d)

        rows.append(row)

    if not rows:
        return None

    times = np.array([r["time_s"] for r in rows], dtype=float)

    best_head_axis, best_head_axis_score = choose_best_head_axis_by_stability(
        rows=rows,
        times=times,
        baseline_start_s=baseline_start_s,
        baseline_end_s=baseline_end_s,
    )

    if best_head_axis is None:
        best_head_axis = "+Z"
        best_head_axis_score = np.nan

    event_df = np_to_event_dataframe(
        rows=rows,
        times=times,
        best_head_axis=best_head_axis,
        baseline_start_s=baseline_start_s,
        baseline_end_s=baseline_end_s,
    )

    av_frames, av_values = angular_velocity_frame_series_from_kinematics(kinematics)
    av_mask = (av_frames >= start_frame) & (av_frames <= end_frame)

    av_times = np.array([
        relative_time_seconds(int(f), onset_frame, frame_to_time)
        for f in av_frames[av_mask]
    ], dtype=float)

    av_values = av_values[av_mask]

    peak_time = None
    if peak_frame is not None:
        peak_time = relative_time_seconds(peak_frame, onset_frame, frame_to_time)

    return {
        "event_df": event_df,
        "best_head_axis": best_head_axis,
        "best_head_axis_score": best_head_axis_score,
        "peak_frame": peak_frame,
        "peak_time": peak_time,
        "av_times": av_times,
        "av_values": av_values,
        "stable_heading": stable_heading,
    }


def np_to_event_dataframe(rows, times, best_head_axis, baseline_start_s, baseline_end_s):
    data = {
        "frame": np.array([r["frame"] for r in rows], dtype=int),
        "time_seconds_relative_to_onset_from_timestamps": times,
    }

    for ref_name in ["stable", "past_05s", "past_10s", "past_15s", "tangent"]:
        head_values = np.array([r[f"head_{best_head_axis}_{ref_name}"] for r in rows], dtype=float)
        shoulder_values = np.array([r[f"shoulder_{ref_name}"] for r in rows], dtype=float)

        head_corrected, head_baseline = baseline_correct(
            head_values,
            times,
            baseline_start_s,
            baseline_end_s,
        )

        shoulder_corrected, shoulder_baseline = baseline_correct(
            shoulder_values,
            times,
            baseline_start_s,
            baseline_end_s,
        )

        data[f"head_{best_head_axis}_{ref_name}_baseline_corrected_deg"] = head_corrected
        data[f"head_{best_head_axis}_{ref_name}_baseline_deg"] = np.full_like(times, head_baseline, dtype=float)

        data[f"shoulder_{ref_name}_baseline_corrected_deg"] = shoulder_corrected
        data[f"shoulder_{ref_name}_baseline_deg"] = np.full_like(times, shoulder_baseline, dtype=float)

    import pandas as pd
    return pd.DataFrame(data)