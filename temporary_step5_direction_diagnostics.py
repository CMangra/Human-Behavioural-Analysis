import os
import sys
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# =============================================================================
# IMPORT EXISTING PROJECT CODE
# =============================================================================

ANALYSIS_ROOT = Path(__file__).resolve().parent / "human_skeleton_analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.append(str(ANALYSIS_ROOT))

from data_analysis.pedestrian_filtering import filter_pedestrians_by_visibility
from data_analysis.turn_detection import detect_multiple_turns_with_onset, compute_kinematics
from visualizations.smpl_video_annotator import load_smpl_model, load_smpl_params
# =============================================================================
# CONFIG
# =============================================================================

WORKSPACE_ROOT = Path(r"G:\My Drive\Desktop\THD\Master\JBData\3. Semester\code")
REPO_ROOT = WORKSPACE_ROOT / r"Third-Semester-Code\pedx"
DATASET_DIR = WORKSPACE_ROOT / r"downloaded_stuff\datasets\pedx\pedx_data"

SEQUENCE = "20171207T2024"

FALLBACK_FPS = 10

PRE_SECONDS = 4.0
POST_SECONDS = 3.0

BASELINE_START_S = -3.0
BASELINE_END_S = -1.5

OUTPUT_DIR = (
    REPO_ROOT
    / "visualisation_human_skeleton_visualisation_analysis"
    / "temp_step5_diagnostics_separate_figures"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ROOT = REPO_ROOT / "body_models"
SMPL_DIR = DATASET_DIR / "labels" / "3d" / "smpl" / SEQUENCE


# =============================================================================
# TIMESTAMP HELPERS
# =============================================================================

def find_timestamp_file(dataset_dir, sequence):
    candidates = [
        dataset_dir / "timestamps" / f"timestamps-images-{sequence}.txt",
        dataset_dir / "timestamps" / "timestamps" / f"timestamps-images-{sequence}.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_frame_timestamps(dataset_dir, sequence):
    timestamp_file = find_timestamp_file(dataset_dir, sequence)

    if timestamp_file is None:
        print("[TIMESTAMPS][WARN] No timestamp file found.")
        print("[TIMESTAMPS][WARN] Falling back to frame/FALLBACK_FPS.")
        return None

    print("[TIMESTAMPS] Using:", timestamp_file)

    frame_to_time = {}

    with open(timestamp_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.replace(",", " ").split()

            frame_id = None

            for token in parts:
                token_clean = (
                    token
                    .replace(".jpg", "")
                    .replace(".png", "")
                    .replace(".ply", "")
                    .replace(".txt", "")
                )

                for sp in token_clean.split("_"):
                    if sp.isdigit() and len(sp) <= 7:
                        try:
                            frame_id = int(sp)
                        except ValueError:
                            pass

            timestamp = None
            for token in reversed(parts):
                try:
                    timestamp = float(token)
                    break
                except ValueError:
                    continue

            if frame_id is None or timestamp is None:
                continue

            frame_to_time[frame_id] = timestamp

    if not frame_to_time:
        print("[TIMESTAMPS][WARN] Timestamp file parsed but no usable frame timestamps found.")
        print("[TIMESTAMPS][WARN] Falling back to frame/FALLBACK_FPS.")
        return None

    values = np.array(list(frame_to_time.values()), dtype=float)
    sorted_values = np.sort(values)

    if len(sorted_values) > 1:
        median_step = float(np.median(np.diff(sorted_values)))
    else:
        median_step = 0.1

    if median_step > 1e6:
        frame_to_time = {k: v / 1e9 for k, v in frame_to_time.items()}
        unit = "nanoseconds -> seconds"
    elif median_step > 1e3:
        frame_to_time = {k: v / 1e6 for k, v in frame_to_time.items()}
        unit = "microseconds -> seconds"
    elif median_step > 1:
        frame_to_time = {k: v / 1e3 for k, v in frame_to_time.items()}
        unit = "milliseconds -> seconds"
    else:
        unit = "seconds"

    print(f"[TIMESTAMPS] Loaded {len(frame_to_time)} frame timestamps ({unit}).")

    sorted_frames = sorted(frame_to_time.keys())
    sorted_times = np.array([frame_to_time[k] for k in sorted_frames], dtype=float)

    if len(sorted_times) > 1:
        diffs = np.diff(sorted_times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]

        if len(diffs) > 0:
            estimated_fps = 1.0 / np.median(diffs)
            print(f"[TIMESTAMPS] Estimated FPS from timestamps: {estimated_fps:.3f}")

    return frame_to_time


def frame_time_seconds(frame_id, frame_to_time):
    if frame_to_time is not None and int(frame_id) in frame_to_time:
        return frame_to_time[int(frame_id)]

    return int(frame_id) / FALLBACK_FPS


def relative_time_seconds(frame_id, onset_frame, frame_to_time):
    return frame_time_seconds(frame_id, frame_to_time) - frame_time_seconds(onset_frame, frame_to_time)


# =============================================================================
# SMPL CONSTANTS
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


# =============================================================================
# BASIC GEOMETRY
# =============================================================================

def normalize_2d(v, eps=1e-9):
    v = np.asarray(v, dtype=float).reshape(2)
    n = np.linalg.norm(v)

    if n < eps or not np.isfinite(n):
        return None

    return v / n


def signed_deviation_deg(reference_dir, target_dir):
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


# =============================================================================
# DATA / FRAME HELPERS
# =============================================================================

def find_smpl_json(frame_id, tid):
    exact = SMPL_DIR / f"{SEQUENCE}_{frame_id:07d}_{tid}.json"

    if exact.exists():
        return exact

    matches = sorted(glob.glob(str(SMPL_DIR / f"{SEQUENCE}_{frame_id:07d}_*{tid[-4:]}*.json")))

    if matches:
        return Path(matches[0])

    return None


def get_kinematic_index(kinematics, frame_id):
    sorted_frames = kinematics["sorted_frames"]

    if frame_id not in sorted_frames:
        return None

    return sorted_frames.index(frame_id)


def walking_dir_past_seconds(kinematics, frame_id, past_seconds, frame_to_time):
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

    if past_idx < 0 or past_idx >= len(sorted_frames):
        return None

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


def stable_pre_turn_heading(kinematics, onset_frame, start_s, end_s, frame_to_time):
    sorted_frames = np.asarray(kinematics["sorted_frames"], dtype=int)
    xs = np.asarray(kinematics["xs"], dtype=float)
    ys = np.asarray(kinematics["ys"], dtype=float)

    times = np.array([
        relative_time_seconds(int(f), onset_frame, frame_to_time)
        for f in sorted_frames
    ], dtype=float)

    start_mask = (times >= start_s) & (times <= start_s + 0.5)
    end_mask = (times >= end_s - 0.5) & (times <= end_s)

    if np.sum(start_mask) < 2 or np.sum(end_mask) < 2:
        return None

    p_start = np.array([np.nanmean(xs[start_mask]), np.nanmean(ys[start_mask])])
    p_end = np.array([np.nanmean(xs[end_mask]), np.nanmean(ys[end_mask])])

    return normalize_2d(p_end - p_start)


def angular_velocity_series_from_kinematics(kinematics):
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

    for p in peaks:
        idx = int(p) + 1

        if 0 <= idx < len(sorted_frames):
            peak_frames.append(sorted_frames[idx])

    after = [f for f in peak_frames if f >= onset_frame]

    if after:
        return min(after)

    if peak_frames:
        return min(peak_frames, key=lambda f: abs(f - onset_frame))

    return None


def baseline_correct(values, times, start_s=BASELINE_START_S, end_s=BASELINE_END_S):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)

    mask = (
        np.isfinite(values)
        & (times >= start_s)
        & (times <= end_s)
    )

    if np.sum(mask) < 3:
        return values, np.nan

    baseline = np.nanmean(values[mask])
    return values - baseline, baseline


def choose_best_head_axis_by_stability(rows, times):
    best_axis = None
    best_score = np.inf

    baseline_mask = (times >= BASELINE_START_S) & (times <= BASELINE_END_S)

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


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def save_figure(fig, base_path):
    png_path = Path(str(base_path) + ".png")
    svg_path = Path(str(base_path) + ".svg")

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")

    plt.close(fig)

    return png_path, svg_path


def plot_bev_trajectory(
    tid,
    short_id,
    onset_frame,
    peak_frame,
    kinematics,
    stable_heading,
    person_output_dir,
):
    xs = np.asarray(kinematics["xs"], dtype=float)
    ys = np.asarray(kinematics["ys"], dtype=float)
    sorted_frames = np.asarray(kinematics["sorted_frames"], dtype=int)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(xs, ys, color="gray", linewidth=2, label="Smoothed BEV trajectory")

    if onset_frame in sorted_frames:
        idx = list(sorted_frames).index(onset_frame)
        ax.scatter(xs[idx], ys[idx], color="green", s=100, label="onset")

        arrow_scale = 3.0
        ax.arrow(
            xs[idx],
            ys[idx],
            stable_heading[0] * arrow_scale,
            stable_heading[1] * arrow_scale,
            head_width=0.4,
            color="blue",
            length_includes_head=True,
            label="stable pre-turn heading",
        )

    if peak_frame in sorted_frames:
        idx = list(sorted_frames).index(peak_frame)
        ax.scatter(xs[idx], ys[idx], color="red", marker="x", s=120, label="peak")

    ax.set_aspect("equal", "datalim")
    ax.set_title(f"BEV trajectory tracking with stable pre-turn heading\nPedestrian {tid}")
    ax.set_xlabel("X position in PedX 3D coordinate system")
    ax.set_ylabel("Y position in PedX 3D coordinate system")
    ax.legend(loc="upper left")
    ax.grid(True)

    base_path = person_output_dir / f"fig_a_bev_trajectory_{short_id}_onset_{onset_frame}"
    return save_figure(fig, base_path)


def plot_angular_velocity(
    tid,
    short_id,
    onset_frame,
    av_time,
    av_window,
    peak_time,
    person_output_dir,
):
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(av_time, av_window, color="purple", linewidth=2, label="Angular velocity")
    ax.axvline(0, color="green", linestyle="--", linewidth=2, label="onset")

    if peak_time is not None:
        ax.axvline(peak_time, color="red", linestyle="--", linewidth=2, label="peak")

    ax.set_title(f"Angular velocity\nPedestrian {tid}")
    ax.set_xlabel("Time relative to onset [s]")
    ax.set_ylabel("Angular velocity [deg/frame]")
    ax.legend(loc="upper left")
    ax.grid(True)

    base_path = person_output_dir / f"fig_b_angular_velocity_{short_id}_onset_{onset_frame}"
    return save_figure(fig, base_path)


def plot_head_sensitivity(
    tid,
    short_id,
    onset_frame,
    peak_time,
    times,
    head_sensitivity,
    person_output_dir,
):
    fig, ax = plt.subplots(figsize=(12, 4))

    for ref_name in ["stable", "past_05s", "past_10s", "past_15s", "tangent"]:
        ax.plot(
            times,
            head_sensitivity[ref_name],
            marker="o",
            linewidth=1.8,
            label=ref_name,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="green", linestyle="--", linewidth=2, label="onset")

    if peak_time is not None:
        ax.axvline(peak_time, color="red", linestyle="--", linewidth=2, label="peak")

    ax.set_title(f"Head orientation sensitivity after baseline correction\nPedestrian {tid}")
    ax.set_xlabel("Time relative to onset [s]")
    ax.set_ylabel("Head deviation change [deg]")
    ax.legend(loc="upper left")
    ax.grid(True)

    base_path = person_output_dir / f"fig_c_head_orientation_{short_id}_onset_{onset_frame}"
    return save_figure(fig, base_path)


def plot_shoulder_sensitivity(
    tid,
    short_id,
    onset_frame,
    peak_time,
    times,
    shoulder_sensitivity,
    person_output_dir,
):
    fig, ax = plt.subplots(figsize=(12, 4))

    for ref_name in ["stable", "past_05s", "past_10s", "past_15s", "tangent"]:
        ax.plot(
            times,
            shoulder_sensitivity[ref_name],
            marker="s",
            linewidth=1.8,
            label=ref_name,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="green", linestyle="--", linewidth=2, label="onset")

    if peak_time is not None:
        ax.axvline(peak_time, color="red", linestyle="--", linewidth=2, label="peak")

    ax.set_title(f"Shoulder / torso orientation sensitivity after baseline correction\nPedestrian {tid}")
    ax.set_xlabel("Time relative to onset [s]")
    ax.set_ylabel("Shoulder deviation change [deg]")
    ax.legend(loc="upper left")
    ax.grid(True)

    base_path = person_output_dir / f"fig_d_shoulder_orientation_{short_id}_onset_{onset_frame}"
    return save_figure(fig, base_path)


# =============================================================================
# EVENT PROCESSING
# =============================================================================

def process_one_event(tid, onset_frame, kinematics, model, device, frame_to_time):
    short_id = tid[-4:]

    person_output_dir = OUTPUT_DIR / f"pedestrian_{short_id}"
    person_output_dir.mkdir(parents=True, exist_ok=True)

    peak_frame = peak_frame_after_onset(kinematics, onset_frame)

    stable_heading = stable_pre_turn_heading(
        kinematics,
        onset_frame,
        BASELINE_START_S,
        BASELINE_END_S,
        frame_to_time,
    )

    if stable_heading is None:
        print(f"[SKIP] {short_id} onset {onset_frame}: could not compute stable heading.")
        return None

    frames = [
        f for f in kinematics["sorted_frames"]
        if -PRE_SECONDS <= relative_time_seconds(f, onset_frame, frame_to_time) <= POST_SECONDS
    ]

    if not frames:
        print(f"[SKIP] {short_id} onset {onset_frame}: no frames in timestamp window.")
        return None

    start_frame = min(frames)
    end_frame = max(frames)

    rows = []

    for frame_id in frames:
        json_path = find_smpl_json(frame_id, tid)

        if json_path is None:
            continue

        betas, pose, trans = load_smpl_params(json_path)
        joints = smpl_forward_joints(model, betas, pose, trans, device)

        global_rots = compute_global_rotations_from_pose(pose)
        head_rot = global_rots[SMPL_HEAD_INDEX]

        left_shoulder = joints[SMPL_LEFT_SHOULDER_INDEX]
        right_shoulder = joints[SMPL_RIGHT_SHOULDER_INDEX]
        shoulder_line = normalize_2d((right_shoulder - left_shoulder)[:2])

        row = {
            "frame": frame_id,
            "time": relative_time_seconds(frame_id, onset_frame, frame_to_time),
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
                continue

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
        print(f"[SKIP] {short_id} onset {onset_frame}: no SMPL rows generated.")
        return None

    times = np.array([r["time"] for r in rows], dtype=float)
    frame_arr = np.array([r["frame"] for r in rows], dtype=int)

    best_head_axis, best_head_axis_score = choose_best_head_axis_by_stability(rows, times)

    if best_head_axis is None:
        best_head_axis = "+Z"
        best_head_axis_score = np.nan

    head_sensitivity = {}
    head_sensitivity_baselines = {}

    for ref_name in ["stable", "past_05s", "past_10s", "past_15s", "tangent"]:
        values = np.array([r[f"head_{best_head_axis}_{ref_name}"] for r in rows], dtype=float)
        corrected, baseline = baseline_correct(values, times)
        head_sensitivity[ref_name] = corrected
        head_sensitivity_baselines[ref_name] = baseline

    shoulder_sensitivity = {}
    shoulder_sensitivity_baselines = {}

    for ref_name in ["stable", "past_05s", "past_10s", "past_15s", "tangent"]:
        values = np.array([r[f"shoulder_{ref_name}"] for r in rows], dtype=float)
        corrected, baseline = baseline_correct(values, times)
        shoulder_sensitivity[ref_name] = corrected
        shoulder_sensitivity_baselines[ref_name] = baseline

    av_frames, av_values = angular_velocity_series_from_kinematics(kinematics)
    av_mask = (av_frames >= start_frame) & (av_frames <= end_frame)

    av_time = np.array([
        relative_time_seconds(int(f), onset_frame, frame_to_time)
        for f in av_frames[av_mask]
    ], dtype=float)

    av_window = av_values[av_mask]

    peak_time = None
    if peak_frame is not None:
        peak_time = relative_time_seconds(peak_frame, onset_frame, frame_to_time)

    event_df = pd.DataFrame({
        "frame": frame_arr,
        "time_seconds_relative_to_onset_from_timestamps": times,
    })

    for ref_name, values in head_sensitivity.items():
        event_df[f"head_{best_head_axis}_{ref_name}_baseline_corrected_deg"] = values

    for ref_name, values in shoulder_sensitivity.items():
        event_df[f"shoulder_{ref_name}_baseline_corrected_deg"] = values

    out_csv = person_output_dir / f"separate_figures_data_{short_id}_onset_{onset_frame}.csv"
    event_df.to_csv(out_csv, index=False)

    bev_png, bev_svg = plot_bev_trajectory(
        tid=tid,
        short_id=short_id,
        onset_frame=onset_frame,
        peak_frame=peak_frame,
        kinematics=kinematics,
        stable_heading=stable_heading,
        person_output_dir=person_output_dir,
    )

    av_png, av_svg = plot_angular_velocity(
        tid=tid,
        short_id=short_id,
        onset_frame=onset_frame,
        av_time=av_time,
        av_window=av_window,
        peak_time=peak_time,
        person_output_dir=person_output_dir,
    )

    head_png, head_svg = plot_head_sensitivity(
        tid=tid,
        short_id=short_id,
        onset_frame=onset_frame,
        peak_time=peak_time,
        times=times,
        head_sensitivity=head_sensitivity,
        person_output_dir=person_output_dir,
    )

    shoulder_png, shoulder_svg = plot_shoulder_sensitivity(
        tid=tid,
        short_id=short_id,
        onset_frame=onset_frame,
        peak_time=peak_time,
        times=times,
        shoulder_sensitivity=shoulder_sensitivity,
        person_output_dir=person_output_dir,
    )

    def window_mean(values, start_s, end_s):
        values = np.asarray(values, dtype=float)
        mask = (
            (times >= start_s)
            & (times <= end_s)
            & np.isfinite(values)
        )

        if np.sum(mask) == 0:
            return np.nan

        return float(np.nanmean(values[mask]))

    summary = {
        "tid": tid,
        "short_id": short_id,
        "onset_frame": onset_frame,
        "peak_frame": peak_frame,
        "peak_time_seconds_from_timestamps": peak_time,
        "best_head_axis": best_head_axis,
        "best_head_axis_baseline_std": best_head_axis_score,
        "head_stable_mean_pre_1s_to_0s": window_mean(head_sensitivity["stable"], -1.0, 0.0),
        "head_stable_mean_0s_to_peak": window_mean(
            head_sensitivity["stable"],
            0.0,
            peak_time if peak_time is not None else POST_SECONDS,
        ),
        "shoulder_stable_mean_pre_1s_to_0s": window_mean(shoulder_sensitivity["stable"], -1.0, 0.0),
        "shoulder_stable_mean_0s_to_peak": window_mean(
            shoulder_sensitivity["stable"],
            0.0,
            peak_time if peak_time is not None else POST_SECONDS,
        ),
        "head_tangent_mean_pre_1s_to_0s": window_mean(head_sensitivity["tangent"], -1.0, 0.0),
        "shoulder_tangent_mean_pre_1s_to_0s": window_mean(shoulder_sensitivity["tangent"], -1.0, 0.0),
        "person_output_dir": str(person_output_dir),
        "event_csv": str(out_csv),
        "bev_png": str(bev_png),
        "bev_svg": str(bev_svg),
        "angular_velocity_png": str(av_png),
        "angular_velocity_svg": str(av_svg),
        "head_png": str(head_png),
        "head_svg": str(head_svg),
        "shoulder_png": str(shoulder_png),
        "shoulder_svg": str(shoulder_svg),
    }

    print(f"[OK] Generated separate figures for {short_id} onset {onset_frame}")
    print(f"     folder   : {person_output_dir}")
    print(f"     BEV      : {bev_png}")
    print(f"     AV       : {av_png}")
    print(f"     Head     : {head_png}")
    print(f"     Shoulder : {shoulder_png}")
    print(f"     CSV      : {out_csv}")

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Timestamp Temporary Step 5 Diagnostic: Separate Figures")
    print("=" * 80)
    print("DATASET_DIR:", DATASET_DIR)
    print("SEQUENCE   :", SEQUENCE)
    print("OUTPUT_DIR :", OUTPUT_DIR)

    print("\n[0] Loading PedX timestamps...")
    frame_to_time = load_frame_timestamps(DATASET_DIR, SEQUENCE)

    if frame_to_time is None:
        print(f"[TIMESTAMPS][WARN] Using fallback timing: frame_difference / {FALLBACK_FPS}")
    else:
        print("[TIMESTAMPS] Real timestamp timing enabled.")

    print("\n[1] Reusing Step 2: filtering pedestrians...")
    qualified_trajectories, vis_stats = filter_pedestrians_by_visibility(
        str(DATASET_DIR),
        SEQUENCE,
    )

    print("\n[2] Reusing Step 2: detecting turns...")
    turn_results, turn_stats = detect_multiple_turns_with_onset(
        qualified_trajectories
    )

    print("\n[STEP 2 SUMMARY]")
    print("Qualified pedestrians:", vis_stats.get("qualified_pedestrians"))
    print("People who turn      :", turn_stats.get("people_who_turn"))
    print("Total turn events    :", turn_stats.get("total_turn_events"))

    print("\n[3] Loading SMPL model once...")
    device = torch.device("cpu")
    model, _ = load_smpl_model(MODEL_ROOT, device)

    print("\n[4] Processing all turn events...")
    summaries = []

    for tid, onsets in turn_results.items():
        short_id = tid[-4:]

        if tid not in qualified_trajectories:
            print(f"[SKIP] {short_id}: ID exists in turn_results but not in qualified_trajectories.")
            continue

        kinematics = compute_kinematics(qualified_trajectories[tid])

        if kinematics is None:
            print(f"[SKIP] {short_id}: no kinematics.")
            continue

        for onset_frame in onsets:
            print("\n" + "-" * 80)
            print(f"Event: ID {short_id} | onset {onset_frame}")

            summary = process_one_event(
                tid=tid,
                onset_frame=onset_frame,
                kinematics=kinematics,
                model=model,
                device=device,
                frame_to_time=frame_to_time,
            )

            if summary is not None:
                summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_csv = OUTPUT_DIR / "all_turn_events_separate_figures_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("[DONE] Separate-figure timestamp diagnostics complete.")
    print("Events processed:", len(summaries))
    print("Summary CSV     :", summary_csv)
    print("Output folder   :", OUTPUT_DIR)

    if len(summaries) > 0:
        print("\n[QUICK SUMMARY]")
        print(summary_df[
            [
                "short_id",
                "onset_frame",
                "peak_frame",
                "peak_time_seconds_from_timestamps",
                "best_head_axis",
                "head_stable_mean_0s_to_peak",
                "shoulder_stable_mean_0s_to_peak",
                "person_output_dir",
                "bev_png",
                "angular_velocity_png",
                "head_png",
                "shoulder_png",
            ]
        ])

    print("=" * 80)


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# =============================================================================
# IMPORT EXISTING PROJECT CODE
# =============================================================================

ANALYSIS_ROOT = Path(__file__).resolve().parent / "human_skeleton_analysis"
