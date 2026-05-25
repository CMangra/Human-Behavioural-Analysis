import numpy as np
from scipy.signal import find_peaks
import config


def smooth_data(data, window):
    """Standard moving average with edge padding to maintain array lengths."""
    pad_size = window // 2
    padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
    return np.convolve(padded_data, np.ones(window) / window, mode='valid')


def compute_kinematics(frames_dict):
    """
    CENTRALIZED MATH ENGINE
    Calculates smoothed paths, angular velocities, peaks, and onsets.
    Returns a dictionary of synchronized arrays, or None if too short.
    """
    sorted_frames = sorted(frames_dict.keys())
    if len(sorted_frames) < 30: return None

    raw_xs = [frames_dict[f][0] for f in sorted_frames]
    raw_ys = [frames_dict[f][1] for f in sorted_frames]

    xs = smooth_data(raw_xs, config.XY_SMOOTHING_WINDOW)
    ys = smooth_data(raw_ys, config.XY_SMOOTHING_WINDOW)

    dx, dy = np.diff(xs), np.diff(ys)
    headings = np.arctan2(dy, dx)

    wrapped_diffs = (np.diff(headings) + np.pi) % (2 * np.pi) - np.pi
    ang_vel_deg = np.degrees(np.abs(wrapped_diffs))

    smoothed_ang_vel = smooth_data(ang_vel_deg, 5)

    peaks, _ = find_peaks(smoothed_ang_vel, prominence=config.TURN_PEAK_PROMINENCE, distance=15)

    onset_indices = []
    last_peak_idx = 0
    last_onset_idx = -999

    for peak_idx in peaks:
        start_idx = max(0, peak_idx - 15)
        end_idx = min(len(smoothed_ang_vel), peak_idx + 15)
        cumulative_turn = np.sum(smoothed_ang_vel[start_idx:end_idx])

        if cumulative_turn >= config.TURN_CUMULATIVE_DEG:
            # S-CURVE FIX: Find the lowest point (valley) between the last turn and this turn
            valley_idx = last_peak_idx + np.argmin(smoothed_ang_vel[last_peak_idx:peak_idx + 1])

            onset_idx = peak_idx
            # Backtrack until we hit the noise floor OR the valley
            while onset_idx > valley_idx and smoothed_ang_vel[onset_idx] > config.TURN_NOISE_FLOOR_DEG:
                onset_idx -= 1

            if (onset_idx - last_onset_idx) > config.TURN_DEBOUNCE_FRAMES:
                onset_indices.append(onset_idx)
                last_onset_idx = onset_idx

            last_peak_idx = peak_idx

    return {
        "sorted_frames": sorted_frames,
        "xs": xs,
        "ys": ys,
        "smoothed_ang_vel": smoothed_ang_vel,
        "peaks": peaks,
        "onset_indices": onset_indices
    }


def detect_multiple_turns_with_onset(trajectories):
    print("\n[DATA_ANALYSIS] Applying centralized kinematic math to qualified trajectories...")
    turn_results = {}
    stats = {"people_who_turn": 0, "total_turn_events": 0}

    for tid, frames_dict in trajectories.items():
        kinematics = compute_kinematics(frames_dict)
        if kinematics is None: continue

        # Map indices back to actual PedX frame IDs
        person_turn_onsets = []
        for idx in kinematics["onset_indices"]:
            if idx + 2 < len(kinematics["sorted_frames"]):
                person_turn_onsets.append(kinematics["sorted_frames"][idx + 2])

        if person_turn_onsets:
            turn_results[tid] = person_turn_onsets
            stats["people_who_turn"] += 1
            stats["total_turn_events"] += len(person_turn_onsets)

    print(
        f"[DATA_ANALYSIS] Found {stats['total_turn_events']} legitimate turn events across {stats['people_who_turn']} people.")
    return turn_results, stats
