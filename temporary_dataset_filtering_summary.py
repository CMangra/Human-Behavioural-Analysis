import os
import sys
import re
import glob
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# =============================================================================
# IMPORT EXISTING PROJECT CODE
# =============================================================================

ANALYSIS_ROOT = Path(__file__).resolve().parent / "human_skeleton_analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.append(str(ANALYSIS_ROOT))

from data_analysis.pedestrian_filtering import filter_pedestrians_by_visibility
from data_analysis.turn_detection import detect_multiple_turns_with_onset, compute_kinematics


# =============================================================================
# CONFIG
# =============================================================================

WORKSPACE_ROOT = Path(r"G:\My Drive\Desktop\THD\Master\JBData\3. Semester\code")
REPO_ROOT = WORKSPACE_ROOT / r"Third-Semester-Code\pedx"
DATASET_DIR = WORKSPACE_ROOT / r"downloaded_stuff\datasets\pedx\pedx_data"

SEQUENCES = [
    "20171130T2000",
    "20171207T2024",
]

OUTPUT_DIR = (
    REPO_ROOT
    / "visualisation_human_skeleton_visualisation_analysis"
    / "temp_dataset_filtering_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK_FPS = 10

PRE_SECONDS = 4.0
POST_SECONDS = 3.0
BASELINE_START_S = -3.0
BASELINE_END_S = -1.5

MIN_SMPL_FRAMES_IN_EVENT_WINDOW = 5
MIN_SMPL_FRAMES_IN_BASELINE = 3


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
        print(f"[TIMESTAMPS][WARN] {sequence}: no timestamp file found. Using fallback FPS={FALLBACK_FPS}.")
        return None

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

            if frame_id is not None and timestamp is not None:
                frame_to_time[frame_id] = timestamp

    if not frame_to_time:
        print(f"[TIMESTAMPS][WARN] {sequence}: timestamp file parsed but no usable timestamps found.")
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

    sorted_frames = sorted(frame_to_time.keys())
    sorted_times = np.array([frame_to_time[k] for k in sorted_frames], dtype=float)

    estimated_fps = np.nan
    if len(sorted_times) > 1:
        diffs = np.diff(sorted_times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs) > 0:
            estimated_fps = 1.0 / np.median(diffs)

    print(
        f"[TIMESTAMPS] {sequence}: loaded {len(frame_to_time)} timestamps "
        f"({unit}), estimated FPS={estimated_fps:.3f}"
    )

    return frame_to_time


def frame_time_seconds(frame_id, frame_to_time):
    if frame_to_time is not None and int(frame_id) in frame_to_time:
        return frame_to_time[int(frame_id)]

    return int(frame_id) / FALLBACK_FPS


def relative_time_seconds(frame_id, onset_frame, frame_to_time):
    return frame_time_seconds(frame_id, frame_to_time) - frame_time_seconds(onset_frame, frame_to_time)


# =============================================================================
# SMPL HELPERS
# =============================================================================

def parse_smpl_filename(path):
    stem = Path(path).stem
    parts = stem.split("_")

    frame_id = None
    for part in parts:
        if re.fullmatch(r"\d{7}", part):
            frame_id = int(part)
            break

    tid = parts[-1]

    if frame_id is None:
        return None

    return frame_id, tid


def collect_smpl_inventory(data_dir, sequence):
    smpl_dir = data_dir / "labels" / "3d" / "smpl" / sequence
    files = sorted(glob.glob(str(smpl_dir / "*.json")))

    by_tid = defaultdict(list)
    by_frame = defaultdict(list)

    for file_path in files:
        parsed = parse_smpl_filename(file_path)

        if parsed is None:
            continue

        frame_id, tid = parsed
        by_tid[tid].append(frame_id)
        by_frame[frame_id].append(tid)

    return {
        "sequence": sequence,
        "smpl_dir": str(smpl_dir),
        "smpl_dir_exists": smpl_dir.exists(),
        "total_smpl_json_files": len(files),
        "unique_smpl_pedestrians": len(by_tid),
        "unique_smpl_frames": len(by_frame),
        "smpl_tids": set(by_tid.keys()),
    }


def smpl_json_exists(data_dir, sequence, frame_id, tid):
    smpl_dir = data_dir / "labels" / "3d" / "smpl" / sequence

    exact = smpl_dir / f"{sequence}_{frame_id:07d}_{tid}.json"
    if exact.exists():
        return True

    matches = list(smpl_dir.glob(f"{sequence}_{frame_id:07d}_*{tid[-4:]}*.json"))
    return len(matches) > 0


# =============================================================================
# TURN HELPERS
# =============================================================================

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


def count_smpl_for_turn_event(data_dir, sequence, tid, onset_frame, kinematics, frame_to_time):
    peak_frame = peak_frame_after_onset(kinematics, onset_frame)

    peak_time = np.nan
    if peak_frame is not None:
        peak_time = relative_time_seconds(peak_frame, onset_frame, frame_to_time)

    event_frames = []
    baseline_frames = []
    onset_to_peak_frames = []

    for f in kinematics["sorted_frames"]:
        t = relative_time_seconds(f, onset_frame, frame_to_time)

        if -PRE_SECONDS <= t <= POST_SECONDS:
            event_frames.append(f)

        if BASELINE_START_S <= t <= BASELINE_END_S:
            baseline_frames.append(f)

        if peak_frame is not None and 0.0 <= t <= peak_time:
            onset_to_peak_frames.append(f)

    smpl_event_hits = [f for f in event_frames if smpl_json_exists(data_dir, sequence, f, tid)]
    smpl_baseline_hits = [f for f in baseline_frames if smpl_json_exists(data_dir, sequence, f, tid)]
    smpl_onset_to_peak_hits = [f for f in onset_to_peak_frames if smpl_json_exists(data_dir, sequence, f, tid)]

    enough_event_smpl = len(smpl_event_hits) >= MIN_SMPL_FRAMES_IN_EVENT_WINDOW
    enough_baseline_smpl = len(smpl_baseline_hits) >= MIN_SMPL_FRAMES_IN_BASELINE

    sufficient_for_orientation = bool(enough_event_smpl and enough_baseline_smpl)

    return {
        "peak_frame": peak_frame,
        "peak_time_seconds": peak_time,
        "event_window_frames": len(event_frames),
        "baseline_window_frames": len(baseline_frames),
        "onset_to_peak_frames": len(onset_to_peak_frames),
        "smpl_event_window_hits": len(smpl_event_hits),
        "smpl_baseline_hits": len(smpl_baseline_hits),
        "smpl_onset_to_peak_hits": len(smpl_onset_to_peak_hits),
        "smpl_any_in_event_window": len(smpl_event_hits) > 0,
        "smpl_sufficient_for_orientation_proxy": sufficient_for_orientation,
        "first_smpl_event_frame": min(smpl_event_hits) if smpl_event_hits else np.nan,
        "last_smpl_event_frame": max(smpl_event_hits) if smpl_event_hits else np.nan,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_sequence(sequence):
    print("\n" + "=" * 80)
    print(f"ANALYSING SEQUENCE: {sequence}")
    print("=" * 80)

    frame_to_time = load_frame_timestamps(DATASET_DIR, sequence)

    print("\n[1] Filtering pedestrians with existing pipeline logic...")
    qualified_trajectories, vis_stats = filter_pedestrians_by_visibility(
        str(DATASET_DIR),
        sequence,
    )

    print("\n[2] Detecting turn events using Step 2 logic...")
    turn_results, turn_stats = detect_multiple_turns_with_onset(
        qualified_trajectories
    )

    print("\n[3] Collecting SMPL annotation inventory...")
    smpl_inventory = collect_smpl_inventory(DATASET_DIR, sequence)

    qualified_tids = set(qualified_trajectories.keys())
    turning_tids = set(turn_results.keys())
    smpl_tids = smpl_inventory["smpl_tids"]

    qualified_with_smpl = qualified_tids.intersection(smpl_tids)
    turning_with_smpl = turning_tids.intersection(smpl_tids)

    print("\n[4] Checking SMPL availability for detected turn events...")
    event_rows = []

    for tid, onsets in turn_results.items():
        kinematics = compute_kinematics(qualified_trajectories[tid])

        if kinematics is None:
            continue

        for onset_frame in onsets:
            smpl_info = count_smpl_for_turn_event(
                data_dir=DATASET_DIR,
                sequence=sequence,
                tid=tid,
                onset_frame=onset_frame,
                kinematics=kinematics,
                frame_to_time=frame_to_time,
            )

            event_rows.append({
                "sequence": sequence,
                "tid": tid,
                "short_id": tid[-4:],
                "onset_frame": onset_frame,
                **smpl_info,
            })

    event_df = pd.DataFrame(event_rows)

    if event_df.empty:
        turn_events_with_any_smpl = 0
        turn_events_sufficient = 0
    else:
        turn_events_with_any_smpl = int(event_df["smpl_any_in_event_window"].sum())
        turn_events_sufficient = int(event_df["smpl_sufficient_for_orientation_proxy"].sum())

    summary = {
        "sequence": sequence,

        "total_lidar_pedestrians": vis_stats["total_lidar_pedestrians"],
        "qualified_pedestrians": vis_stats["qualified_pedestrians"],
        "rejected_pedestrians": vis_stats["rejected_pedestrians"],
        "people_who_turn": turn_stats["people_who_turn"],
        "total_turn_events": turn_stats["total_turn_events"],

        "smpl_dir_exists": smpl_inventory["smpl_dir_exists"],
        "total_smpl_json_files": smpl_inventory["total_smpl_json_files"],
        "unique_smpl_pedestrians": smpl_inventory["unique_smpl_pedestrians"],
        "unique_smpl_frames": smpl_inventory["unique_smpl_frames"],

        "qualified_pedestrians_with_smpl": len(qualified_with_smpl),
        "turning_pedestrians_with_smpl": len(turning_with_smpl),

        "turn_events_with_any_smpl_in_event_window": turn_events_with_any_smpl,
        "turn_events_sufficient_for_orientation_proxy": turn_events_sufficient,

        "event_window_seconds": f"[-{PRE_SECONDS}, +{POST_SECONDS}]",
        "baseline_window_seconds": f"[{BASELINE_START_S}, {BASELINE_END_S}]",
    }

    inventory_row = {
        "sequence": sequence,
        "smpl_dir": smpl_inventory["smpl_dir"],
        "smpl_dir_exists": smpl_inventory["smpl_dir_exists"],
        "total_smpl_json_files": smpl_inventory["total_smpl_json_files"],
        "unique_smpl_pedestrians": smpl_inventory["unique_smpl_pedestrians"],
        "unique_smpl_frames": smpl_inventory["unique_smpl_frames"],
        "qualified_pedestrians_with_smpl": len(qualified_with_smpl),
        "turning_pedestrians_with_smpl": len(turning_with_smpl),
    }

    return summary, event_rows, inventory_row


def main():
    print("=" * 80)
    print("TEMPORARY DATASET FILTERING / TURN / SMPL AVAILABILITY SUMMARY")
    print("=" * 80)
    print("DATASET_DIR:", DATASET_DIR)
    print("OUTPUT_DIR :", OUTPUT_DIR)
    print("SEQUENCES  :", SEQUENCES)

    all_summaries = []
    all_event_rows = []
    all_inventory_rows = []

    for sequence in SEQUENCES:
        summary, event_rows, inventory_row = analyze_sequence(sequence)

        all_summaries.append(summary)
        all_event_rows.extend(event_rows)
        all_inventory_rows.append(inventory_row)

    summary_df = pd.DataFrame(all_summaries)
    event_df = pd.DataFrame(all_event_rows)
    inventory_df = pd.DataFrame(all_inventory_rows)

    summary_csv = OUTPUT_DIR / "dataset_filtering_summary_by_sequence.csv"
    event_csv = OUTPUT_DIR / "turn_events_smpl_availability.csv"
    inventory_csv = OUTPUT_DIR / "smpl_annotation_inventory.csv"
    summary_json = OUTPUT_DIR / "dataset_filtering_summary.json"

    summary_df.to_csv(summary_csv, index=False)
    event_df.to_csv(event_csv, index=False)
    inventory_df.to_csv(inventory_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "=" * 80)
    print("[DONE] Dataset filtering / turn / SMPL availability summary complete.")
    print("Summary by sequence :", summary_csv)
    print("Turn-event details  :", event_csv)
    print("SMPL inventory      :", inventory_csv)
    print("JSON summary        :", summary_json)
    print("=" * 80)

    print("\n[SUMMARY TABLE]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()