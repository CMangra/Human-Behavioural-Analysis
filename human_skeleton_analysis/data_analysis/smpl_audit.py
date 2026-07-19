import os
import glob
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from data_analysis.timestamp_utils import relative_time_seconds


def parse_smpl_filename(path):
    stem = Path(path).stem
    parts = stem.split("_")
    frame_id = None
    for part in parts:
        if re.fullmatch(r"\d{7}", part):
            frame_id = int(part)
            break
    tid = parts[-1]
    return frame_id, tid


def collect_smpl_inventory(data_dir, sequence):
    smpl_dir = Path(data_dir) / "labels" / "3d" / "smpl" / sequence
    files = sorted(glob.glob(str(smpl_dir / "*.json")))
    by_tid = defaultdict(list)
    by_frame = defaultdict(list)

    for file_path in files:
        parsed = parse_smpl_filename(file_path)
        if parsed is None: continue
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
    smpl_dir = Path(data_dir) / "labels" / "3d" / "smpl" / sequence
    exact = smpl_dir / f"{sequence}_{frame_id:07d}_{tid}.json"
    if exact.exists(): return True
    matches = list(smpl_dir.glob(f"{sequence}_{frame_id:07d}_*{tid[-4:]}*.json"))
    return len(matches) > 0


def count_smpl_for_turn_event(data_dir, sequence, tid, onset_frame, kinematics, frame_to_time, pre_s, post_s,
                              base_start_s, base_end_s, min_event_frames=5, min_base_frames=3):
    from data_analysis.turn_detection import compute_kinematics
    # Re-use peak logic from orientation metrics if needed, or simplified here
    peaks = kinematics["peaks"]
    peak_frame = None
    if peaks is not None and len(peaks) > 0:
        peak_frames = [kinematics["sorted_frames"][int(p) + 1] for p in peaks if
                       0 <= int(p) + 1 < len(kinematics["sorted_frames"])]
        after = [f for f in peak_frames if f >= onset_frame]
        peak_frame = min(after) if after else min(peak_frames, key=lambda f: abs(f - onset_frame))

    peak_time = relative_time_seconds(peak_frame, onset_frame, frame_to_time) if peak_frame else np.nan

    event_frames, baseline_frames, onset_to_peak_frames = [], [], []
    for f in kinematics["sorted_frames"]:
        t = relative_time_seconds(f, onset_frame, frame_to_time)
        if -pre_s <= t <= post_s: event_frames.append(f)
        if base_start_s <= t <= base_end_s: baseline_frames.append(f)
        if peak_frame is not None and 0.0 <= t <= peak_time: onset_to_peak_frames.append(f)

    smpl_event_hits = [f for f in event_frames if smpl_json_exists(data_dir, sequence, f, tid)]
    smpl_baseline_hits = [f for f in baseline_frames if smpl_json_exists(data_dir, sequence, f, tid)]
    smpl_onset_to_peak_hits = [f for f in onset_to_peak_frames if smpl_json_exists(data_dir, sequence, f, tid)]

    sufficient_for_orientation = bool(
        len(smpl_event_hits) >= min_event_frames and len(smpl_baseline_hits) >= min_base_frames)

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


def run_dataset_audit(data_dir, sequence, qualified_trajectories, turn_results, frame_to_time, output_dir, pre_s=4.0,
                      post_s=3.0, base_start_s=-3.0, base_end_s=-1.5):
    from data_analysis.turn_detection import compute_kinematics

    os.makedirs(output_dir, exist_ok=True)
    smpl_inventory = collect_smpl_inventory(data_dir, sequence)

    event_rows = []
    for tid, onsets in turn_results.items():
        kinematics = compute_kinematics(qualified_trajectories[tid])
        if kinematics is None: continue
        for onset_frame in onsets:
            smpl_info = count_smpl_for_turn_event(
                data_dir, sequence, tid, onset_frame, kinematics, frame_to_time, pre_s, post_s, base_start_s, base_end_s
            )
            event_rows.append(
                {"sequence": sequence, "tid": tid, "short_id": tid[-4:], "onset_frame": onset_frame, **smpl_info})

    event_df = pd.DataFrame(event_rows)
    inventory_df = pd.DataFrame([smpl_inventory])

    # Export audit files
    event_df.to_csv(os.path.join(output_dir, f"turn_events_smpl_availability_{sequence}.csv"), index=False)
    inventory_df.drop(columns=["smpl_tids"]).to_csv(
        os.path.join(output_dir, f"smpl_annotation_inventory_{sequence}.csv"), index=False)

    # Filter for usable turns
    usable_turn_results = defaultdict(list)
    if not event_df.empty:
        valid_events = event_df[event_df["smpl_sufficient_for_orientation_proxy"] == True]
        for _, row in valid_events.iterrows():
            usable_turn_results[row["tid"]].append(row["onset_frame"])

    return dict(usable_turn_results), event_df