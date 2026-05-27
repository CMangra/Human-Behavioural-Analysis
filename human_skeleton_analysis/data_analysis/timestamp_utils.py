from pathlib import Path
import numpy as np


FALLBACK_FPS = 10


def find_timestamp_file(dataset_dir, sequence):
    """
    Finds the PedX image timestamp file for a sequence.

    Handles possible extraction layouts:
        pedx_data/timestamps/timestamps-images-SEQ.txt
        pedx_data/timestamps/timestamps/timestamps-images-SEQ.txt
    """
    dataset_dir = Path(dataset_dir)

    candidates = [
        dataset_dir / "timestamps" / f"timestamps-images-{sequence}.txt",
        dataset_dir / "timestamps" / "timestamps" / f"timestamps-images-{sequence}.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_frame_timestamps(dataset_dir, sequence):
    """
    Loads PedX frame timestamps.

    Returns:
        dict: frame_id -> timestamp_seconds

    If no timestamp file is found or parsed, returns None.
    """
    timestamp_file = find_timestamp_file(dataset_dir, sequence)

    if timestamp_file is None:
        print("[TIMESTAMPS][WARN] No timestamp file found.")
        print(f"[TIMESTAMPS][WARN] Falling back to frame/FALLBACK_FPS = {FALLBACK_FPS}.")
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
        print(f"[TIMESTAMPS][WARN] Falling back to frame/FALLBACK_FPS = {FALLBACK_FPS}.")
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


def frame_time_seconds(frame_id, frame_to_time, fallback_fps=FALLBACK_FPS):
    """
    Absolute timestamp in seconds for a frame.
    Falls back to frame_id / fallback_fps if timestamps are unavailable.
    """
    frame_id = int(frame_id)

    if frame_to_time is not None and frame_id in frame_to_time:
        return frame_to_time[frame_id]

    return frame_id / fallback_fps


def relative_time_seconds(frame_id, onset_frame, frame_to_time, fallback_fps=FALLBACK_FPS):
    """
    Time relative to onset in seconds.
    """
    return (
        frame_time_seconds(frame_id, frame_to_time, fallback_fps)
        - frame_time_seconds(onset_frame, frame_to_time, fallback_fps)
    )