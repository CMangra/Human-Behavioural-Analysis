import os
import json


def extract_skeleton_metrics(data_dir, sequence, tid, onset_frame, cameras):
    """Extracts pre-turn and post-turn skeletal metrics for a specific onset event."""
    start_f = onset_frame - 15  # 1.5 seconds before turn
    end_f = onset_frame + 5  # 0.5 seconds after turn begins

    cam_counts = {}
    cam_data = {c: {} for c in cameras}

    # 1. Find the best camera for this specific timeframe
    for cam in cameras:
        count = 0
        for f_id in range(start_f, end_f + 1):
            json_path = os.path.join(data_dir, 'labels', '2d', sequence, f"{sequence}_{cam}_{f_id:07d}_{tid}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    kp = data.get('keypoint') or {}
                    if 'lsho' in kp and 'rsho' in kp and 'nose' in kp:
                        cam_data[cam][f_id] = kp
                        count += 1
        cam_counts[cam] = count

    best_cam = max(cam_counts, key=cam_counts.get)
    if cam_counts[best_cam] < 5:
        return None  # Rejected: Not enough 2D skeleton data during this specific turn

    # 2. Calculate the metrics
    frames_seq, shoulder_widths, head_offsets = [], [], []

    for f_id in range(start_f, end_f + 1):
        if f_id in cam_data[best_cam]:
            kp = cam_data[best_cam][f_id]
            lx, rx = kp['lsho']['x'], kp['rsho']['x']
            nx = kp['nose']['x']

            width = abs(lx - rx)
            center_sho = (lx + rx) / 2.0
            max_reach = width / 2.0
            offset = (nx - center_sho) / max_reach if max_reach > 0 else 0

            frames_seq.append(f_id - onset_frame)  # Normalize so 0 is the Onset Frame!
            shoulder_widths.append(width)
            head_offsets.append(offset)

    return {"cam": best_cam, "frames_seq": frames_seq, "shoulder_widths": shoulder_widths, "head_offsets": head_offsets}