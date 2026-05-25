import os
import glob
import json
from collections import defaultdict
import config
from .lidar_parser import extract_3d_trajectories


def filter_pedestrians_by_visibility(data_dir, sequence):
    """
    Evaluates all LiDAR-tracked pedestrians and checks their 2D camera visibility.
    Returns tracking dictionaries and statistics.
    """
    print("\n[DATA_ANALYSIS] Extracting LiDAR trajectories to determine lifespans...")
    trajectories = extract_3d_trajectories(data_dir, sequence)

    stats = {
        "total_lidar_pedestrians": len(trajectories),
        "qualified_pedestrians": 0,
        "rejected_pedestrians": 0,
        "cam_visibility_counts": {1: 0, 2: 0, 3: 0, 4: 0}  # How many people are seen by X cameras
    }

    qualified_trajectories = {}

    for tid, frames_dict in trajectories.items():
        sorted_frames = sorted(frames_dict.keys())
        start_f, end_f = sorted_frames[0], sorted_frames[-1]
        lifespan = end_f - start_f + 1

        # Check 2D JSONs across all cameras
        visible_frames = set()
        cams_seeing_person = set()

        for cam in config.CAMERAS:
            saw_in_this_cam = False
            for f_id in range(start_f, end_f + 1):
                json_path = os.path.join(data_dir, 'labels', '2d', sequence, f"{sequence}_{cam}_{f_id:07d}_{tid}.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    kp = data.get('keypoint') or {}
                    if 'nose' in kp or 'lsho' in kp:
                        visible_frames.add(f_id)
                        saw_in_this_cam = True
            if saw_in_this_cam:
                cams_seeing_person.add(cam)

        if len(cams_seeing_person) > 0:
            stats["cam_visibility_counts"][len(cams_seeing_person)] += 1

        visibility_ratio = len(visible_frames) / lifespan if lifespan > 0 else 0

        if visibility_ratio >= config.MIN_VISIBILITY_RATIO:
            qualified_trajectories[tid] = frames_dict
            stats["qualified_pedestrians"] += 1
        else:
            stats["rejected_pedestrians"] += 1

    print(
        f"[DATA_ANALYSIS] Filtering Complete: {stats['qualified_pedestrians']} Qualified, {stats['rejected_pedestrians']} Rejected.")
    return qualified_trajectories, stats