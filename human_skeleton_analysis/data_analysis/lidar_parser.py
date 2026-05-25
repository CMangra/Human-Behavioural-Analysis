import os
import glob
import numpy as np
import trimesh
from collections import defaultdict


def extract_3d_trajectories(data_dir, sequence):
    """
    Scans segmented PLY files and builds 2D (X,Y) trajectories per pedestrian
    by calculating the centroid of their 3D point cloud.
    """
    lidar_dir = os.path.join(data_dir, 'labels', '3d', 'segment', sequence)
    ply_files = sorted(glob.glob(os.path.join(lidar_dir, '*.ply')))

    trajectories = defaultdict(dict)

    for ply_path in ply_files:
        basename = os.path.basename(ply_path)
        parts = basename.split('_')

        # Filename format: 20171207T2024_0000055_2d5b.ply
        frame_id = int(parts[-2])
        tid = parts[-1].split('.')[0]

        try:
            # Load point cloud without processing the strict PLY headers
            pcd = trimesh.load(ply_path, process=False)
            pts = np.array(pcd.vertices)

            # Ignore noise/ghosts with less than 5 points
            if len(pts) > 5:
                # We only care about X and Y for ground trajectories (ignore Z height)
                centroid_x = np.mean(pts[:, 0])
                centroid_y = np.mean(pts[:, 1])
                trajectories[tid][frame_id] = (centroid_x, centroid_y)
        except Exception:
            pass

    return trajectories