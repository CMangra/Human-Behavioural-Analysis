import os
import glob
import numpy as np
import trimesh
from collections import defaultdict


def extract_3d_trajectories(data_dir, sequence):
    """
    Scans segmented PLY files and builds 2D (X,Y) trajectories per pedestrian
    by calculating the centroid of their 3D point cloud.
    Also extracts the Z-axis limits to establish the LiDAR ground plane.
    """
    lidar_dir = os.path.join(data_dir, 'labels', '3d', 'segment', sequence)
    ply_files = sorted(glob.glob(os.path.join(lidar_dir, '*.ply')))

    trajectories = defaultdict(dict)

    for ply_path in ply_files:
        basename = os.path.basename(ply_path)
        parts = basename.split('_')

        frame_id = int(parts[-2])
        tid = parts[-1].split('.')[0]

        try:
            pcd = trimesh.load(ply_path, process=False)
            pts = np.array(pcd.vertices)

            if len(pts) > 5:
                centroid_x = np.mean(pts[:, 0])
                centroid_y = np.mean(pts[:, 1])

                # Extract Z bounds. Assuming +Z points downwards, max_z is the floor.
                min_z = np.min(pts[:, 2])
                max_z = np.max(pts[:, 2])

                trajectories[tid][frame_id] = (centroid_x, centroid_y, min_z, max_z)
        except Exception:
            pass

    return trajectories