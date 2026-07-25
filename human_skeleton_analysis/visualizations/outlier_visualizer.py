import os
import cv2
import numpy as np
import torch
from pathlib import Path

from visualizations.smpl_video_annotator import load_camera_calibration, project_points, build_sampled_mesh_edges, \
    draw_wireframe, draw_vertices
from data_analysis.smpl_orientation_metrics import smpl_forward_joints, build_smpl_json_path
from visualizations.smpl_video_annotator import load_smpl_params

import config


def get_color(tid):
    np.random.seed(hash(tid) % (2 ** 32))
    return tuple(int(x) for x in np.random.randint(50, 255, 3))


def generate_outlier_collage(data_dir, sequence, tid, fail_frame, max_extension, mean_variance, smpl_model, faces,
                             device, output_dir):
    """
    Finds the specific frame where the biomechanical constraints failed,
    projects the SMPL mesh onto all 4 cameras, and creates a 2x2 collage.
    """
    print(f"     -> Generating outlier visual for {tid[-4:]} at frame {fail_frame}...")

    PANEL_W, PANEL_H = 640, 480
    panels = []

    json_path = build_smpl_json_path(data_dir, sequence, fail_frame, tid)
    if not os.path.exists(json_path):
        print(f"     -> [ERROR] Could not find SMPL json for outlier frame {fail_frame}.")
        return

    betas, pose, trans = load_smpl_params(json_path)

    # We need full vertices for the wireframe, not just joints
    betas_t = torch.tensor(betas.reshape(1, 10), dtype=torch.float32, device=device)
    global_orient_t = torch.tensor(pose[:3].reshape(1, 3), dtype=torch.float32, device=device)
    body_pose_t = torch.tensor(pose[3:].reshape(1, 69), dtype=torch.float32, device=device)
    transl_t = torch.tensor(trans.reshape(1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        output = smpl_model(betas=betas_t, global_orient=global_orient_t, body_pose=body_pose_t, transl=transl_t,
                            return_verts=True)
    vertices = output.vertices.detach().cpu().numpy()[0]

    sampled_edges = build_sampled_mesh_edges(faces, stride=4)
    color = get_color(tid)

    for cam in config.CAMERAS:
        img_path = os.path.join(data_dir, 'images', sequence, cam, f"{sequence}_{cam}_{fail_frame:07d}.jpg")
        img = cv2.imread(img_path)

        if img is None:
            panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            cv2.putText(panel, f"Missing: {cam}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            panels.append(panel)
            continue

        # Load calib and project
        P_rect, R_rect, T_range_to_cam = load_camera_calibration(Path(data_dir), sequence, cam)
        uv_vertices, valid_vertices = project_points(vertices, P_rect, R_rect, T_range_to_cam)

        # Draw the broken mesh
        overlay = img.copy()
        draw_wireframe(overlay, uv_vertices, valid_vertices, sampled_edges, color, thickness=1)
        draw_vertices(overlay, uv_vertices, valid_vertices, color, max_vertices=1000, radius=2)

        # Blend
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        # Add labels
        cv2.putText(img, cam, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        panels.append(cv2.resize(img, (PANEL_W, PANEL_H)))

    # Assemble 2x2 Collage
    top_row = np.hstack((panels[0], panels[1]))
    bottom_row = np.hstack((panels[2], panels[3]))
    collage = np.vstack((top_row, bottom_row))

    # Add the massive failure warning banner at the top
    banner_height = 80
    banner = np.zeros((banner_height, collage.shape[1], 3), dtype=np.uint8)
    banner[:] = (0, 0, 150)  # Dark Red background

    warning_text = f"OUTLIER DETECTED (ID: {tid[-4:]} | Frame: {fail_frame})  -->  Max Extension: {max_extension:.1f} DEG  |  Mean Variance: {mean_variance:.1f} DEG"
    cv2.putText(banner, warning_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    final_img = np.vstack((banner, collage))

    out_path = os.path.join(output_dir, f"outlier_proof_{tid[-4:]}_frame_{fail_frame}.jpg")
    cv2.imwrite(out_path, final_img)