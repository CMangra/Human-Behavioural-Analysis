import os
import cv2
import numpy as np
import torch
from pathlib import Path

from visualizations.smpl_video_annotator import load_camera_calibration, project_points, build_sampled_mesh_edges, \
    draw_wireframe, draw_vertices
from data_analysis.smpl_orientation_metrics import build_smpl_json_path
from visualizations.smpl_video_annotator import load_smpl_params


def get_color(tid):
    np.random.seed(hash(tid) % (2 ** 32))
    return tuple(int(x) for x in np.random.randint(50, 255, 3))


def generate_outlier_collage(data_dir, sequence, tid, fail_frame, max_extension, mean_variance, mean_foot_error,
                             smpl_model, faces, device, output_dir):
    print(f"     -> Generating HIGH-RES outlier visual for {tid[-4:]} at frame {fail_frame}...")

    PANEL_W, PANEL_H = 1280, 960
    panels = []

    json_path = build_smpl_json_path(data_dir, sequence, fail_frame, tid)
    if not os.path.exists(json_path):
        return

    betas, pose, trans = load_smpl_params(json_path)

    betas_t = torch.tensor(betas.reshape(1, 10), dtype=torch.float32, device=device)
    global_orient_t = torch.tensor(pose[:3].reshape(1, 3), dtype=torch.float32, device=device)
    body_pose_t = torch.tensor(pose[3:].reshape(1, 69), dtype=torch.float32, device=device)
    transl_t = torch.tensor(trans.reshape(1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        output = smpl_model(betas=betas_t, global_orient=global_orient_t, body_pose=body_pose_t, transl=transl_t,
                            return_verts=True)
    vertices = output.vertices.detach().cpu().numpy()[0]

    sampled_edges = build_sampled_mesh_edges(faces, stride=4)
    mesh_color = (0, 255, 255)

    primary_cameras = ['blu79CF', 'ylw79D0']

    for cam in primary_cameras:
        img_path = os.path.join(data_dir, 'images', sequence, cam, f"{sequence}_{cam}_{fail_frame:07d}.jpg")
        img = cv2.imread(img_path)

        if img is None:
            panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            panels.append(panel)
            continue

        P_rect, R_rect, T_range_to_cam = load_camera_calibration(Path(data_dir), sequence, cam)
        uv_vertices, valid_vertices = project_points(vertices, P_rect, R_rect, T_range_to_cam)

        img = cv2.addWeighted(img, 0.5, np.zeros_like(img), 0.5, 0)
        draw_wireframe(img, uv_vertices, valid_vertices, sampled_edges, mesh_color, thickness=2)

        if len(uv_vertices) > 0 and np.any(valid_vertices):
            valid_uv = uv_vertices[valid_vertices]
            min_x, min_y = np.min(valid_uv, axis=0).astype(int)
            max_x, max_y = np.max(valid_uv, axis=0).astype(int)

            pad = 60
            min_x = max(0, min_x - pad)
            min_y = max(0, min_y - pad)
            max_x = min(img.shape[1], max_x + pad)
            max_y = min(img.shape[0], max_y + pad)

            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 4, cv2.LINE_AA)

            # Appended the Foot Error explicitly onto the image label!
            label = f"OUTLIER: {tid[-4:]} | Error: {mean_foot_error:.2f}m"
            cv2.putText(img, label, (min_x, max(40, min_y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4,
                        cv2.LINE_AA)

        cv2.putText(img, cam, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)
        panels.append(cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA))

    collage = np.hstack((panels[0], panels[1]))

    filename = f"outlier_{tid[-4:]}_frame_{fail_frame}.png"
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, collage)