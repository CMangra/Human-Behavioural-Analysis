import os
import cv2
import math
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


def generate_outlier_collage(data_dir, sequence, tid, fail_frame, max_extension, mean_variance, smpl_model, faces,
                             device, output_dir):
    print(f"     -> Generating HIGH-RES outlier visual for {tid[-4:]} at frame {fail_frame}...")

    PANEL_W, PANEL_H = 1280, 960
    panels = []

    json_path = build_smpl_json_path(data_dir, sequence, fail_frame, tid)
    if not os.path.exists(json_path):
        return None

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
            label = f"OUTLIER: {tid[-4:]}"
            cv2.putText(img, label, (min_x, max(40, min_y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4,
                        cv2.LINE_AA)

        cv2.putText(img, cam, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)
        panels.append(cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA))

    collage = np.hstack((panels[0], panels[1]))

    filename = f"outlier_{tid[-4:]}_frame_{fail_frame}_ext_{max_extension:.1f}_var_{mean_variance:.1f}.png"
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, collage)
    return out_path


def generate_foot_contact_collage(data_dir, sequence, tid, fail_frame, max_foot_error, lidar_max_z, smpl_model, faces,
                                  device, output_dir):
    print(f"     -> Generating LiDAR Foot Contact visual for {tid[-4:]} at frame {fail_frame}...")

    PANEL_W, PANEL_H = 1280, 960
    panels = []

    json_path = build_smpl_json_path(data_dir, sequence, fail_frame, tid)
    if not os.path.exists(json_path):
        return None

    betas, pose, trans = load_smpl_params(json_path)

    betas_t = torch.tensor(betas.reshape(1, 10), dtype=torch.float32, device=device)
    global_orient_t = torch.tensor(pose[:3].reshape(1, 3), dtype=torch.float32, device=device)
    body_pose_t = torch.tensor(pose[3:].reshape(1, 69), dtype=torch.float32, device=device)
    transl_t = torch.tensor(trans.reshape(1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        output = smpl_model(betas=betas_t, global_orient=global_orient_t, body_pose=body_pose_t, transl=transl_t,
                            return_verts=True)
    vertices = output.vertices.detach().cpu().numpy()[0]
    joints = output.joints.detach().cpu().numpy()[0]

    sampled_edges = build_sampled_mesh_edges(faces, stride=4)
    mesh_color = (0, 165, 255)
    ground_color = (0, 255, 0)

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
        img = cv2.addWeighted(img, 0.4, np.zeros_like(img), 0.6, 0)
        draw_wireframe(img, uv_vertices, valid_vertices, sampled_edges, mesh_color, thickness=2)

        floor_pts = np.array([
            [-10, -10, lidar_max_z],
            [10, -10, lidar_max_z],
            [10, 10, lidar_max_z],
            [-10, 10, lidar_max_z]
        ])
        uv_floor, valid_floor = project_points(floor_pts, P_rect, R_rect, T_range_to_cam)

        if np.all(valid_floor):
            cv2.line(img, tuple(uv_floor[0].astype(int)), tuple(uv_floor[1].astype(int)), ground_color, 4, cv2.LINE_AA)
            cv2.putText(img, "LiDAR Ground Plane", tuple(uv_floor[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        ground_color, 3, cv2.LINE_AA)

        uv_joints, valid_joints = project_points(joints, P_rect, R_rect, T_range_to_cam)
        for f_idx in [10, 11]:
            if valid_joints[f_idx]:
                fx, fy = int(uv_joints[f_idx, 0]), int(uv_joints[f_idx, 1])
                cv2.circle(img, (fx, fy), 12, (0, 0, 255), -1, cv2.LINE_AA)

        label = f"OUTLIER: {tid[-4:]} | Error: {max_foot_error:.2f}m"
        cv2.putText(img, label, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)

        cv2.putText(img, cam, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)
        panels.append(cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA))

    collage = np.hstack((panels[0], panels[1]))

    filename = f"foot_ground_error_{tid[-4:]}_frame_{fail_frame}_dist_{max_foot_error:.2f}m.png"
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, collage)

    return out_path


def generate_global_outlier_matrix(image_paths, output_path, cols=2, line_thickness=10, line_color=(255, 255, 255)):
    """
    Takes a list of individual outlier collages and stitches them together
    into a single, massive high-res grid for the thesis.
    Now forces a 2-column layout and adds separator lines.
    """
    if not image_paths:
        return

    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)

    if not images:
        return

    # Force the grid layout to 'cols' (default 2)
    n = len(images)
    rows = math.ceil(n / cols)

    img_h, img_w = images[0].shape[:2]

    # Calculate canvas size, adding space for the separator lines between images
    canvas_h = (rows * img_h) + ((rows - 1) * line_thickness)
    canvas_w = (cols * img_w) + ((cols - 1) * line_thickness)

    # Initialize the canvas with the line color (default white)
    canvas = np.full((canvas_h, canvas_w, 3), line_color, dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols

        # Ensure exact shape matching before assignment
        if img.shape[:2] != (img_h, img_w):
            img = cv2.resize(img, (img_w, img_h))

        # Calculate placement coordinates, factoring in the line gaps
        y_start = r * (img_h + line_thickness)
        y_end = y_start + img_h

        x_start = c * (img_w + line_thickness)
        x_end = x_start + img_w

        canvas[y_start:y_end, x_start:x_end] = img

    cv2.imwrite(output_path, canvas)
    print(f"     -> Saved Full-Res Global Outlier Matrix to {output_path}")


def generate_height_group_collage(data_dir, sequence, tid, frame_id, height_m, smpl_model, faces, device,
                                  output_base_dir):
    """
    Projects the SMPL mesh onto the primary cameras and saves it into
    subfolders based on the height (<1.5m or >=1.5m) for investigation.
    """
    # Sort into respective folders
    group_folder = "group_under_1_5m" if height_m < 1.5 else "group_over_1_5m"
    output_dir = os.path.join(output_base_dir, group_folder)
    os.makedirs(output_dir, exist_ok=True)

    print(f"     -> Saving Height Investigation Frame ({height_m:.2f}m) into {group_folder}...")

    PANEL_W, PANEL_H = 1280, 960
    panels = []

    json_path = build_smpl_json_path(data_dir, sequence, frame_id, tid)
    if not os.path.exists(json_path):
        return None

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
    mesh_color = (255, 0, 255)  # Magenta for visibility

    primary_cameras = ['blu79CF', 'ylw79D0']

    for cam in primary_cameras:
        img_path = os.path.join(data_dir, 'images', sequence, cam, f"{sequence}_{cam}_{frame_id:07d}.jpg")
        img = cv2.imread(img_path)

        if img is None:
            panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            cv2.putText(panel, f"Missing: {cam}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            panels.append(panel)
            continue

        P_rect, R_rect, T_range_to_cam = load_camera_calibration(Path(data_dir), sequence, cam)
        uv_vertices, valid_vertices = project_points(vertices, P_rect, R_rect, T_range_to_cam)

        img = cv2.addWeighted(img, 0.7, np.zeros_like(img), 0.3, 0)
        draw_wireframe(img, uv_vertices, valid_vertices, sampled_edges, mesh_color, thickness=2)

        cv2.putText(img, cam, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)

        # Add height label
        label = f"ID: {tid[-4:]} | Height: {height_m:.2f}m"
        cv2.putText(img, label, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4, cv2.LINE_AA)

        panels.append(cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA))

    collage = np.hstack((panels[0], panels[1]))

    filename = f"height_investigation_{tid[-4:]}_frame_{frame_id}_h_{height_m:.2f}m.png"
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, collage)
    return out_path