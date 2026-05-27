import json
import glob
import re
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import smplx


# =============================================================================
# STEP 4: OFFICIAL PEDX SMPL VIDEO ANNOTATION
# =============================================================================

SMPL_24_BONES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]


def parse_instance_filename(path):
    """
    Expected:
        20171207T2024_0000055_146c364b4657488ea6575101b219803e.json
    """
    stem = Path(path).stem
    parts = stem.split("_")

    frame_id = None
    for part in parts:
        if re.fullmatch(r"\d{7}", part):
            frame_id = int(part)
            break

    ped_id = parts[-1]

    if frame_id is None:
        return None

    return frame_id, ped_id


def parse_calib_file(path):
    data = {}

    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()

            try:
                data[key] = np.array([float(x) for x in value.split()], dtype=float)
            except ValueError:
                continue

    return data


def group_smpl_json_by_frame(smpl_label_dir):
    files = sorted(glob.glob(str(smpl_label_dir / "*.json")))
    grouped = defaultdict(list)

    for file_path in files:
        parsed = parse_instance_filename(file_path)

        if parsed is None:
            continue

        frame_id, ped_id = parsed
        grouped[frame_id].append((ped_id, Path(file_path)))

    return dict(grouped)


def image_path_for_frame(image_dir, sequence, camera, frame_id):
    return image_dir / f"{sequence}_{camera}_{frame_id:07d}.jpg"


def load_smpl_params(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    betas = np.asarray(data["betas"], dtype=np.float32).reshape(10)
    pose = np.asarray(data["pose"], dtype=np.float32).reshape(72)
    trans = np.asarray(data["trans"], dtype=np.float32).reshape(3)

    return betas, pose, trans


def load_camera_calibration(dataset_dir, sequence, camera):
    calib_dir = dataset_dir / "calib" / "calib"

    if not calib_dir.exists():
        calib_dir = dataset_dir / "calib"

    if camera == "blu79CF":
        cam_to_range_file = calib_dir / "calib_cam_to_range_blu79CF.txt"
        cam_to_cam_file = calib_dir / "calib_cam_to_cam_blu79CF-grn43E3.txt"
        p_key = "P_rect_00"
        r_key = "R_rect_00"

    elif camera == "ylw79D0":
        cam_to_range_file = calib_dir / "calib_cam_to_range_ylw79D0.txt"
        cam_to_cam_file = calib_dir / "calib_cam_to_cam_ylw79D0-red707B.txt"
        p_key = "P_rect_00"
        r_key = "R_rect_00"

    else:
        raise ValueError(
            "Step 4 currently supports direct projection for CAMERA='blu79CF' or CAMERA='ylw79D0'."
        )

    print("\n[STEP 4][CALIB]")
    print("cam_to_range:", cam_to_range_file)
    print("cam_to_cam  :", cam_to_cam_file)

    c2r = parse_calib_file(cam_to_range_file)
    c2c = parse_calib_file(cam_to_cam_file)

    T_cam_to_range = np.eye(4)

    if "R" in c2r and "T" in c2r:
        T_cam_to_range[:3, :3] = c2r["R"].reshape(3, 3)
        T_cam_to_range[:3, 3] = c2r["T"].reshape(3)

    elif "Tr" in c2r:
        T_cam_to_range[:3, :4] = c2r["Tr"].reshape(3, 4)

    else:
        raise KeyError(
            f"No R/T or Tr found in {cam_to_range_file}. Keys found: {list(c2r.keys())}"
        )

    # This direction was verified visually using trans projection.
    T_range_to_cam = np.linalg.inv(T_cam_to_range)

    P_rect = c2c[p_key].reshape(3, 4)

    R_rect = np.eye(4)
    if r_key in c2c:
        R_rect[:3, :3] = c2c[r_key].reshape(3, 3)

    return P_rect, R_rect, T_range_to_cam


def load_smpl_model(model_root, device):
    pkl_path = model_root / "smpl" / "SMPL_NEUTRAL.pkl"
    npz_path = model_root / "smpl" / "SMPL_NEUTRAL.npz"

    print("\n[STEP 4][SMPL]")
    print("MODEL_ROOT:", model_root)
    print("PKL exists:", pkl_path.exists(), "|", pkl_path)
    print("NPZ exists:", npz_path.exists(), "|", npz_path)

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"SMPL_NEUTRAL.pkl not found at {pkl_path}. "
            f"Convert/download the SMPL model before running Step 4."
        )

    model = smplx.create(
        str(model_root),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
    ).to(device)

    model.eval()

    faces = np.asarray(model.faces, dtype=np.int32)

    print("SMPL model loaded.")
    print("faces shape:", faces.shape)

    return model, faces


def run_smpl_batch(model, person_entries, device):
    ped_ids = []
    betas_list = []
    pose_list = []
    trans_list = []

    for ped_id, json_path in person_entries:
        betas, pose, trans = load_smpl_params(json_path)

        ped_ids.append(ped_id)
        betas_list.append(betas)
        pose_list.append(pose)
        trans_list.append(trans)

    betas_np = np.stack(betas_list, axis=0)
    pose_np = np.stack(pose_list, axis=0)
    trans_np = np.stack(trans_list, axis=0)

    betas_t = torch.tensor(betas_np, dtype=torch.float32, device=device)
    global_orient_t = torch.tensor(pose_np[:, :3], dtype=torch.float32, device=device)
    body_pose_t = torch.tensor(pose_np[:, 3:], dtype=torch.float32, device=device)
    transl_t = torch.tensor(trans_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        output = model(
            betas=betas_t,
            global_orient=global_orient_t,
            body_pose=body_pose_t,
            transl=transl_t,
            return_verts=True,
        )

    vertices = output.vertices.detach().cpu().numpy()
    joints = output.joints.detach().cpu().numpy()

    return ped_ids, vertices, joints


def project_points(points_3d, P_rect, R_rect, T_range_to_cam):
    points_3d = np.asarray(points_3d, dtype=float)

    if points_3d.ndim != 2 or points_3d.shape[1] != 3 or len(points_3d) == 0:
        return np.empty((0, 2)), np.empty((0,), dtype=bool)

    pts_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

    pts_cam = (T_range_to_cam @ pts_h.T).T
    pts_rect = (R_rect @ pts_cam.T).T
    proj = (P_rect @ pts_rect.T).T

    z = proj[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)

    uv = np.full((points_3d.shape[0], 2), np.nan, dtype=float)
    uv[valid, 0] = proj[valid, 0] / z[valid]
    uv[valid, 1] = proj[valid, 1] / z[valid]

    return uv, valid


def inside_image_mask(uv, valid, image_shape):
    h, w = image_shape[:2]

    return (
        valid
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )


def color_for_id(ped_id):
    seed = abs(hash(ped_id)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    color = rng.integers(70, 255, size=3)

    return int(color[0]), int(color[1]), int(color[2])


def build_sampled_mesh_edges(faces, stride=8):
    edges = set()

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))

    edges = np.array(sorted(edges), dtype=np.int32)

    if stride is not None and stride > 1:
        edges = edges[::stride]

    return edges


def draw_vertices(frame_overlay, uv, valid, color, max_vertices=2500, radius=1):
    mask = inside_image_mask(uv, valid, frame_overlay.shape)
    points = uv[mask].astype(np.int32)

    if len(points) > max_vertices:
        idx = np.linspace(0, len(points) - 1, max_vertices).astype(int)
        points = points[idx]

    for x, y in points:
        cv2.circle(
            frame_overlay,
            (int(x), int(y)),
            radius,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )

    return len(points)


def draw_wireframe(frame_overlay, uv, valid, edges, color, thickness=1):
    h, w = frame_overlay.shape[:2]
    drawn = 0

    for a, b in edges:
        if a >= len(uv) or b >= len(uv):
            continue

        if not valid[a] or not valid[b]:
            continue

        x1, y1 = uv[a]
        x2, y2 = uv[b]

        if not np.all(np.isfinite([x1, y1, x2, y2])):
            continue

        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue

        cv2.line(
            frame_overlay,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        drawn += 1

    return drawn


def draw_joints_and_skeleton(frame, joints, P_rect, R_rect, T_range_to_cam):
    uv, valid = project_points(joints, P_rect, R_rect, T_range_to_cam)
    mask = inside_image_mask(uv, valid, frame.shape)

    visible_count = int(np.sum(mask))
    n = min(24, len(joints))

    for a, b in SMPL_24_BONES:
        if a >= n or b >= n:
            continue

        if not valid[a] or not valid[b]:
            continue

        x1, y1 = uv[a]
        x2, y2 = uv[b]

        h, w = frame.shape[:2]

        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue

        cv2.line(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    for i in range(n):
        if not mask[i]:
            continue

        x, y = uv[i].astype(int)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

    return visible_count


def draw_person_id(frame, joints, ped_id, P_rect, R_rect, T_range_to_cam, color):
    uv, valid = project_points(joints, P_rect, R_rect, T_range_to_cam)
    mask = inside_image_mask(uv, valid, frame.shape)

    if not np.any(mask):
        return

    visible = uv[mask]
    x = int(np.median(visible[:, 0]))
    y = int(np.min(visible[:, 1]))

    cv2.putText(
        frame,
        ped_id[-6:],
        (x + 8, max(25, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def run(
    workspace_root=None,
    sequence="20171207T2024",
    camera="blu79CF",
    fps=10,
    max_frames=None,
    mesh_alpha=0.55,
    max_vertices_drawn_per_person=2500,
    edge_sample_stride=8,
):
    """
    Step 4 entry point.

    Generates a video with official PedX SMPL meshes projected onto the real camera frames.
    """

    print("\n" + "=" * 80)
    print("STEP 4: OFFICIAL PEDX SMPL VIDEO ANNOTATION")
    print("=" * 80)

    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
        # __file__ = pedx/human_skeleton_analysis/visualizations/smpl_video_annotator.py
        # parents[2] = pedx repo root

    workspace_root = Path(workspace_root)

    repo_root = workspace_root
    code_root = repo_root.parent.parent

    dataset_dir = code_root / r"downloaded_stuff\datasets\pedx\pedx_data"
    model_root = repo_root / "body_models"

    image_dir = dataset_dir / "images" / sequence / camera
    smpl_label_dir = dataset_dir / "labels" / "3d" / "smpl" / sequence

    output_dir = repo_root / "visualisation_human_skeleton_visualisation_analysis" / "step4_smpl_video_annotation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"pedx_scene2_{camera}_smpl_mesh_overlay_REAL.mp4"

    print("repo_root      :", repo_root)
    print("dataset_dir    :", dataset_dir)
    print("model_root     :", model_root)
    print("sequence       :", sequence)
    print("camera         :", camera)
    print("image_dir      :", image_dir)
    print("smpl_label_dir :", smpl_label_dir)
    print("output_video   :", output_video)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory missing: {image_dir}")

    if not smpl_label_dir.exists():
        raise FileNotFoundError(f"SMPL label directory missing: {smpl_label_dir}")

    device = torch.device("cpu")

    by_frame = group_smpl_json_by_frame(smpl_label_dir)
    frame_ids = sorted(by_frame.keys())

    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]

    print("\n[STEP 4][DISCOVER]")
    print("Frames with SMPL labels:", len(by_frame))
    print("Frames to write        :", len(frame_ids))

    if not frame_ids:
        print("[STEP 4][STOP] No SMPL frames found.")
        return None

    P_rect, R_rect, T_range_to_cam = load_camera_calibration(dataset_dir, sequence, camera)
    model, faces = load_smpl_model(model_root, device)
    sampled_edges = build_sampled_mesh_edges(faces, stride=edge_sample_stride)

    print("\n[STEP 4][MESH]")
    print("faces shape        :", faces.shape)
    print("sampled edges shape:", sampled_edges.shape)

    first_img = None
    first_frame_id = None

    for frame_id in frame_ids:
        path = image_path_for_frame(image_dir, sequence, camera, frame_id)

        if not path.exists():
            continue

        img = cv2.imread(str(path))

        if img is not None:
            first_img = img
            first_frame_id = frame_id
            break

    if first_img is None:
        print("[STEP 4][STOP] No readable images found.")
        return None

    h, w = first_img.shape[:2]

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video}")

    print("\n[STEP 4][VIDEO]")
    print("First readable frame:", first_frame_id)
    print("Resolution          :", w, "x", h)
    print("Writing video...")

    total_written = 0
    total_persons = 0
    total_visible_meshes = 0

    first_debug_printed = False

    for frame_id in frame_ids:
        img_path = image_path_for_frame(image_dir, sequence, camera, frame_id)

        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))

        if frame is None:
            continue

        persons = by_frame[frame_id]

        if not persons:
            continue

        ped_ids, vertices_batch, joints_batch = run_smpl_batch(model, persons, device)

        overlay = frame.copy()
        visible_meshes_this_frame = 0

        for i, ped_id in enumerate(ped_ids):
            vertices = vertices_batch[i]
            joints = joints_batch[i]

            if not first_debug_printed:
                print("\n[STEP 4][DEBUG FIRST PERSON]")
                print("frame_id      :", frame_id)
                print("ped_id        :", ped_id)
                print("vertices shape:", vertices.shape)
                print("joints shape  :", joints.shape)
                print("vertices min  :", vertices.min(axis=0))
                print("vertices max  :", vertices.max(axis=0))
                first_debug_printed = True

            color = color_for_id(ped_id)

            uv_vertices, valid_vertices = project_points(
                vertices,
                P_rect,
                R_rect,
                T_range_to_cam,
            )

            drawn_vertices = draw_vertices(
                overlay,
                uv_vertices,
                valid_vertices,
                color,
                max_vertices=max_vertices_drawn_per_person,
                radius=1,
            )

            drawn_edges = draw_wireframe(
                overlay,
                uv_vertices,
                valid_vertices,
                sampled_edges,
                color,
                thickness=1,
            )

            visible_joints = draw_joints_and_skeleton(
                frame,
                joints,
                P_rect,
                R_rect,
                T_range_to_cam,
            )

            draw_person_id(
                frame,
                joints,
                ped_id,
                P_rect,
                R_rect,
                T_range_to_cam,
                color,
            )

            if drawn_vertices > 0 or drawn_edges > 0 or visible_joints > 0:
                visible_meshes_this_frame += 1

            total_persons += 1

        frame = cv2.addWeighted(overlay, mesh_alpha, frame, 1.0 - mesh_alpha, 0)

        cv2.putText(
            frame,
            f"STEP 4 | {sequence} | {camera} | frame {frame_id:07d} | visible SMPL {visible_meshes_this_frame}/{len(persons)}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        total_written += 1
        total_visible_meshes += visible_meshes_this_frame

        if total_written % 25 == 0:
            print(
                f"[STEP 4] written={total_written:4d}, "
                f"frame={frame_id:07d}, "
                f"persons={len(persons)}, "
                f"visible_smpl={visible_meshes_this_frame}"
            )

    writer.release()

    print("\n[STEP 4][SUMMARY]")
    print("Frames written    :", total_written)
    print("Persons processed :", total_persons)
    print("Visible mesh hits :", total_visible_meshes)
    print("Output video      :", output_video)

    print("\nSTEP 4 COMPLETE")
    print("=" * 80)

    return output_video


if __name__ == "__main__":
    run()