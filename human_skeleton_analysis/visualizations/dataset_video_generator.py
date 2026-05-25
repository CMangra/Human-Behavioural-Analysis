import os
import cv2
import json
import glob
import numpy as np
import trimesh

def pre_process_and_visualize_dataset(data_dir, sequence, visualize=False, save_visualisation_path=None, max_frames=None):
    """
    Step 1: General PedX Visualization.
    Iterates through the dataset and builds a 6-panel overview video to verify data integrity.
    """
    print(f"\n[VISUALIZATION] Starting general dataset overview for sequence: {sequence}")
    print(f"[VISUALIZATION] Data directory set to: {data_dir}")
    
    cameras = ['blu79CF', 'grn43E3', 'red707B', 'ylw79D0']
    
    # 1. Scan for frames to determine sequence length
    search_path = os.path.join(data_dir, 'images', sequence, cameras[0], '*.jpg')
    img_files = glob.glob(search_path)
    
    if not img_files:
        print(f"[ERROR] Could not find any images at {search_path}. Check your data_dir path!")
        return

    frame_ids = sorted([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in img_files])
    
    if max_frames:
        frame_ids = frame_ids[:max_frames]
        print(f"[VISUALIZATION] max_frames set to {max_frames}. Truncating sequence.")

    total_frames = len(frame_ids)
    print(f"[VISUALIZATION] Found {total_frames} frames to process (From ID {frame_ids[0]} to {frame_ids[-1]}).")

    # 2. Prepare Output Video
    out_video = None
    PANEL_W, PANEL_H = 640, 480
    
    if save_visualisation_path:
        # Delete existing file if overwriting
        if os.path.exists(save_visualisation_path):
            print(f"[VISUALIZATION] Overwriting existing video at: {save_visualisation_path}")
            os.remove(save_visualisation_path)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(save_visualisation_path, fourcc, 10.0, (PANEL_W * 3, PANEL_H * 2))
        print(f"[VISUALIZATION] Video writer initialized -> {save_visualisation_path}")

    # 3. Helper Functions for Panels
    def get_color(tid):
        np.random.seed(hash(tid) % (2**32))
        return tuple(int(x) for x in np.random.randint(50, 255, 3))

    def render_cam(cam, f_id):
        img_path = os.path.join(data_dir, 'images', sequence, cam, f"{sequence}_{cam}_{f_id:07d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            cv2.putText(img, f"Missing: {cam}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return img

        json_pattern = os.path.join(data_dir, 'labels', '2d', sequence, f"{sequence}_{cam}_{f_id:07d}_*.json")
        for jf in glob.glob(json_pattern):
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                tid = data.get('tracking_id', 'unknown')
                color = get_color(tid)
                
                if data.get('polygon'):
                    poly = np.array(data['polygon'], np.int32)
                    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2)
                if data.get('keypoint'):
                    for jname, coords in data['keypoint'].items():
                        if coords.get('visible', True):
                            cv2.circle(img, (int(coords['x']), int(coords['y'])), 4, (0, 0, 255), -1)
            except Exception as e:
                print(f"[WARNING] Corrupt JSON {os.path.basename(jf)}: {e}")

        cv2.putText(img, cam, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return cv2.resize(img, (PANEL_W, PANEL_H))

    def render_bev(f_id):
        bev = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        def pt2px(x, y):
            return int((x + 40) / 80 * PANEL_W), int((40 - y) / 80 * PANEL_H)

        ply_pattern = os.path.join(data_dir, 'labels', '3d', 'segment', sequence, f"{sequence}_{f_id:07d}_*.ply")
        for ply_path in glob.glob(ply_pattern):
            tid = os.path.basename(ply_path).split('_')[-1].split('.')[0]
            try:
                pcd = trimesh.load(ply_path, process=False)
                pts = np.array(pcd.vertices)
                for p in pts[::2]: # Downsample for speed
                    u, v = pt2px(p[0], p[1])
                    if 0 <= u < PANEL_W and 0 <= v < PANEL_H:
                        cv2.circle(bev, (u, v), 2, get_color(tid), -1)
            except Exception:
                pass
                
        cv2.putText(bev, "LiDAR BEV", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return bev

    def render_info(f_id, curr_idx):
        panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        cv2.putText(panel, "Step 1: Dataset Overview", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(panel, f"Sequence: {sequence}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(panel, f"Frame ID: {f_id:07d}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(panel, f"Progress: {curr_idx}/{total_frames}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return panel

    # 4. Processing Loop
    print("[VISUALIZATION] Commencing render loop...")
    for i, curr_frame in enumerate(frame_ids):
        if i % 10 == 0:
            print(f"  -> Processing frame index {i}/{total_frames} (ID: {curr_frame})")
            
        top_row = np.hstack((render_cam(cameras[0], curr_frame), render_cam(cameras[1], curr_frame), render_cam(cameras[2], curr_frame)))
        bottom_row = np.hstack((render_cam(cameras[3], curr_frame), render_bev(curr_frame), render_info(curr_frame, i+1)))
        collage = np.vstack((top_row, bottom_row))
        
        if save_visualisation_path:
            out_video.write(collage)
            
        if visualize:
            # Resize for screen viewing
            display_img = cv2.resize(collage, (1280, 720))
            cv2.imshow("PedX Overview", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[VISUALIZATION] Visualization interrupted by user.")
                break

    if out_video:
        out_video.release()
    cv2.destroyAllWindows()
    print("[VISUALIZATION] Step 1 Complete.")