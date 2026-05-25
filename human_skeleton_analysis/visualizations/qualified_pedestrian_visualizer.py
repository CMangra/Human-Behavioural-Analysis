import os
import cv2
import json
import numpy as np
import config
from data_analysis.turn_detection import compute_kinematics


def generate_qualified_summaries(data_dir, sequence, qualified_trajectories, turn_results, out_dir):
    print("\n[VISUALIZATION] Generating static summary panels for qualified pedestrians...")
    os.makedirs(out_dir, exist_ok=True)
    PANEL_W, PANEL_H = 640, 480

    def get_color(tid):
        np.random.seed(hash(tid) % (2 ** 32))
        return tuple(int(x) for x in np.random.randint(50, 255, 3))

    for tid, frames_dict in qualified_trajectories.items():
        kinematics = compute_kinematics(frames_dict)
        if kinematics is None: continue

        sorted_frames = kinematics["sorted_frames"]
        xs, ys = kinematics["xs"], kinematics["ys"]
        onsets = turn_results.get(tid, [])

        rep_frame = onsets[0] if onsets else sorted_frames[len(sorted_frames) // 2]

        def render_cam(cam):
            img_path = os.path.join(data_dir, 'images', sequence, cam, f"{sequence}_{cam}_{rep_frame:07d}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                return img

            json_path = os.path.join(data_dir, 'labels', '2d', sequence, f"{sequence}_{cam}_{rep_frame:07d}_{tid}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if data.get('polygon'):
                    poly = np.array(data['polygon'], np.int32)
                    cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=3)
                if data.get('keypoint'):
                    for jname, coords in data['keypoint'].items():
                        if coords.get('visible', True):
                            cv2.circle(img, (int(coords['x']), int(coords['y'])), 6, (0, 0, 255), -1)

            cv2.putText(img, cam, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return cv2.resize(img, (PANEL_W, PANEL_H))

        def render_bev():
            bev = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

            # Anchor mapping to the representation frame's smoothed coordinate
            try:
                rep_idx = sorted_frames.index(rep_frame)
                cx, cy = xs[rep_idx], ys[rep_idx]
            except ValueError:
                cx, cy = xs[0], ys[0]

            def pt2px(x, y):
                return int((x - cx + 40) / 80 * PANEL_W), int((40 - (y - cy)) / 80 * PANEL_H)

            for i in range(0, PANEL_W, int(PANEL_W / 8)):
                cv2.line(bev, (i, 0), (i, PANEL_H), (30, 30, 30), 1)
                cv2.line(bev, (0, i), (PANEL_W, i), (30, 30, 30), 1)

            bev_pts = [pt2px(xs[i], ys[i]) for i in range(len(xs))]

            for i in range(1, len(bev_pts)):
                cv2.line(bev, bev_pts[i - 1], bev_pts[i], (200, 200, 200), 2)

            # Add Blue Start Marker
            if bev_pts:
                cv2.rectangle(bev, (bev_pts[0][0] - 6, bev_pts[0][1] - 6), (bev_pts[0][0] + 6, bev_pts[0][1] + 6),
                              (255, 0, 0), -1)

            # Add Red Onset Dots
            for onset_f in onsets:
                if onset_f in sorted_frames:
                    idx = sorted_frames.index(onset_f)
                    ox, oy = pt2px(xs[idx], ys[idx])
                    cv2.circle(bev, (ox, oy), 8, (0, 0, 255), -1)

            # Current frame yellow dot
            cv2.circle(bev, pt2px(cx, cy), 6, (0, 255, 255), -1)

            cv2.putText(bev, "LiDAR BEV Tracker", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if onsets: cv2.putText(bev, "RED DOTS = Turn Onsets", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                                   2)
            return bev

        def render_info():
            panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            cv2.putText(panel, f"ID: {tid[-4:]}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(panel, f"Turn Events: {len(onsets)}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2)
            cv2.putText(panel, f"Total Tracking: {len(sorted_frames)} frames", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (200, 200, 200), 2)
            if onsets:
                cv2.putText(panel, f"Onset Frames: {[int(f) for f in onsets]}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            return panel

        cam1 = render_cam(config.CAMERAS[0])
        cam2 = render_cam(config.CAMERAS[1])
        cam3 = render_cam(config.CAMERAS[2])
        cam4 = render_cam(config.CAMERAS[3])
        bev = render_bev()
        info = render_info()

        top_row = np.hstack((cam1, cam2, cam3))
        bottom_row = np.hstack((cam4, bev, info))
        collage = np.vstack((top_row, bottom_row))

        status = "TURNER" if onsets else "STRAIGHT"
        out_path = os.path.join(out_dir, f"{status}_person_{tid[-4:]}.jpg")
        cv2.imwrite(out_path, collage)

    print(f"[VISUALIZATION] Rendered {len(qualified_trajectories)} summary panels to {out_dir}")
