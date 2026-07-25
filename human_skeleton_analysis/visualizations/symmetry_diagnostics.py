import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import seaborn as sns


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def map_to_matplotlib_coords(joints):
    """
    Maps SMPL coordinates to Matplotlib 3D coordinates.
    Based on visual diagnostics:
    - Column 0 is lateral (X)
    - Column 1 is depth (Y)
    - Column 2 is vertical, but inverted (Z)
    """
    j_mapped = np.zeros_like(joints)
    j_mapped[:, 0] = joints[:, 0]  # X stays X
    j_mapped[:, 1] = joints[:, 1]  # Y stays Y (Depth)
    j_mapped[:, 2] = -joints[:, 2]  # Invert Z (Flips feet from ceiling to floor)

    return j_mapped


def plot_gait_symmetry_diagnostics(tid, times, theta_1_list, theta_2_list, diff_list, sample_joints, output_dir):
    short_id = tid[-4:]
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # PLOT 1: The Biomechanical Time-Series
    # ---------------------------------------------------------
    fig_ts, ax_ts = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig_ts.suptitle(f"Biomechanical Gait Symmetry Analysis | Pedestrian {short_id}", fontsize=14)

    ax_ts[0].plot(times, theta_1_list, label="Left Leg Angle (θ1)", color="blue", linewidth=2)
    ax_ts[0].plot(times, theta_2_list, label="Right Leg Angle (θ2)", color="orange", linewidth=2)
    ax_ts[0].set_ylabel("Angle from Center [deg]")
    ax_ts[0].set_title("Oscillation of Legs During Walking Cycle")
    ax_ts[0].axvline(0, color="green", linestyle="--", label="Turn Onset")
    ax_ts[0].legend()
    ax_ts[0].grid(True, linestyle=":", alpha=0.7)

    ax_ts[1].plot(times, diff_list, label="|θ1| - |θ2| (Symmetry Delta)", color="red", linewidth=2)
    ax_ts[1].axhline(0, color="black", linewidth=1)
    ax_ts[1].axvline(0, color="green", linestyle="--")
    ax_ts[1].set_ylabel("Symmetry Difference [deg]")
    ax_ts[1].set_xlabel("Time relative to onset [s]")
    ax_ts[1].set_title("Symmetry Variance (Outlier Detection Metric)")
    ax_ts[1].legend()
    ax_ts[1].grid(True, linestyle=":", alpha=0.7)

    ts_path = Path(output_dir) / f"gait_timeseries_{short_id}.png"
    save_figure(fig_ts, ts_path)

    # ---------------------------------------------------------
    # PLOT 2: The 3D Skeleton Prover
    # ---------------------------------------------------------
    fig_3d = plt.figure(figsize=(8, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.set_title(f"3D Structural Math Proof | Frame at Turn Onset\nCenter Line vs. Leg Vectors")

    PELVIS, L_HIP, R_HIP, L_KNEE, R_KNEE, NECK = 0, 1, 2, 4, 5, 12

    # Map coordinates so it stands upright
    j = map_to_matplotlib_coords(sample_joints)
    xs, ys, zs = j[:, 0], j[:, 1], j[:, 2]

    ax_3d.scatter(xs, ys, zs, c='black', s=20)
    ax_3d.plot([j[PELVIS, 0], j[NECK, 0]], [j[PELVIS, 1], j[NECK, 1]], [j[PELVIS, 2], j[NECK, 2]], 'k--', linewidth=3,
               label="Center Line")
    ax_3d.plot([j[L_HIP, 0], j[L_KNEE, 0]], [j[L_HIP, 1], j[L_KNEE, 1]], [j[L_HIP, 2], j[L_KNEE, 2]], 'b-', linewidth=3,
               label="Left Thigh")
    ax_3d.plot([j[R_HIP, 0], j[R_KNEE, 0]], [j[R_HIP, 1], j[R_KNEE, 1]], [j[R_HIP, 2], j[R_KNEE, 2]], 'r-', linewidth=3,
               label="Right Thigh")
    ax_3d.plot([j[L_HIP, 0], j[R_HIP, 0]], [j[L_HIP, 1], j[R_HIP, 1]], [j[L_HIP, 2], j[R_HIP, 2]], 'gray', linewidth=2)

    ax_3d.legend()

    max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (xs.max() + xs.min()) * 0.5, (ys.max() + ys.min()) * 0.5, (zs.max() + zs.min()) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    skel_path = Path(output_dir) / f"gait_skeleton_proof_{short_id}.png"
    save_figure(fig_3d, skel_path)


def generate_gait_symmetry_video(tid, onset_frame, times, theta_1_list, theta_2_list, diff_list, joints_list,
                                 output_dir, fps=10):
    """
    Renders a split-screen MP4. Left: 3D Skeleton. Right: Evolving time-series graph.
    """
    short_id = tid[-4:]
    video_path = str(Path(output_dir) / f"gait_animation_{short_id}_onset_{onset_frame}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    PELVIS, L_HIP, R_HIP, L_KNEE, R_KNEE, NECK = 0, 1, 2, 4, 5, 12

    print(f"     -> Rendering {len(times)} video frames for {short_id}...")

    for i in range(len(times)):
        fig = plt.figure(figsize=(16, 6))

        # --- LEFT PANEL: 3D SKELETON ---
        ax_3d = fig.add_subplot(121, projection='3d')

        # Map coordinates so it stands upright in the video
        j = map_to_matplotlib_coords(joints_list[i])
        xs, ys, zs = j[:, 0], j[:, 1], j[:, 2]

        ax_3d.scatter(xs, ys, zs, c='black', s=10)
        ax_3d.plot([j[PELVIS, 0], j[NECK, 0]], [j[PELVIS, 1], j[NECK, 1]], [j[PELVIS, 2], j[NECK, 2]], 'k--',
                   linewidth=2, label="Spine")
        ax_3d.plot([j[L_HIP, 0], j[L_KNEE, 0]], [j[L_HIP, 1], j[L_KNEE, 1]], [j[L_HIP, 2], j[L_KNEE, 2]], 'b-',
                   linewidth=3, label="Left Leg")
        ax_3d.plot([j[R_HIP, 0], j[R_KNEE, 0]], [j[R_HIP, 1], j[R_KNEE, 1]], [j[R_HIP, 2], j[R_KNEE, 2]], 'r-',
                   linewidth=3, label="Right Leg")
        ax_3d.plot([j[L_HIP, 0], j[R_HIP, 0]], [j[L_HIP, 1], j[R_HIP, 1]], [j[L_HIP, 2], j[R_HIP, 2]], 'gray',
                   linewidth=2)

        mid_x, mid_y, mid_z = np.mean(xs), np.mean(ys), np.mean(zs)
        ax_3d.set_xlim(mid_x - 0.8, mid_x + 0.8)
        ax_3d.set_ylim(mid_y - 0.8, mid_y + 0.8)
        ax_3d.set_zlim(mid_z - 0.8, mid_z + 0.8)

        ax_3d.set_title(f"3D Biomechanics | Time: {times[i]:.2f}s", fontsize=12)
        ax_3d.view_init(elev=20, azim=45)
        ax_3d.legend(loc="upper left")

        # --- RIGHT PANEL: EVOLVING GRAPH ---
        ax_2d = fig.add_subplot(122)

        ax_2d.plot(times[:i + 1], theta_1_list[:i + 1], label="Left Leg Angle", color="blue", linewidth=2)
        ax_2d.plot(times[:i + 1], theta_2_list[:i + 1], label="Right Leg Angle", color="orange", linewidth=2)
        ax_2d.plot(times[:i + 1], diff_list[:i + 1], label="Symmetry Delta", color="red", linewidth=2, linestyle=':')

        ax_2d.axvline(times[i], color="black", linewidth=1.5, label="Current Frame")
        ax_2d.axvline(0, color="green", linestyle="--", alpha=0.5, label="Turn Onset")

        ax_2d.set_xlim(min(times), max(times))
        y_min = min(min(theta_1_list), min(diff_list)) - 5
        y_max = max(max(theta_2_list), max(diff_list)) + 5
        ax_2d.set_ylim(y_min, y_max)

        ax_2d.set_title("Gait Symmetry Time Series", fontsize=12)
        ax_2d.set_xlabel("Time relative to turn onset [s]")
        ax_2d.set_ylabel("Degrees")
        ax_2d.legend(loc="upper right")
        ax_2d.grid(True, linestyle=":", alpha=0.7)

        # --- RENDER TO OPENCV ---
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        if writer is None:
            h, w = img_bgr.shape[:2]
            writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        writer.write(img_bgr)
        plt.close(fig)

    if writer is not None:
        writer.release()
    print(f"     -> Video saved: {video_path}")



def plot_leg_extension_distribution(all_theta_l, all_theta_r, threshold, output_dir):
    """
    Generates a high-quality distribution plot of the leg extension angles
    for the thesis, marking the Constraint A threshold.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn for a smooth kernel density estimate (KDE) and histogram
    sns.histplot(all_theta_l, color="blue", label="Left Leg (\u03B8_L)", kde=True, stat="density", alpha=0.4, bins=30,
                 ax=ax)
    sns.histplot(all_theta_r, color="orange", label="Right Leg (\u03B8_R)", kde=True, stat="density", alpha=0.4,
                 bins=30, ax=ax)

    # Add the threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2.5,
               label=f"Constraint A Threshold ({threshold}\u00B0)")

    # Formatting for thesis quality
    ax.set_xlabel("Absolute Angle Relative to Vertical Spine Vector [Degrees]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Save PNG and SVG
    png_path = Path(output_dir) / "constraint_A_distribution.png"
    svg_path = Path(output_dir) / "constraint_A_distribution.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"     -> Saved Distribution Plots to {output_dir}")