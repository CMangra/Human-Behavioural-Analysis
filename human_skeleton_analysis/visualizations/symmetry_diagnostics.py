import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import seaborn as sns


def save_figure(fig, base_path):
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def map_to_matplotlib_coords(joints):
    j_mapped = np.zeros_like(joints)
    j_mapped[:, 0] = joints[:, 0]
    j_mapped[:, 1] = joints[:, 1]
    j_mapped[:, 2] = -joints[:, 2]
    return j_mapped


def plot_gait_symmetry_diagnostics(tid, times, theta_1_list, theta_2_list, diff_list, sample_joints, output_dir):
    short_id = tid[-4:]
    os.makedirs(output_dir, exist_ok=True)

    fig_ts, ax_ts = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_ts[0].plot(times, theta_1_list, label="Left Leg Angle (\u03B8L)", color="blue", linewidth=2)
    ax_ts[0].plot(times, theta_2_list, label="Right Leg Angle (\u03B8R)", color="orange", linewidth=2)
    ax_ts[0].set_ylabel("Angle from Center [deg]")
    ax_ts[0].axvline(0, color="green", linestyle="--", label="Turn Onset")
    ax_ts[0].legend()
    ax_ts[0].grid(True, linestyle=":", alpha=0.7)

    ax_ts[1].plot(times, diff_list, label="|\u03B8L| - |\u03B8R| (Symmetry Delta)", color="red", linewidth=2)
    ax_ts[1].axhline(0, color="black", linewidth=1)
    ax_ts[1].axvline(0, color="green", linestyle="--")
    ax_ts[1].set_ylabel("Symmetry Difference [deg]")
    ax_ts[1].set_xlabel("Time relative to onset [s]")
    ax_ts[1].legend()
    ax_ts[1].grid(True, linestyle=":", alpha=0.7)

    ts_path = Path(output_dir) / f"gait_timeseries_{short_id}"
    save_figure(fig_ts, ts_path)

    fig_3d = plt.figure(figsize=(10, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    PELVIS, L_HIP, R_HIP, L_KNEE, R_KNEE, NECK = 0, 1, 2, 4, 5, 12
    HEAD, L_SHOULDER, R_SHOULDER = 15, 16, 17

    j = map_to_matplotlib_coords(sample_joints)
    xs, ys, zs = j[:, 0], j[:, 1], j[:, 2]

    SMPL_BONES = [
        (0, 1), (0, 2), (0, 3), (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11),
        (3, 6), (6, 9), (9, 12), (12, 15), (12, 13), (13, 16), (16, 18), (18, 20),
        (20, 22), (12, 14), (14, 17), (17, 19), (19, 21), (21, 23)
    ]

    for bone in SMPL_BONES:
        ax_3d.plot([xs[bone[0]], xs[bone[1]]], [ys[bone[0]], ys[bone[1]]], [zs[bone[0]], zs[bone[1]]], color='gray',
                   linewidth=1.5, alpha=0.6)

    ax_3d.scatter(xs, ys, zs, c='black', s=25, alpha=0.8)

    gait_indices = [PELVIS, L_HIP, R_HIP, L_KNEE, R_KNEE, NECK]
    ax_3d.scatter(xs[gait_indices], ys[gait_indices], zs[gait_indices], c='red', s=50, zorder=5)

    orient_indices = [HEAD, L_SHOULDER, R_SHOULDER]
    ax_3d.scatter(xs[orient_indices], ys[orient_indices], zs[orient_indices], c='purple', s=50, zorder=5,
                  label="Head & Shoulder Orientation Joints")

    ax_3d.plot([j[PELVIS, 0], j[NECK, 0]], [j[PELVIS, 1], j[NECK, 1]], [j[PELVIS, 2], j[NECK, 2]], 'k--', linewidth=3,
               label=r"Spine Vector ($\vec{V}_{spine}$)", zorder=4)
    ax_3d.plot([j[L_HIP, 0], j[L_KNEE, 0]], [j[L_HIP, 1], j[L_KNEE, 1]], [j[L_HIP, 2], j[L_KNEE, 2]], 'b-', linewidth=3,
               label=r"Left Thigh Vector ($\vec{V}_{L\_leg}$)", zorder=4)
    ax_3d.plot([j[R_HIP, 0], j[R_KNEE, 0]], [j[R_HIP, 1], j[R_KNEE, 1]], [j[R_HIP, 2], j[R_KNEE, 2]], 'r-', linewidth=3,
               label=r"Right Thigh Vector ($\vec{V}_{R\_leg}$)", zorder=4)

    for idx in range(len(xs)):
        if idx in gait_indices:
            ax_3d.text(xs[idx], ys[idx], zs[idx] + 0.04, f"{idx}", color='red', fontsize=12, fontweight='bold',
                       zorder=6)
        elif idx in orient_indices:
            ax_3d.text(xs[idx], ys[idx], zs[idx] + 0.04, f"{idx}", color='purple', fontsize=12, fontweight='bold',
                       zorder=6)
        else:
            ax_3d.text(xs[idx], ys[idx], zs[idx] + 0.03, f"{idx}", color='black', fontsize=7, zorder=6)

    ax_3d.legend()

    max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (xs.max() + xs.min()) * 0.5, (ys.max() + ys.min()) * 0.5, (zs.max() + zs.min()) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    save_figure(fig_3d, Path(output_dir) / f"gait_skeleton_proof_{short_id}")


def plot_foot_contact_diagnostics(tid, times, l_foot_zs, r_foot_zs, ground_zs, output_dir):
    short_id = tid[-4:]
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, l_foot_zs, label="Left Foot Height", color="blue", linewidth=2)
    ax.plot(times, r_foot_zs, label="Right Foot Height", color="orange", linewidth=2)
    ax.plot(times, ground_zs, label="LiDAR Ground Plane", color="green", linestyle="--", linewidth=3)

    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Time relative to turn onset [s]")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5, label="Turn Onset")

    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.7)

    save_figure(fig, Path(output_dir) / f"gait_foot_contact_{short_id}")


def plot_foot_contact_3d_proof(tid, frame_id, joints, lidar_max_z, output_dir):
    short_id = tid[-4:]
    os.makedirs(output_dir, exist_ok=True)

    fig_3d = plt.figure(figsize=(10, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    PELVIS = 0
    L_HIP, R_HIP = 1, 2
    L_KNEE, R_KNEE = 4, 5
    L_ANKLE, R_ANKLE = 7, 8
    L_FOOT, R_FOOT = 10, 11

    j = map_to_matplotlib_coords(joints)
    xs, ys, zs = j[:, 0], j[:, 1], j[:, 2]

    ax_3d.plot([xs[PELVIS], xs[L_HIP]], [ys[PELVIS], ys[L_HIP]], [zs[PELVIS], zs[L_HIP]], 'gray', linewidth=2)
    ax_3d.plot([xs[PELVIS], xs[R_HIP]], [ys[PELVIS], ys[R_HIP]], [zs[PELVIS], zs[R_HIP]], 'gray', linewidth=2)

    ax_3d.plot([xs[L_HIP], xs[L_KNEE]], [ys[L_HIP], ys[L_KNEE]], [zs[L_HIP], zs[L_KNEE]], 'b-', linewidth=4,
               label="Left Leg")
    ax_3d.plot([xs[L_KNEE], xs[L_ANKLE]], [ys[L_KNEE], ys[L_ANKLE]], [zs[L_KNEE], zs[L_ANKLE]], 'b-', linewidth=4)
    ax_3d.plot([xs[L_ANKLE], xs[L_FOOT]], [ys[L_ANKLE], ys[L_FOOT]], [zs[L_ANKLE], zs[L_FOOT]], 'b-', linewidth=4)

    ax_3d.plot([xs[R_HIP], xs[R_KNEE]], [ys[R_HIP], ys[R_KNEE]], [zs[R_HIP], zs[R_KNEE]], 'r-', linewidth=4,
               label="Right Leg")
    ax_3d.plot([xs[R_KNEE], xs[R_ANKLE]], [ys[R_KNEE], ys[R_ANKLE]], [zs[R_KNEE], zs[R_ANKLE]], 'r-', linewidth=4)
    ax_3d.plot([xs[R_ANKLE], xs[R_FOOT]], [ys[R_ANKLE], ys[R_FOOT]], [zs[R_ANKLE], zs[R_FOOT]], 'r-', linewidth=4)

    foot_indices = [PELVIS, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT]
    ax_3d.scatter(xs[foot_indices], ys[foot_indices], zs[foot_indices], c='black', s=60, zorder=5)

    for idx in foot_indices:
        ax_3d.text(xs[idx], ys[idx], zs[idx] + 0.03, f"{idx}", color='black', fontsize=14, fontweight='bold', zorder=6)

    mapped_ground_z = -lidar_max_z

    xx, yy = np.meshgrid(np.linspace(xs.min() - 0.5, xs.max() + 0.5, 2), np.linspace(ys.min() - 0.5, ys.max() + 0.5, 2))
    zz = np.full_like(xx, mapped_ground_z)
    ax_3d.plot_surface(xx, yy, zz, alpha=0.3, color='green')

    ax_3d.plot([], [], [], color='green', alpha=0.3, linewidth=10, label="LiDAR Ground Plane")

    z_max = max(zs[L_HIP], zs[R_HIP]) + 0.1
    z_min = min(zs[L_FOOT], zs[R_FOOT], mapped_ground_z) - 0.2
    ax_3d.set_zlim(z_min, z_max)

    mid_x, mid_y = (xs.max() + xs.min()) * 0.5, (ys.max() + ys.min()) * 0.5
    xy_range = 0.5
    ax_3d.set_xlim(mid_x - xy_range, mid_x + xy_range)
    ax_3d.set_ylim(mid_y - xy_range, mid_y + xy_range)

    ax_3d.view_init(elev=10, azim=45)
    ax_3d.legend(loc="upper left")

    save_figure(fig_3d, Path(output_dir) / f"foot_contact_proof_{short_id}_frame_{frame_id}")


def plot_skeleton_height_proof(tid, frame_id, joints, head_pos, mid_foot_pos, height_m, output_dir):
    """
    Generates a 3D visualization showing exactly how the Anatomical Height
    is measured via a line from the Head to the geometric Mid-Foot.
    """
    short_id = tid[-4:]
    os.makedirs(output_dir, exist_ok=True)

    fig_3d = plt.figure(figsize=(8, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    j = map_to_matplotlib_coords(joints)
    xs, ys, zs = j[:, 0], j[:, 1], j[:, 2]

    # Map the custom head and mid_foot vectors to the matplotlib coordinate space
    head_mapped = np.array([head_pos[0], head_pos[1], -head_pos[2]])
    foot_mapped = np.array([mid_foot_pos[0], mid_foot_pos[1], -mid_foot_pos[2]])

    SMPL_BONES = [
        (0, 1), (0, 2), (0, 3), (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11),
        (3, 6), (6, 9), (9, 12), (12, 15), (12, 13), (13, 16), (16, 18), (18, 20),
        (20, 22), (12, 14), (14, 17), (17, 19), (19, 21), (21, 23)
    ]

    # Draw full skeleton in light gray
    for bone in SMPL_BONES:
        ax_3d.plot([xs[bone[0]], xs[bone[1]]], [ys[bone[0]], ys[bone[1]]], [zs[bone[0]], zs[bone[1]]],
                   color='lightgray', linewidth=2)

    ax_3d.scatter(xs, ys, zs, c='lightgray', s=20)

    # Highlight the Head and the two Feet
    ax_3d.scatter(xs[[15, 10, 11]], ys[[15, 10, 11]], zs[[15, 10, 11]], c='black', s=50, zorder=5)

    # Draw the bold measurement line
    ax_3d.plot([head_mapped[0], foot_mapped[0]],
               [head_mapped[1], foot_mapped[1]],
               [head_mapped[2], foot_mapped[2]],
               color='magenta', linewidth=4, linestyle='--', label=f"Skeleton Height: {height_m:.2f} m", zorder=6)

    # Add the text directly onto the line
    mid_line_z = (head_mapped[2] + foot_mapped[2]) / 2.0
    ax_3d.text(head_mapped[0], head_mapped[1], mid_line_z, f"{height_m:.2f} m", color='magenta', fontsize=14,
               fontweight='bold', zorder=7)

    ax_3d.legend(loc="upper left")

    max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (xs.max() + xs.min()) * 0.5, (ys.max() + ys.min()) * 0.5, (zs.max() + zs.min()) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    save_figure(fig_3d, Path(output_dir) / f"body_height_proof_{short_id}_frame_{frame_id}")


def plot_leg_extension_distribution(all_theta_l, all_theta_r, threshold, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(all_theta_l, color="blue", label="Left Leg (\u03B8_L)", stat="count", alpha=0.4, bins=30, ax=ax)
    sns.histplot(all_theta_r, color="orange", label="Right Leg (\u03B8_R)", stat="count", alpha=0.4, bins=30, ax=ax)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2.5,
               label=f"Constraint A Threshold ({threshold}\u00B0)")

    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Absolute Angle Relative to Vertical Spine Vector [Degrees]", fontsize=12)
    ax.set_ylabel("Count (Number of Poses)", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    save_figure(fig, Path(output_dir) / "global_constraint_A_distribution")


def plot_symmetry_variance_distribution(all_mean_variances, threshold, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(all_mean_variances, color="purple", stat="count", alpha=0.4, bins=15, ax=ax)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2.5, label=f"Upper Bound (+{threshold}\u00B0)")
    ax.axvline(-threshold, color='red', linestyle='--', linewidth=2.5, label=f"Lower Bound (-{threshold}\u00B0)")

    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Sequence Mean Symmetry Variance [Degrees]", fontsize=12)
    ax.set_ylabel("Count (Number of Sequences)", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    save_figure(fig, Path(output_dir) / "global_constraint_B_distribution")


def plot_foot_error_distribution(all_foot_errors, threshold, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(all_foot_errors, color="green", stat="count", alpha=0.4, bins=30, ax=ax)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2.5, label=f"Constraint C Threshold ({threshold}m)")
    ax.axvline(-threshold, color='red', linestyle='--', linewidth=2.5, label=f"Constraint C Threshold (-{threshold}m)")

    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Frame Distance from Ground Plane [Meters]", fontsize=12)
    ax.set_ylabel("Count (Number of Poses)", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    save_figure(fig, Path(output_dir) / "global_constraint_C_distribution")


def plot_height_distribution(all_heights, min_thresh, max_thresh, output_dir):
    """
    Generates the distribution plot for Constraint D (Body Size / Height).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(all_heights, color="magenta", stat="count", alpha=0.4, bins=30, ax=ax)
    ax.axvline(min_thresh, color='red', linestyle='--', linewidth=2.5, label=f"Minimum Height ({min_thresh}m)")
    ax.axvline(max_thresh, color='red', linestyle='-', linewidth=2.5, label=f"Maximum Height ({max_thresh}m)")

    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Anatomical Skeleton Height [Meters]", fontsize=12)
    ax.set_ylabel("Count (Number of Poses)", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    save_figure(fig, Path(output_dir) / "global_constraint_D_distribution")


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
