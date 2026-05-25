import os
import matplotlib.pyplot as plt
import config
from data_analysis.turn_detection import compute_kinematics


def generate_kinematic_debug_graphs(trajectories, out_dir):
    print("\n[VISUALIZATION] Generating Kinematic Turn Math Debug Graphs...")
    os.makedirs(out_dir, exist_ok=True)

    for tid, frames_dict in trajectories.items():
        kinematics = compute_kinematics(frames_dict)
        if kinematics is None: continue

        sorted_frames = kinematics["sorted_frames"]
        xs, ys = kinematics["xs"], kinematics["ys"]
        smoothed_ang_vel = kinematics["smoothed_ang_vel"]
        peaks, onset_indices = kinematics["peaks"], kinematics["onset_indices"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Top Panel
        ax1.set_title(f"ID: {tid[-4:]} - X/Y Trajectory (Top-Down)")
        ax1.plot(xs, ys, c='gray', label="Smoothed Path")
        if len(xs) > 0: ax1.scatter(xs[0], ys[0], c='blue', marker='s', s=100, label="Start Point")

        if peaks.size > 0:
            ax1.scatter([xs[p + 1] for p in peaks], [ys[p + 1] for p in peaks], c='red', marker='x', s=100,
                        label="Detected Peaks")
        if onset_indices:
            ax1.scatter([xs[idx + 1] for idx in onset_indices], [ys[idx + 1] for idx in onset_indices], c='green',
                        s=100, label="Backtracked Onsets")

        ax1.legend()
        ax1.set_aspect('equal', 'datalim')

        # Bottom Panel
        ax2.set_title("Angular Velocity (Degrees per Frame)")
        ax2.plot(sorted_frames[1:-1], smoothed_ang_vel, label="Smoothed Ang Vel")
        ax2.axhline(y=config.TURN_NOISE_FLOOR_DEG, c='k', linestyle='--', label="Noise Floor")

        if peaks.size > 0:
            ax2.scatter([sorted_frames[p + 1] for p in peaks], [smoothed_ang_vel[p] for p in peaks], c='red',
                        marker='x', s=100, label="Peaks")
        if onset_indices:
            ax2.scatter([sorted_frames[idx + 1] for idx in onset_indices],
                        [smoothed_ang_vel[idx] for idx in onset_indices], c='green', s=100, label="Onsets")

        ax2.set_xlabel("Frame ID")
        ax2.set_ylabel("Degrees/Frame")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"debug_math_{tid[-4:]}.png"))
        plt.close()
