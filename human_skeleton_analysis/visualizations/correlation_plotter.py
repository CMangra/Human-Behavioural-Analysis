import os
import matplotlib.pyplot as plt


def plot_behavioral_correlation(tid, onset_frame, metrics, out_dir):
    """Generates the final thesis graphs proving pre-turn biomechanics."""
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        f"Behavioral Pre-Turn Correlation\nID: {tid[-4:]} | Onset Frame: {onset_frame} | Camera: {metrics['cam']}")

    # Metric 1: Shoulder Compression (Foreshortening)
    ax1.plot(metrics['frames_seq'], metrics['shoulder_widths'], marker='o', color='b', label='Projected Shoulder Width')
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Turn Onset (Trajectory Changes)')
    ax1.set_ylabel("Width (pixels)")
    ax1.legend()
    ax1.grid(True)

    # Metric 2: Head Yaw
    ax2.plot(metrics['frames_seq'], metrics['head_offsets'], marker='s', color='g', label='Head Yaw Offset')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-')
    ax2.set_ylabel("Normalized Head Offset")
    ax2.set_xlabel("Frames Relative to Turn (Negative = Before Turn)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"correlation_{tid[-4:]}_onset_{onset_frame}.png"))
    plt.close()