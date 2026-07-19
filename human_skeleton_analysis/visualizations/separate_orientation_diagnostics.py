import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data_analysis.turn_detection import compute_kinematics
from data_analysis.timestamp_utils import load_frame_timestamps
from data_analysis.smpl_orientation_metrics import compute_event_orientation_sensitivity
from visualizations.smpl_video_annotator import load_smpl_model


def save_figure(fig, base_path):
    png_path = Path(str(base_path) + ".png")
    svg_path = Path(str(base_path) + ".svg")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return str(png_path), str(svg_path)


def plot_individual_event_figures(tid, onset_frame, event_result, kinematics, output_dir):
    short_id = tid[-4:]
    person_dir = Path(output_dir) / f"pedestrian_{short_id}"
    person_dir.mkdir(parents=True, exist_ok=True)

    event_df = event_result["event_df"]
    best_head_axis = event_result["best_head_axis"]
    peak_frame = event_result["peak_frame"]
    peak_time = event_result["peak_time"]
    av_times = event_result["av_times"]
    av_values = event_result["av_values"]
    stable_heading = event_result["stable_heading"]
    times = event_df["time_seconds_relative_to_onset_from_timestamps"].to_numpy(dtype=float)

    # 1. BEV Trajectory
    fig_bev, ax_bev = plt.subplots(figsize=(12, 4))
    xs, ys = kinematics["xs"], kinematics["ys"]
    sorted_frames = kinematics["sorted_frames"]
    ax_bev.plot(xs, ys, color="gray", linewidth=2, label="Smoothed BEV trajectory")
    if onset_frame in sorted_frames:
        idx = list(sorted_frames).index(onset_frame)
        ax_bev.scatter(xs[idx], ys[idx], color="green", s=100, label="onset")
        ax_bev.arrow(xs[idx], ys[idx], stable_heading[0] * 3, stable_heading[1] * 3, head_width=0.4, color="blue",
                     length_includes_head=True, label="stable heading")
    if peak_frame in sorted_frames:
        idx = list(sorted_frames).index(peak_frame)
        ax_bev.scatter(xs[idx], ys[idx], color="red", marker="x", s=120, label="peak")
    ax_bev.set_aspect("equal", "datalim")
    ax_bev.set_title(f"BEV Trajectory\nPedestrian {tid}")
    ax_bev.legend(loc="upper left")
    ax_bev.grid(True)
    bev_png, bev_svg = save_figure(fig_bev, person_dir / f"fig_a_bev_{short_id}_onset_{onset_frame}")

    # 2. Angular Velocity
    fig_av, ax_av = plt.subplots(figsize=(12, 4))
    ax_av.plot(av_times, av_values, color="purple", linewidth=2, label="Angular velocity")
    ax_av.axvline(0, color="green", linestyle="--", linewidth=2, label="onset")
    if peak_time is not None: ax_av.axvline(peak_time, color="red", linestyle="--", linewidth=2, label="peak")
    ax_av.set_title(f"Angular Velocity\nPedestrian {tid}")
    ax_av.set_xlabel("Time relative to onset [s]")
    ax_av.legend(loc="upper left")
    ax_av.grid(True)
    av_png, av_svg = save_figure(fig_av, person_dir / f"fig_b_av_{short_id}_onset_{onset_frame}")

    # 3. Head Sensitivity
    fig_head, ax_head = plt.subplots(figsize=(12, 4))
    for ref in ["stable", "past_05s", "past_10s", "past_15s", "tangent"]:
        col = f"head_{best_head_axis}_{ref}_baseline_corrected_deg"
        if col in event_df.columns:
            ax_head.plot(times, event_df[col], marker="o", label=ref)
    ax_head.axhline(0, color="black")
    ax_head.axvline(0, color="green", linestyle="--")
    if peak_time is not None: ax_head.axvline(peak_time, color="red", linestyle="--")
    ax_head.set_title(f"Head Sensitivity\nPedestrian {tid}")
    ax_head.legend(loc="upper left")
    ax_head.grid(True)
    head_png, head_svg = save_figure(fig_head, person_dir / f"fig_c_head_{short_id}_onset_{onset_frame}")

    # 4. Shoulder Sensitivity
    fig_sho, ax_sho = plt.subplots(figsize=(12, 4))
    for ref in ["stable", "past_05s", "past_10s", "past_15s", "tangent"]:
        col = f"shoulder_{ref}_baseline_corrected_deg"
        if col in event_df.columns:
            ax_sho.plot(times, event_df[col], marker="s", label=ref)
    ax_sho.axhline(0, color="black")
    ax_sho.axvline(0, color="green", linestyle="--")
    if peak_time is not None: ax_sho.axvline(peak_time, color="red", linestyle="--")
    ax_sho.set_title(f"Shoulder Sensitivity\nPedestrian {tid}")
    ax_sho.legend(loc="upper left")
    ax_sho.grid(True)
    sho_png, sho_svg = save_figure(fig_sho, person_dir / f"fig_d_shoulder_{short_id}_onset_{onset_frame}")

    # Export CSV
    csv_path = person_dir / f"event_data_{short_id}_onset_{onset_frame}.csv"
    event_df.to_csv(csv_path, index=False)

    return {
        "person_output_dir": str(person_dir),
        "event_csv": str(csv_path),
        "bev_png": bev_png, "angular_velocity_png": av_png,
        "head_png": head_png, "shoulder_png": sho_png
    }


def window_mean(event_df, column, start_s, end_s):
    if column not in event_df.columns: return np.nan
    times = event_df["time_seconds_relative_to_onset_from_timestamps"].to_numpy(dtype=float)
    values = event_df[column].to_numpy(dtype=float)
    mask = (times >= start_s) & (times <= end_s) & np.isfinite(values)
    if np.sum(mask) == 0: return np.nan
    return float(np.nanmean(values[mask]))


def run(data_dir, sequence, qualified_trajectories, turn_results, output_base, model_root, pre_onset_seconds=4.0,
        post_onset_seconds=3.0, baseline_start_s=-3.0, baseline_end_s=-1.5):
    output_dir = os.path.join(output_base, "step5_diagnostics_separate_figures")
    os.makedirs(output_dir, exist_ok=True)

    frame_to_time = load_frame_timestamps(Path(data_dir), sequence)
    device = torch.device("cpu")
    smpl_model, _ = load_smpl_model(model_root, device)

    summaries = []
    for tid, onsets in turn_results.items():
        if tid not in qualified_trajectories: continue
        kinematics = compute_kinematics(qualified_trajectories[tid])
        if kinematics is None: continue

        for onset_frame in onsets:
            result = compute_event_orientation_sensitivity(
                data_dir, sequence, tid, onset_frame, kinematics, smpl_model, device, frame_to_time,
                pre_onset_seconds, post_onset_seconds, baseline_start_s, baseline_end_s
            )
            if result is None: continue

            plot_paths = plot_individual_event_figures(tid, onset_frame, result, kinematics, output_dir)
            event_df = result["event_df"]
            best_axis = result["best_head_axis"]
            peak_time = result["peak_time"] if result["peak_time"] is not None else post_onset_seconds

            summary_row = {
                "tid": tid, "short_id": tid[-4:], "onset_frame": onset_frame,
                "peak_frame": result["peak_frame"], "peak_time_s": result["peak_time"],
                "best_head_axis": best_axis,
                "head_stable_mean_pre_1s_to_0s": window_mean(event_df,
                                                             f"head_{best_axis}_stable_baseline_corrected_deg", -1.0,
                                                             0.0),
                "head_stable_mean_0s_to_peak": window_mean(event_df, f"head_{best_axis}_stable_baseline_corrected_deg",
                                                           0.0, peak_time),
                "shoulder_stable_mean_pre_1s_to_0s": window_mean(event_df, "shoulder_stable_baseline_corrected_deg",
                                                                 -1.0, 0.0),
                "shoulder_stable_mean_0s_to_peak": window_mean(event_df, "shoulder_stable_baseline_corrected_deg", 0.0,
                                                               peak_time),
                **plot_paths
            }
            summaries.append(summary_row)

    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join(output_dir, "all_turn_events_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[STEP 5.2 COMPLETE] Generated diagnostics for {len(summaries)} events. Summary: {summary_csv}")