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


# =============================================================================
# STEP 5: TIMESTAMP-AWARE 3D SMPL ORIENTATION DEVIATION VISUALISATION
# =============================================================================

REFERENCE_NAMES = ["stable", "past_05s", "past_10s", "past_15s", "tangent"]


def plot_event_orientation_sensitivity(
    tid,
    onset_frame,
    event_result,
    output_dir,
):
    short_id = tid[-4:]

    event_df = event_result["event_df"]
    best_head_axis = event_result["best_head_axis"]
    peak_frame = event_result["peak_frame"]
    peak_time = event_result["peak_time"]
    av_times = event_result["av_times"]
    av_values = event_result["av_values"]

    os.makedirs(output_dir, exist_ok=True)

    times = event_df["time_seconds_relative_to_onset_from_timestamps"].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 14),
        sharex=False,
        constrained_layout=True,
    )

    fig.suptitle(
        f"Step 5: Timestamp-Aware 3D SMPL Orientation Sensitivity\n"
        f"ID: {short_id} | onset frame {onset_frame} | peak frame {peak_frame}\n"
        f"Time axis uses PedX image timestamps | selected diagnostic head axis: {best_head_axis}",
        fontsize=14,
    )

    # -------------------------------------------------------------------------
    # Plot 1: Angular velocity
    # -------------------------------------------------------------------------
    axes[0].plot(
        av_times,
        av_values,
        color="purple",
        linewidth=2,
        label="Step 2 angular velocity",
    )

    axes[0].axvline(0, color="green", linestyle="--", linewidth=2, label="Turn onset")

    if peak_time is not None:
        axes[0].axvline(peak_time, color="red", linestyle="--", linewidth=2, label="Turn peak")

    axes[0].set_ylabel("Angular velocity [deg/frame]")
    axes[0].set_xlabel("Time relative to onset [s], from PedX timestamps")
    axes[0].set_title(
        "Existing Step 2 angular velocity "
        "(y-axis remains deg/frame; x-axis uses real timestamps)"
    )
    axes[0].legend()
    axes[0].grid(True)

    # -------------------------------------------------------------------------
    # Plot 2: Head sensitivity
    # -------------------------------------------------------------------------
    for ref_name in REFERENCE_NAMES:
        col = f"head_{best_head_axis}_{ref_name}_baseline_corrected_deg"
        baseline_col = f"head_{best_head_axis}_{ref_name}_baseline_deg"

        if col not in event_df.columns:
            continue

        baseline_value = event_df[baseline_col].dropna().iloc[0] if baseline_col in event_df.columns and not event_df[baseline_col].dropna().empty else np.nan

        axes[1].plot(
            times,
            event_df[col].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            label=f"{ref_name}, baseline={baseline_value:.1f}",
        )

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(0, color="green", linestyle="--", linewidth=2)

    if peak_time is not None:
        axes[1].axvline(peak_time, color="red", linestyle="--", linewidth=2)

    axes[1].set_ylabel("Head deviation change [deg]")
    axes[1].set_xlabel("Time relative to onset [s], from PedX timestamps")
    axes[1].set_title(
        f"Head orientation sensitivity after baseline correction "
        f"(selected diagnostic head axis: {best_head_axis})"
    )
    axes[1].legend()
    axes[1].grid(True)

    # -------------------------------------------------------------------------
    # Plot 3: Shoulder / torso sensitivity
    # -------------------------------------------------------------------------
    for ref_name in REFERENCE_NAMES:
        col = f"shoulder_{ref_name}_baseline_corrected_deg"
        baseline_col = f"shoulder_{ref_name}_baseline_deg"

        if col not in event_df.columns:
            continue

        baseline_value = event_df[baseline_col].dropna().iloc[0] if baseline_col in event_df.columns and not event_df[baseline_col].dropna().empty else np.nan

        axes[2].plot(
            times,
            event_df[col].to_numpy(dtype=float),
            marker="s",
            linewidth=1.8,
            label=f"{ref_name}, baseline={baseline_value:.1f}",
        )

    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].axvline(0, color="green", linestyle="--", linewidth=2)

    if peak_time is not None:
        axes[2].axvline(peak_time, color="red", linestyle="--", linewidth=2)

    axes[2].set_ylabel("Shoulder deviation change [deg]")
    axes[2].set_xlabel("Time relative to onset [s], from PedX timestamps")
    axes[2].set_title("Shoulder / torso orientation sensitivity after baseline correction")
    axes[2].legend()
    axes[2].grid(True)

    out_path = os.path.join(
        output_dir,
        f"orientation_sensitivity_{short_id}_onset_{onset_frame}.png",
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    csv_path = os.path.join(
        output_dir,
        f"orientation_sensitivity_{short_id}_onset_{onset_frame}.csv",
    )

    event_df.to_csv(csv_path, index=False)

    return out_path, csv_path


def window_mean(event_df, column, start_s, end_s):
    if column not in event_df.columns:
        return np.nan

    times = event_df["time_seconds_relative_to_onset_from_timestamps"].to_numpy(dtype=float)
    values = event_df[column].to_numpy(dtype=float)

    mask = (
        (times >= start_s)
        & (times <= end_s)
        & np.isfinite(values)
    )

    if np.sum(mask) == 0:
        return np.nan

    return float(np.nanmean(values[mask]))


def run(
    data_dir,
    sequence,
    qualified_trajectories,
    turn_results,
    output_base,
    repo_root=None,
    pre_onset_seconds=4.0,
    post_onset_seconds=3.0,
    baseline_start_s=-3.0,
    baseline_end_s=-1.5,
):
    """
    Step 5 entry point.

    Reuses:
        - qualified_trajectories from Step 2
        - turn_results from Step 2
        - compute_kinematics() from Step 2
        - SMPL loading from Step 4
        - PedX timestamps for x-axis seconds and past-direction windows

    Generates:
        - one timestamp-aware orientation sensitivity plot per turn event
        - one CSV per turn event
        - one summary CSV
    """

    print("\n" + "=" * 80)
    print("STEP 5: TIMESTAMP-AWARE 3D SMPL ORIENTATION SENSITIVITY VISUALISATION")
    print("=" * 80)

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    else:
        repo_root = Path(repo_root)

    model_root = repo_root / "body_models"

    output_dir = os.path.join(
        output_base,
        "step5_3d_smpl_orientation_sensitivity",
    )

    os.makedirs(output_dir, exist_ok=True)

    print("data_dir          :", data_dir)
    print("sequence          :", sequence)
    print("repo_root         :", repo_root)
    print("model_root        :", model_root)
    print("output_dir        :", output_dir)
    print("turning pedestrians:", len(turn_results))

    print("\n[STEP 5][TIMESTAMPS]")
    frame_to_time = load_frame_timestamps(Path(data_dir), sequence)

    if frame_to_time is None:
        print("[STEP 5][TIMESTAMPS][WARN] Using fallback frame/FPS timing.")
    else:
        print("[STEP 5][TIMESTAMPS] Real PedX timestamp timing enabled.")

    device = torch.device("cpu")

    smpl_model, _ = load_smpl_model(
        model_root=model_root,
        device=device,
    )

    generated_outputs = []
    summaries = []

    for tid, onsets in turn_results.items():
        short_id = tid[-4:]

        if tid not in qualified_trajectories:
            print(f"[STEP 5][WARN] ID {short_id} is in turn_results but not qualified_trajectories.")
            continue

        print("\n" + "-" * 80)
        print(f"[STEP 5] Processing ID {short_id} with {len(onsets)} turn event(s)")

        frames_dict = qualified_trajectories[tid]
        kinematics = compute_kinematics(frames_dict)

        if kinematics is None:
            print(f"[STEP 5][WARN] Kinematics unavailable for ID {short_id}")
            continue

        for onset_frame in onsets:
            result = compute_event_orientation_sensitivity(
                data_dir=data_dir,
                sequence=sequence,
                tid=tid,
                onset_frame=onset_frame,
                kinematics=kinematics,
                smpl_model=smpl_model,
                device=device,
                frame_to_time=frame_to_time,
                pre_seconds=pre_onset_seconds,
                post_seconds=post_onset_seconds,
                baseline_start_s=baseline_start_s,
                baseline_end_s=baseline_end_s,
            )

            if result is None:
                print(f"[STEP 5][WARN] Could not compute event for ID {short_id}, onset {onset_frame}")
                continue

            out_path, csv_path = plot_event_orientation_sensitivity(
                tid=tid,
                onset_frame=onset_frame,
                event_result=result,
                output_dir=output_dir,
            )

            generated_outputs.append(out_path)

            event_df = result["event_df"]
            best_head_axis = result["best_head_axis"]

            summary_row = {
                "tid": tid,
                "short_id": short_id,
                "onset_frame": onset_frame,
                "peak_frame": result["peak_frame"],
                "peak_time_seconds_from_timestamps": result["peak_time"],
                "best_head_axis": best_head_axis,
                "best_head_axis_baseline_std": result["best_head_axis_score"],
                "head_stable_mean_pre_1s_to_0s": window_mean(
                    event_df,
                    f"head_{best_head_axis}_stable_baseline_corrected_deg",
                    -1.0,
                    0.0,
                ),
                "head_stable_mean_0s_to_peak": window_mean(
                    event_df,
                    f"head_{best_head_axis}_stable_baseline_corrected_deg",
                    0.0,
                    result["peak_time"] if result["peak_time"] is not None else post_onset_seconds,
                ),
                "shoulder_stable_mean_pre_1s_to_0s": window_mean(
                    event_df,
                    "shoulder_stable_baseline_corrected_deg",
                    -1.0,
                    0.0,
                ),
                "shoulder_stable_mean_0s_to_peak": window_mean(
                    event_df,
                    "shoulder_stable_baseline_corrected_deg",
                    0.0,
                    result["peak_time"] if result["peak_time"] is not None else post_onset_seconds,
                ),
                "head_tangent_mean_pre_1s_to_0s": window_mean(
                    event_df,
                    f"head_{best_head_axis}_tangent_baseline_corrected_deg",
                    -1.0,
                    0.0,
                ),
                "shoulder_tangent_mean_pre_1s_to_0s": window_mean(
                    event_df,
                    "shoulder_tangent_baseline_corrected_deg",
                    -1.0,
                    0.0,
                ),
                "event_csv": csv_path,
                "event_plot": out_path,
            }

            summaries.append(summary_row)

            print(f"[STEP 5] Generated: {out_path}")

    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join(output_dir, "all_turn_events_orientation_sensitivity_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n[STEP 5][SUMMARY]")
    print("Generated plots:", len(generated_outputs))
    print("Output folder  :", output_dir)
    print("Summary CSV    :", summary_csv)

    print("\nSTEP 5 COMPLETE")
    print("=" * 80)

    return generated_outputs