import os
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from data_analysis.pedestrian_filtering import filter_pedestrians_by_visibility
from data_analysis.turn_detection import detect_multiple_turns_with_onset, compute_kinematics
from visualizations.statistics_plotter import generate_step2_graphs
from visualizations.qualified_pedestrian_visualizer import generate_qualified_summaries
from visualizations.turn_math_debugger import generate_kinematic_debug_graphs
from data_analysis.skeleton_metrics import extract_skeleton_metrics
from visualizations.correlation_plotter import plot_behavioral_correlation

from data_analysis.timestamp_utils import load_frame_timestamps, relative_time_seconds
from data_analysis.smpl_audit import run_dataset_audit
from visualizations.separate_orientation_diagnostics import run as run_step5_separate_diagnostics
from visualizations.smpl_video_annotator import load_smpl_model, load_smpl_params

# Imports for Step 6 & 7
from data_analysis.smpl_orientation_metrics import smpl_forward_joints, build_smpl_json_path
from data_analysis.gait_biomechanics import compute_leg_symmetry_simple, compute_foot_contact
from data_analysis.outlier_elimination import evaluate_biomechanical_plausibility
from visualizations.symmetry_diagnostics import plot_gait_symmetry_diagnostics, plot_foot_contact_diagnostics, \
    plot_leg_extension_distribution, plot_symmetry_variance_distribution, plot_foot_error_distribution
from visualizations.outlier_visualizer import generate_outlier_collage

ENVIRONMENT = "SERVER"

if ENVIRONMENT == "SERVER":
    WORKSPACE_ROOT = Path("/workspace")
    REPO_ROOT = WORKSPACE_ROOT / "Human-Behavioural-Analysis"
    DATASET_DIR = WORKSPACE_ROOT / "datasets" / "pedx"
    MODEL_ROOT = REPO_ROOT / "body_models"
    OUTPUT_BASE = REPO_ROOT / "visualisation_human_skeleton_visualisation_analysis"
    TARGET_SEQUENCES = ["20171207T2024", "20171130T2000"]
else:
    WORKSPACE_ROOT = Path(r"G:\My Drive\Desktop\THD\Master\JBData\3. Semester\code")
    REPO_ROOT = WORKSPACE_ROOT / r"Third-Semester-Code\pedx"
    DATASET_DIR = WORKSPACE_ROOT / r"downloaded_stuff\datasets\pedx\pedx_data"
    MODEL_ROOT = REPO_ROOT / "body_models"
    OUTPUT_BASE = REPO_ROOT / r"visualisation_human_skeleton_visualisation_analysis"
    TARGET_SEQUENCES = ["20171207T2024"]


def main():
    print("=" * 50)
    print("PEDX HUMAN SKELETON ANALYSIS PIPELINE")
    print("=" * 50)

    thesis_global_theta_l = []
    thesis_global_theta_r = []
    thesis_global_sequence_means = []
    thesis_global_foot_errors = []

    for sequence in TARGET_SEQUENCES:
        print(f"\n=== PROCESSING SEQUENCE: {sequence} ===")
        seq_output_base = OUTPUT_BASE / sequence

        step2_graphs_dir = seq_output_base / "step2_pedestrian_filtering" / "analysis_graphs"
        step2_frames_dir = seq_output_base / "step2_pedestrian_filtering" / "frames"
        step2_math_debug_dir = seq_output_base / "step2_pedestrian_filtering" / "math_debug_graphs"

        qualified_trajectories, vis_stats = filter_pedestrians_by_visibility(str(DATASET_DIR), sequence)
        generate_kinematic_debug_graphs(qualified_trajectories, str(step2_math_debug_dir))
        turn_results, turn_stats = detect_multiple_turns_with_onset(qualified_trajectories)
        generate_step2_graphs(vis_stats, turn_stats, str(step2_graphs_dir))
        generate_qualified_summaries(str(DATASET_DIR), sequence, qualified_trajectories, turn_results,
                                     str(step2_frames_dir))

        step3_out_dir = seq_output_base / "step3_behavioral_correlation"
        for tid, onsets in turn_results.items():
            for onset in onsets:
                metrics = extract_skeleton_metrics(str(DATASET_DIR), sequence, tid, onset, config.CAMERAS)
                if metrics:
                    plot_behavioral_correlation(tid, onset, metrics, str(step3_out_dir))

        frame_to_time = load_frame_timestamps(DATASET_DIR, sequence)
        audit_out_dir = seq_output_base / "step5_dataset_filtering_results"

        usable_turn_results, event_df = run_dataset_audit(
            data_dir=str(DATASET_DIR), sequence=sequence, qualified_trajectories=qualified_trajectories,
            turn_results=turn_results, frame_to_time=frame_to_time, output_dir=str(audit_out_dir)
        )

        if usable_turn_results:
            run_step5_separate_diagnostics(
                data_dir=str(DATASET_DIR), sequence=sequence, qualified_trajectories=qualified_trajectories,
                turn_results=usable_turn_results, output_base=str(seq_output_base), model_root=Path(MODEL_ROOT)
            )

        print("\n" + "=" * 80)
        print("STEP 6: BIOMECHANICAL GAIT SYMMETRY ANALYSIS (FUSION)")
        print("=" * 80)

        step6_out_dir = seq_output_base / "step6_biomechanics_diagnostics"
        step6_out_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cpu")
        smpl_model, faces = load_smpl_model(MODEL_ROOT, device)

        outlier_audit_results = []
        outliers_to_visualize = []

        if usable_turn_results:
            for tid, onsets in usable_turn_results.items():
                short_id = tid[-4:]
                frames_dict = qualified_trajectories[tid]
                kinematics = compute_kinematics(frames_dict)

                for onset_frame in onsets:
                    times, theta_l_list, theta_r_list, sym_variance_list = [], [], [], []
                    l_foot_z_list, r_foot_z_list, ground_z_list, foot_error_list = [], [], [], []

                    frame_ids_list = []
                    sample_joints = None

                    for f in kinematics["sorted_frames"]:
                        t = relative_time_seconds(f, onset_frame, frame_to_time)
                        if -4.0 <= t <= 3.0:
                            json_path = build_smpl_json_path(str(DATASET_DIR), sequence, f, tid)
                            if os.path.exists(json_path):
                                betas, pose, trans = load_smpl_params(json_path)
                                joints = smpl_forward_joints(smpl_model, betas, pose, trans, device)

                                t_l_deg, t_r_deg, sym_variance = compute_leg_symmetry_simple(joints)

                                # Fetch LiDAR ground plane Z for this specific frame
                                idx = kinematics["sorted_frames"].index(f)
                                lidar_max_z = kinematics["max_zs"][idx]
                                l_foot_z, r_foot_z, ground_z, foot_dist = compute_foot_contact(joints, lidar_max_z)

                                times.append(t)
                                frame_ids_list.append(f)
                                theta_l_list.append(t_l_deg)
                                theta_r_list.append(t_r_deg)
                                sym_variance_list.append(sym_variance)

                                l_foot_z_list.append(l_foot_z)
                                r_foot_z_list.append(r_foot_z)
                                ground_z_list.append(ground_z)
                                foot_error_list.append(foot_dist)

                                thesis_global_theta_l.append(t_l_deg)
                                thesis_global_theta_r.append(t_r_deg)

                                if f == onset_frame:
                                    sample_joints = joints

                    if times and sample_joints is not None:
                        print(f"\n[STEP 6] Analyzing fusion biomechanics for {short_id} at onset {onset_frame}")

                        evaluation = evaluate_biomechanical_plausibility(theta_l_list, theta_r_list, sym_variance_list,
                                                                         foot_error_list)
                        thesis_global_sequence_means.append(evaluation["mean_symmetry_variance_deg"])
                        thesis_global_foot_errors.append(evaluation["mean_foot_ground_error_m"])

                        outlier_audit_results.append({
                            "tid": tid, "onset_frame": onset_frame, **evaluation
                        })

                        if not evaluation["is_valid"]:
                            # Find the specific frame with the maximum extension to visualize
                            max_idx_l = np.argmax(theta_l_list)
                            max_idx_r = np.argmax(theta_r_list)
                            culprit_frame = frame_ids_list[max_idx_l] if theta_l_list[max_idx_l] > theta_r_list[
                                max_idx_r] else frame_ids_list[max_idx_r]

                            outliers_to_visualize.append({
                                "tid": tid,
                                "fail_frame": culprit_frame,
                                "max_extension": evaluation["max_extension_observed_deg"],
                                "mean_variance": evaluation["mean_symmetry_variance_deg"],
                                "mean_foot_error": evaluation["mean_foot_ground_error_m"]
                            })

                        plot_gait_symmetry_diagnostics(
                            tid, times, theta_l_list, theta_r_list, sym_variance_list, sample_joints, str(step6_out_dir)
                        )
                        plot_foot_contact_diagnostics(
                            tid, times, l_foot_z_list, r_foot_z_list, ground_z_list, str(step6_out_dir)
                        )

            if outlier_audit_results:
                audit_df = pd.DataFrame(outlier_audit_results)
                audit_csv_path = step6_out_dir / "dataset_outlier_audit_results.csv"
                audit_df.to_csv(audit_csv_path, index=False)
                print("\n=== OUTLIER AUDIT RESULTS ===")
                print(audit_df.to_string())

        # STEP 7: OUTLIER VISUALIZATION
        if outliers_to_visualize:
            print("\n" + "=" * 80)
            print("STEP 7: GENERATING OUTLIER VISUAL PROOFS")
            print("=" * 80)

            step7_out_dir = seq_output_base / "step7_outlier_proofs"
            step7_out_dir.mkdir(parents=True, exist_ok=True)

            for outlier in outliers_to_visualize:
                generate_outlier_collage(
                    data_dir=str(DATASET_DIR), sequence=sequence, tid=outlier["tid"], fail_frame=outlier["fail_frame"],
                    max_extension=outlier["max_extension"], mean_variance=outlier["mean_variance"],
                    mean_foot_error=outlier["mean_foot_error"], smpl_model=smpl_model, faces=faces,
                    device=device, output_dir=str(step7_out_dir)
                )

    print("\n" + "=" * 80)
    print("FINAL THESIS GRAPH GENERATION (AGGREGATED)")
    print("=" * 80)

    thesis_out_dir = OUTPUT_BASE / "global_thesis_evaluations"
    thesis_out_dir.mkdir(parents=True, exist_ok=True)

    if thesis_global_theta_l and thesis_global_theta_r:
        plot_leg_extension_distribution(thesis_global_theta_l, thesis_global_theta_r, threshold=60.0,
                                        output_dir=str(thesis_out_dir))

    if thesis_global_sequence_means:
        plot_symmetry_variance_distribution(thesis_global_sequence_means, threshold=10.0,
                                            output_dir=str(thesis_out_dir))

    if thesis_global_foot_errors:
        plot_foot_error_distribution(thesis_global_foot_errors, threshold=0.20, output_dir=str(thesis_out_dir))

    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()