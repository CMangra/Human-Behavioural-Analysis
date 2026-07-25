import os
import sys
from pathlib import Path
import torch
import pandas as pd

# Add the parent directory to sys.path
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

# Imports for Step 6
from data_analysis.gait_biomechanics import compute_leg_symmetry_bio_lstm
from data_analysis.outlier_elimination import evaluate_biomechanical_plausibility
from visualizations.symmetry_diagnostics import plot_gait_symmetry_diagnostics, generate_gait_symmetry_video, \
    plot_leg_extension_distribution, plot_cosine_delta_distribution
from data_analysis.smpl_orientation_metrics import smpl_forward_joints, build_smpl_json_path
from visualizations.smpl_video_annotator import load_smpl_params, load_smpl_model

# ==========================================
# ENVIRONMENT CONFIGURATION (LOCAL vs SERVER)
# ==========================================
ENVIRONMENT = "SERVER"  # Change to "LOCAL" when running on your machine

if ENVIRONMENT == "SERVER":
    WORKSPACE_ROOT = Path("/workspace")
    REPO_ROOT = WORKSPACE_ROOT / "Human-Behavioural-Analysis"
    DATASET_DIR = WORKSPACE_ROOT / "datasets" / "pedx"
    MODEL_ROOT = REPO_ROOT / "body_models"
    OUTPUT_BASE = REPO_ROOT / "visualisation_human_skeleton_visualisation_analysis"
    TARGET_SEQUENCES = ["20171207T2024", "20171130T2000"]

elif ENVIRONMENT == "LOCAL":
    WORKSPACE_ROOT = Path(r"G:\My Drive\Desktop\THD\Master\JBData\3. Semester\code")
    REPO_ROOT = WORKSPACE_ROOT / r"Third-Semester-Code\pedx"
    DATASET_DIR = WORKSPACE_ROOT / r"downloaded_stuff\datasets\pedx\pedx_data"
    MODEL_ROOT = REPO_ROOT / "body_models"
    OUTPUT_BASE = REPO_ROOT / r"visualisation_human_skeleton_visualisation_analysis"
    TARGET_SEQUENCES = ["20171207T2024"]  # Add more sequences if available locally

else:
    raise ValueError("Invalid ENVIRONMENT setting. Must be 'LOCAL' or 'SERVER'.")


def main():
    print("=" * 50)
    print("PEDX HUMAN SKELETON ANALYSIS PIPELINE")
    print("=" * 50)

    for sequence in TARGET_SEQUENCES:
        print(f"\n=== PROCESSING SEQUENCE: {sequence} ===")

        # Create sequence-specific output base to avoid overwriting files
        seq_output_base = OUTPUT_BASE / sequence

        # ---------------------------------------------------------
        # STEP 2: Pedestrian Filtering & Turn Onset Detection
        # ---------------------------------------------------------
        step2_graphs_dir = seq_output_base / "step2_pedestrian_filtering" / "analysis_graphs"
        step2_frames_dir = seq_output_base / "step2_pedestrian_filtering" / "frames"
        step2_math_debug_dir = seq_output_base / "step2_pedestrian_filtering" / "math_debug_graphs"

        qualified_trajectories, vis_stats = filter_pedestrians_by_visibility(str(DATASET_DIR), sequence)
        generate_kinematic_debug_graphs(qualified_trajectories, str(step2_math_debug_dir))
        turn_results, turn_stats = detect_multiple_turns_with_onset(qualified_trajectories)
        generate_step2_graphs(vis_stats, turn_stats, str(step2_graphs_dir))
        generate_qualified_summaries(str(DATASET_DIR), sequence, qualified_trajectories, turn_results,
                                     str(step2_frames_dir))

        # ---------------------------------------------------------
        # STEP 3: Behavioral Correlation Analysis
        # ---------------------------------------------------------
        print("\n[STEP 3] Analyzing Skeleton Behavior prior to Onsets...")
        step3_out_dir = seq_output_base / "step3_behavioral_correlation"
        for tid, onsets in turn_results.items():
            for onset in onsets:
                metrics = extract_skeleton_metrics(str(DATASET_DIR), sequence, tid, onset, config.CAMERAS)
                if metrics:
                    plot_behavioral_correlation(tid, onset, metrics, str(step3_out_dir))

        # ---------------------------------------------------------
        # STEP 4: OFFICIAL PEDX SMPL VIDEO ANNOTATION (Commented)
        # ---------------------------------------------------------
        print("\n" + "=" * 80)
        print("STEP 4: OFFICIAL PEDX SMPL VIDEO ANNOTATION")
        print("=" * 80)

        # ---------------------------------------------------------
        # STEP 5.1: Dataset Filtering & SMPL Audit
        # ---------------------------------------------------------
        print("\n" + "=" * 80)
        print("STEP 5.1: DATASET SMPL AVAILABILITY AUDIT")
        print("=" * 80)

        frame_to_time = load_frame_timestamps(DATASET_DIR, sequence)
        audit_out_dir = seq_output_base / "step5_dataset_filtering_results"

        usable_turn_results, event_df = run_dataset_audit(
            data_dir=str(DATASET_DIR),
            sequence=sequence,
            qualified_trajectories=qualified_trajectories,
            turn_results=turn_results,
            frame_to_time=frame_to_time,
            output_dir=str(audit_out_dir)
        )

        # ---------------------------------------------------------
        # STEP 5.2: Separate Figures Direction Diagnostics
        # ---------------------------------------------------------
        print("\n" + "=" * 80)
        print("STEP 5.2: 3D SMPL ORIENTATION SENSITIVITY VISUALISATION")
        print("=" * 80)

        if not usable_turn_results:
            print("[WARNING] No usable turn events passed the SMPL audit. Skipping Step 5.2.")
        else:
            run_step5_separate_diagnostics(
                data_dir=str(DATASET_DIR),
                sequence=sequence,
                qualified_trajectories=qualified_trajectories,
                turn_results=usable_turn_results,
                output_base=str(seq_output_base),
                model_root=Path(MODEL_ROOT)
            )

        # ---------------------------------------------------------
        # STEP 6: Biomechanical Gait Symmetry & Outlier Detection
        # ---------------------------------------------------------
        print("\n" + "=" * 80)
        print("STEP 6: BIOMECHANICAL GAIT SYMMETRY ANALYSIS (BIO-LSTM)")
        print("=" * 80)

        step6_out_dir = seq_output_base / "step6_biomechanics_diagnostics"
        step6_out_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cpu")
        smpl_model, _ = load_smpl_model(MODEL_ROOT, device)

        outlier_audit_results = []

        # Track global metrics for the distribution graphs
        global_theta_l = []
        global_theta_r = []
        global_cosine_deltas = []

        if not usable_turn_results:
            print("[WARNING] No usable turns for Step 6.")
        else:
            for tid, onsets in usable_turn_results.items():
                short_id = tid[-4:]
                frames_dict = qualified_trajectories[tid]
                kinematics = compute_kinematics(frames_dict)

                for onset_frame in onsets:
                    times, theta_l_list, theta_r_list, bio_lstm_deltas, all_joints = [], [], [], [], []
                    sample_joints = None

                    for f in kinematics["sorted_frames"]:
                        t = relative_time_seconds(f, onset_frame, frame_to_time)
                        if -4.0 <= t <= 3.0:
                            json_path = build_smpl_json_path(str(DATASET_DIR), sequence, f, tid)
                            if os.path.exists(json_path):
                                betas, pose, trans = load_smpl_params(json_path)
                                joints = smpl_forward_joints(smpl_model, betas, pose, trans, device)

                                t_l_deg, t_r_deg, cos_l, cos_r, bio_delta = compute_leg_symmetry_bio_lstm(joints)

                                times.append(t)
                                theta_l_list.append(t_l_deg)
                                theta_r_list.append(t_r_deg)
                                bio_lstm_deltas.append(bio_delta)
                                all_joints.append(joints)

                                # Append to the sequence's global lists
                                global_theta_l.append(t_l_deg)
                                global_theta_r.append(t_r_deg)
                                global_cosine_deltas.append(bio_delta)

                                if f == onset_frame:
                                    sample_joints = joints

                    if times and sample_joints is not None:
                        print(f"\n[STEP 6] Analyzing gait symmetry for {short_id} at onset {onset_frame}")

                        # 1. Run the Outlier Evaluation
                        evaluation = evaluate_biomechanical_plausibility(theta_l_list, theta_r_list, bio_lstm_deltas)

                        outlier_audit_results.append({
                            "tid": tid,
                            "onset_frame": onset_frame,
                            **evaluation
                        })

                        # 2. Visualizations
                        plot_gait_symmetry_diagnostics(
                            tid, times, theta_l_list, theta_r_list, bio_lstm_deltas, sample_joints, str(step6_out_dir)
                        )
                        generate_gait_symmetry_video(
                            tid, onset_frame, times, theta_l_list, theta_r_list, bio_lstm_deltas, all_joints,
                            str(step6_out_dir)
                        )

                        # 3. CSV Export
                        df_sym = pd.DataFrame({
                            "time_s": times, "theta_L_deg": theta_l_list, "theta_R_deg": theta_r_list,
                            "bio_lstm_cosine_delta": bio_lstm_deltas
                        })
                        df_sym.to_csv(step6_out_dir / f"bio_lstm_symmetry_{short_id}_onset_{onset_frame}.csv",
                                      index=False)

            # Output the final audit table for this sequence
            if outlier_audit_results:
                audit_df = pd.DataFrame(outlier_audit_results)
                audit_csv_path = step6_out_dir / "dataset_outlier_audit_results.csv"
                audit_df.to_csv(audit_csv_path, index=False)
                print("\n=== OUTLIER AUDIT RESULTS ===")
                print(audit_df.to_string())

            # Output the global distribution plots for this sequence
            if global_theta_l and global_theta_r and global_cosine_deltas:
                print(f"\n[STEP 6] Generating Global Distribution Plots for Sequence {sequence}...")
                plot_leg_extension_distribution(
                    global_theta_l,
                    global_theta_r,
                    threshold=60.0,
                    output_dir=str(step6_out_dir)
                )
                plot_cosine_delta_distribution(
                    global_cosine_deltas,
                    threshold=0.15,
                    output_dir=str(step6_out_dir)
                )

    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
