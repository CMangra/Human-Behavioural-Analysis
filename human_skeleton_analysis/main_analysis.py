import os
import sys
from pathlib import Path

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
from visualizations.symmetry_diagnostics import plot_gait_symmetry_diagnostics, generate_gait_symmetry_video
from data_analysis.smpl_orientation_metrics import smpl_forward_joints, build_smpl_json_path
from visualizations.smpl_video_annotator import load_smpl_params, load_smpl_model
import torch
import pandas as pd

# ==========================================
# PIPELINE CONFIGURATION
# ==========================================
WORKSPACE_ROOT = Path(r"G:\My Drive\Desktop\THD\Master\JBData\3. Semester\code")
REPO_ROOT = WORKSPACE_ROOT / r"Third-Semester-Code\pedx"
DATASET_DIR = WORKSPACE_ROOT / r"downloaded_stuff\datasets\pedx\pedx_data"
MODEL_ROOT = REPO_ROOT / "body_models"
OUTPUT_BASE = REPO_ROOT / r"visualisation_human_skeleton_visualisation_analysis"
TARGET_SEQUENCE = "20171207T2024"


def main():
    print("=" * 50)
    print("PEDX HUMAN SKELETON ANALYSIS PIPELINE")
    print("=" * 50)

    # ---------------------------------------------------------
    # STEP 2: Pedestrian Filtering & Turn Onset Detection
    # ---------------------------------------------------------
    step2_graphs_dir = OUTPUT_BASE / "step2_pedestrian_filtering" / "analysis_graphs"
    step2_frames_dir = OUTPUT_BASE / "step2_pedestrian_filtering" / "frames"
    step2_math_debug_dir = OUTPUT_BASE / "step2_pedestrian_filtering" / "math_debug_graphs"

    qualified_trajectories, vis_stats = filter_pedestrians_by_visibility(str(DATASET_DIR), TARGET_SEQUENCE)
    generate_kinematic_debug_graphs(qualified_trajectories, str(step2_math_debug_dir))
    turn_results, turn_stats = detect_multiple_turns_with_onset(qualified_trajectories)
    generate_step2_graphs(vis_stats, turn_stats, str(step2_graphs_dir))
    generate_qualified_summaries(str(DATASET_DIR), TARGET_SEQUENCE, qualified_trajectories, turn_results,
                                 str(step2_frames_dir))

    # ---------------------------------------------------------
    # STEP 3: Behavioral Correlation Analysis
    # ---------------------------------------------------------
    print("\n[STEP 3] Analyzing Skeleton Behavior prior to Onsets...")
    step3_out_dir = OUTPUT_BASE / "step3_behavioral_correlation"
    for tid, onsets in turn_results.items():
        for onset in onsets:
            metrics = extract_skeleton_metrics(str(DATASET_DIR), TARGET_SEQUENCE, tid, onset, config.CAMERAS)
            if metrics:
                plot_behavioral_correlation(tid, onset, metrics, str(step3_out_dir))

    # Step 4
    print("\n" + "=" * 80)
    print("STEP 4: OFFICIAL PEDX SMPL VIDEO ANNOTATION")
    print("=" * 80)

    #run_step4_smpl_video_annotation(
    #    sequence="20171207T2024",
    #    camera="blu79CF",
    #    fps=10,
    #    max_frames=None,
    #)


    # ---------------------------------------------------------
    # STEP 5.1: Dataset Filtering & SMPL Audit
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5.1: DATASET SMPL AVAILABILITY AUDIT")
    print("=" * 80)

    frame_to_time = load_frame_timestamps(DATASET_DIR, TARGET_SEQUENCE)
    audit_out_dir = OUTPUT_BASE / "step5_dataset_filtering_results"

    usable_turn_results, event_df = run_dataset_audit(
        data_dir=str(DATASET_DIR),
        sequence=TARGET_SEQUENCE,
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
            sequence=TARGET_SEQUENCE,
            qualified_trajectories=qualified_trajectories,
            turn_results=usable_turn_results,  # PASSING ONLY USABLE TURNS
            output_base=str(OUTPUT_BASE),
            model_root=Path(MODEL_ROOT)
        )

    # ---------------------------------------------------------
    # STEP 6: Biomechanical Gait Symmetry & Outlier Detection
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6: BIOMECHANICAL GAIT SYMMETRY ANALYSIS (BIO-LSTM)")
    print("=" * 80)

    step6_out_dir = OUTPUT_BASE / "step6_biomechanics_diagnostics"
    step6_out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    smpl_model, _ = load_smpl_model(MODEL_ROOT, device)

    outlier_audit_results = []

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
                        json_path = build_smpl_json_path(str(DATASET_DIR), TARGET_SEQUENCE, f, tid)
                        if os.path.exists(json_path):
                            betas, pose, trans = load_smpl_params(json_path)
                            joints = smpl_forward_joints(smpl_model, betas, pose, trans, device)

                            t_l_deg, t_r_deg, cos_l, cos_r, bio_delta = compute_leg_symmetry_bio_lstm(joints)

                            times.append(t)
                            theta_l_list.append(t_l_deg)
                            theta_r_list.append(t_r_deg)
                            bio_lstm_deltas.append(bio_delta)
                            all_joints.append(joints)

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
                    # Note: You may need to slightly adjust symmetry_diagnostics.py to label the delta graph as "Cosine Delta" instead of Degrees
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

        # Output the final audit table
        audit_df = pd.DataFrame(outlier_audit_results)
        audit_csv_path = step6_out_dir / "dataset_outlier_audit_results.csv"
        audit_df.to_csv(audit_csv_path, index=False)
        print("\n=== OUTLIER AUDIT RESULTS ===")
        print(audit_df.to_string())
    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()