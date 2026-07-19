import os
import sys
from pathlib import Path
import config

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualizations.dataset_video_generator import pre_process_and_visualize_dataset
from data_analysis.pedestrian_filtering import filter_pedestrians_by_visibility
from data_analysis.turn_detection import detect_multiple_turns_with_onset
from visualizations.statistics_plotter import generate_step2_graphs
from visualizations.qualified_pedestrian_visualizer import generate_qualified_summaries
from visualizations.turn_math_debugger import generate_kinematic_debug_graphs
from data_analysis.skeleton_metrics import extract_skeleton_metrics
from visualizations.correlation_plotter import plot_behavioral_correlation

# New Imports for Step 5
from data_analysis.timestamp_utils import load_frame_timestamps
from data_analysis.smpl_audit import run_dataset_audit
from visualizations.separate_orientation_diagnostics import run as run_step5_separate_diagnostics

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

    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()