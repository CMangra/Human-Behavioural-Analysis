import os
import sys
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
from visualizations.smpl_video_annotator import run as run_step4_smpl_video_annotation
from visualizations.smpl_orientation_deviation_plotter import run as run_step5_orientation_deviation

# ==========================================
# PIPELINE CONFIGURATION
# ==========================================
WORKSPACE_ROOT = r"G:\My Drive\Desktop\THD\Master\JBData\3. Semester\code"
DATASET_DIR = os.path.join(WORKSPACE_ROOT, r"downloaded_stuff\datasets\pedx\pedx_data")
OUTPUT_BASE = os.path.join(WORKSPACE_ROOT,
                           r"Third-Semester-Code\pedx\visualisation_human_skeleton_visualisation_analysis")
TARGET_SEQUENCE = "20171207T2024"


def main():
    print("=" * 50)
    print("PEDX HUMAN SKELETON ANALYSIS PIPELINE")
    print("=" * 50)

    # ---------------------------------------------------------
    # STEP 2: Pedestrian Filtering & Turn Onset Detection
    # ---------------------------------------------------------
    step2_graphs_dir = os.path.join(OUTPUT_BASE, "step2_pedestrian_filtering", "analysis_graphs")
    step2_frames_dir = os.path.join(OUTPUT_BASE, "step2_pedestrian_filtering", "frames")

    # NEW FOLDER FOR MATH DEBUGGER
    step2_math_debug_dir = os.path.join(OUTPUT_BASE, "step2_pedestrian_filtering", "math_debug_graphs")

    # A. Filter by Visibility
    qualified_trajectories, vis_stats = filter_pedestrians_by_visibility(DATASET_DIR, TARGET_SEQUENCE)

    # B. Generate the Debug Graphs BEFORE the turn math drops bad data
    generate_kinematic_debug_graphs(qualified_trajectories, step2_math_debug_dir)

    # C. Kinematic Onset Math (NOW FIXED)
    turn_results, turn_stats = detect_multiple_turns_with_onset(qualified_trajectories)

    # D. Plot Statistics
    generate_step2_graphs(vis_stats, turn_stats, step2_graphs_dir)

    # E. Render Composite Summaries
    generate_qualified_summaries(DATASET_DIR, TARGET_SEQUENCE, qualified_trajectories, turn_results, step2_frames_dir)

    # ---------------------------------------------------------
    # STEP 3: Behavioral Correlation Analysis
    # ---------------------------------------------------------
    print("\n[STEP 3] Analyzing Skeleton Behavior prior to Onsets...")
    step3_out_dir = os.path.join(OUTPUT_BASE, "step3_behavioral_correlation")

    for tid, onsets in turn_results.items():
        for onset in onsets:
            metrics = extract_skeleton_metrics(DATASET_DIR, TARGET_SEQUENCE, tid, onset, config.CAMERAS)
            if metrics:
                plot_behavioral_correlation(tid, onset, metrics, step3_out_dir)
                print(f"  -> Generated correlation graph for ID {tid[-4:]} at frame {onset}")
            else:
                print(f"  -> Skipped ID {tid[-4:]} at frame {onset} (Insufficient 2D skeleton data)")

    print("\n=== PIPELINE COMPLETE ===")

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
    # STEP 5: 3D SMPL Orientation Deviation Analysis
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: 3D SMPL ORIENTATION DEVIATION VISUALISATION")
    print("=" * 80)

    run_step5_orientation_deviation(
        data_dir=DATASET_DIR,
        sequence=TARGET_SEQUENCE,
        qualified_trajectories=qualified_trajectories,
        turn_results=turn_results,
        output_base=OUTPUT_BASE,
        pre_onset_seconds=4.0,
        post_onset_seconds=3.0,
        baseline_start_s=-3.0,
        baseline_end_s=-1.5,
    )


if __name__ == "__main__":
    main()
