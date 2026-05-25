import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualizations.dataset_video_generator import pre_process_and_visualize_dataset
from data_analysis.pedestrian_filtering import filter_pedestrians_by_visibility
from data_analysis.turn_detection import detect_multiple_turns_with_onset
from visualizations.statistics_plotter import generate_step2_graphs
from visualizations.qualified_pedestrian_visualizer import generate_qualified_summaries

# NEW IMPORT
from visualizations.turn_math_debugger import generate_kinematic_debug_graphs

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


if __name__ == "__main__":
    main()