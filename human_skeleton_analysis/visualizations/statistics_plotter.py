import os
import matplotlib.pyplot as plt


def generate_step2_graphs(vis_stats, turn_stats, out_dir):
    print("\n[VISUALIZATION] Generating statistical graphs...")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Funnel/Bar Chart: Total -> Qualified -> Turners
    plt.figure(figsize=(10, 6))
    categories = ['Total in LiDAR', 'Visually Qualified', 'People Who Turn', 'Total Turn Events']
    counts = [vis_stats['total_lidar_pedestrians'], vis_stats['qualified_pedestrians'],
              turn_stats['people_who_turn'], turn_stats['total_turn_events']]

    bars = plt.bar(categories, counts, color=['#cccccc', '#3498db', '#e74c3c', '#9b59b6'])
    plt.title('Pedestrian Filtering & Turn Event Funnel')
    plt.ylabel('Count')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, yval, ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'filtering_funnel.png'))
    plt.close()

    # 2. Pie Chart: Multi-Camera Visibility
    plt.figure(figsize=(8, 8))
    labels = [f"Seen by {k} Cams" for k, v in vis_stats['cam_visibility_counts'].items() if v > 0]
    sizes = [v for k, v in vis_stats['cam_visibility_counts'].items() if v > 0]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=['#f1c40f', '#e67e22', '#e74c3c', '#c0392b'])
    plt.title('Camera Coverage of Pedestrians')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'camera_coverage_pie.png'))
    plt.close()
    print(f"[VISUALIZATION] Graphs saved to {out_dir}")