import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Style to match sample.png
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})

# Data from paper tables
configs = [
    {
        'title': 'Qwen2.5-3B (In-Domain)',
        'r1p': 75.67,
        'ours': 77.02,
        'oracle': 79.12,
    },
    {
        'title': 'Qwen2.5-1.5B (In-Domain)',
        'r1p': 62.72,
        'ours': 68.10,
        'oracle': 70.47,
    },
    {
        'title': 'Qwen2.5-3B (Out-of-Domain)',
        'r1p': 64.44,
        'ours': 66.96,
        'oracle': 66.07,
    },
    {
        'title': 'Qwen2.5-1.5B (Out-of-Domain)',
        'r1p': 58.81,
        'ours': 62.81,
        'oracle': 62.00,
    },
]

# Colors matching sample.png palette
gray_color = '#808080'       # R1-p baseline (like Base gray in sample)
purple_color = '#8B2FC9'     # MemReward (like SAR purple in sample)
teal_color = '#50C4AA'       # Oracle (like ER teal in sample)
edge_gray = '#555555'
edge_purple = '#5A1A8A'
edge_teal = '#2E8B73'
dashed_color = '#C0C0C0'

marker_size = 220

fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), dpi=200)
axes = axes.flatten()

for ax, cfg in zip(axes, configs):
    r1p = cfg['r1p']
    ours = cfg['ours']
    oracle = cfg['oracle']

    # Dashed reference line from R1-p to Oracle
    ax.plot([20, 100], [r1p, oracle], '--', color=dashed_color,
            linewidth=1.8, zorder=1)

    # Markers (large, bold, matching sample.png)
    ax.scatter(20, r1p, marker='X', s=marker_size, color=gray_color,
               edgecolors=edge_gray, linewidths=1.2, zorder=3,
               label='R1-p (20% GT)')
    ax.scatter(20, ours, marker='o', s=marker_size, color=purple_color,
               edgecolors=edge_purple, linewidths=1.2, zorder=3,
               label='MemReward (Ours)')
    ax.scatter(100, oracle, marker='^', s=marker_size, color=teal_color,
               edgecolors=edge_teal, linewidths=1.2, zorder=3,
               label='R1-Oracle')

    # Title
    ax.set_title(cfg['title'], fontsize=15, fontweight='bold', pad=8)

    # Axis labels
    ax.set_xlabel('Labels (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Avg Accuracy (%)', fontsize=15, fontweight='bold')

    # X-axis
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(-5, 110)

    # Y-axis: zoom in with padding
    ymin = min(r1p, ours, oracle)
    ymax = max(r1p, ours, oracle)
    yrange = ymax - ymin
    pad = max(yrange * 0.4, 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    # Grid (visible solid lines like sample.png)
    ax.grid(True, linestyle='-', alpha=0.25, color='#AAAAAA')
    ax.set_axisbelow(True)

    # Tick styling
    ax.tick_params(axis='both', labelsize=13, width=0.8)

    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    # Legend
    ax.legend(fontsize=10, loc='lower right', frameon=True, fancybox=True,
              edgecolor='#CCCCCC', framealpha=0.9,
              markerscale=0.6, handletextpad=0.5, labelspacing=0.6)

plt.tight_layout(pad=1.5)
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotation_accuracy.png')
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()
