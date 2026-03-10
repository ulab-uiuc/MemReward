import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Data from Table 4 (GT ratio experiment on Qwen2.5-3B)
gt_ratios = ['20%', '30%', '40%', '50%', '60%', '70%', '100%\n(Oracle)']
avg_scores = [77.02, 77.56, 77.53, 77.95, 78.19, 78.64, 79.12]
oracle_score = 79.12

# Colors: gradient from MemReward purple to Oracle green
colors = ['#9B59B6', '#8E6FB8', '#7F85BA', '#6BA5B5', '#5EB5B0', '#3EBE8E', '#2ECC71']

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)

# Style
plt.rcParams['font.family'] = 'DejaVu Sans'
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color('gray')

bars = ax.bar(gt_ratios, avg_scores, color=colors, edgecolor='white',
              linewidth=1.5, width=0.55, alpha=0.9)

# Oracle reference line
ax.axhline(y=oracle_score, color='#2ECC71', linestyle='--', linewidth=2,
           alpha=0.7, label='Oracle (100% GT)', zorder=1)

# Add value labels on top of bars
for bar, score in zip(bars, avg_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
            f'{score:.2f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='black')

# Axis labels
ax.set_ylabel('Average Score', fontsize=18, fontweight='bold')
ax.set_xlabel('Ground-Truth Label Ratio', fontsize=18, fontweight='bold')

# Y-axis range: zoom in to show differences
ax.set_ylim(76.0, 80.0)

# Tick styling
ax.tick_params(axis='both', labelsize=14)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Grid
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='lightgray', alpha=0.7)

# Legend
ax.legend(fontsize=14, loc='upper right', frameon=True, fancybox=True, shadow=True)

fig.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gt_ratio_bar.png')
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()
