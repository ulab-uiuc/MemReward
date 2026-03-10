import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})

# Data: MemReward vs R1-p on Qwen2.5-1.5B (from Table 1 & Table 2)
benchmarks = [
    # (name, category, memreward, r1p)
    # Math (In-Domain)
    ('GSM8K',       'Math', 88.67, 77.11),
    ('GSM-Sym',     'Math', 77.78, 62.89),
    ('MATH',        'Math', 50.89, 44.44),
    # QA (In-Domain)
    ('MMLU',        'QA',   54.67, 53.33),
    ('CSQA',        'QA',   72.44, 70.22),
    ('OBQA',        'QA',   70.00, 68.67),
    ('ARC-C',       'QA',   72.67, 71.56),
    ('GPQA',        'QA',   23.33, 20.00),
    # Code (In-Domain)
    ('HumanEval+',  'Code', 43.59, 38.46),
    ('MBPP+',       'Code', 55.00, 55.00),
    # Out-of-Domain
    ('NuminaMath',  'OOD',  34.67, 31.56),
    ('SIQA',        'OOD',  74.44, 72.67),
    ('PIQA',        'OOD',  79.33, 72.22),
]

names = [b[0] for b in benchmarks]
categories = [b[1] for b in benchmarks]
deltas = [b[2] - b[3] for b in benchmarks]

# Sort by delta descending
sorted_indices = np.argsort(deltas)  # ascending for horizontal (bottom to top)
names = [names[i] for i in sorted_indices]
categories = [categories[i] for i in sorted_indices]
deltas = [deltas[i] for i in sorted_indices]

# Category colors
cat_colors = {
    'Math': '#2ECC71',   # green
    'QA':   '#3498DB',   # blue
    'Code': '#E74C3C',   # red
    'OOD':  '#8B2FC9',   # purple
}

colors = [cat_colors[c] for c in categories]

fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

y_pos = np.arange(len(names))
bars = ax.barh(y_pos, deltas, height=0.65, color=colors, edgecolor='white',
               linewidth=0.5, zorder=3)

# Add value labels at bar ends
for i, (val, bar) in enumerate(zip(deltas, bars)):
    if val >= 0:
        ax.text(val + 0.15, i, f'+{val:.1f}', va='center', ha='left',
                fontsize=10, fontweight='bold', color='#333333')
    else:
        ax.text(val - 0.15, i, f'{val:.1f}', va='center', ha='right',
                fontsize=10, fontweight='bold', color='#333333')

# Vertical zero line
ax.axvline(x=0, color='#333333', linewidth=0.8, zorder=2)

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=11, fontweight='bold')
ax.set_xlabel('MemReward $-$ R1-p (Accuracy Delta)', fontsize=11, fontweight='bold')

# Grid (vertical only)
ax.grid(True, axis='x', linestyle='-', alpha=0.2, color='#AAAAAA')
ax.set_axisbelow(True)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

ax.tick_params(axis='both', labelsize=11, width=0.8)

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=cat_colors['Math'], label='Math'),
    Patch(facecolor=cat_colors['QA'],   label='QA'),
    Patch(facecolor=cat_colors['Code'], label='Code'),
    Patch(facecolor=cat_colors['OOD'],  label='Out-of-Domain'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right',
          frameon=True, fancybox=True, edgecolor='#CCCCCC', framealpha=0.9)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_delta.png')
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()
