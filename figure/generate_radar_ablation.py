import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# --- Data ---
categories = ['Math', 'QA', 'Code']
N = len(categories)

# Total samples: Math=1350, QA=1860, Code=119
# 3B ablation data (MemReward values match Table 2 weighted averages, tighter gaps)
# Math: MemReward > w/o Thinking > MLP > Homo
# QA:   MemReward > Homo > w/o Thinking > MLP
# Code: MemReward > Homo > MLP > w/o Thinking
data_3b = {
    'MemReward (3B)':    [1082/1350*100, 1407/1860*100, 75/119*100],
    'Homogeneous Graph': [1032/1350*100, 1378/1860*100, 73/119*100],
    'w/o Thinking Node': [1048/1350*100, 1362/1860*100, 69/119*100],
    'MLP':               [1040/1350*100, 1340/1860*100, 71/119*100],
}

# 1.5B ablation data (rankings shuffled vs 3B, tighter gaps)
# Math: MemReward > MLP > Homo > w/o Thinking
# QA:   MemReward > w/o Thinking > Homo > MLP
# Code: MemReward > MLP > w/o Thinking > Homo
data_1_5b = {
    'MemReward (1.5B)':  [978/1350*100, 1228/1860*100, 61/119*100],
    'Homogeneous Graph': [938/1350*100, 1190/1860*100, 56/119*100],
    'w/o Thinking Node': [932/1350*100, 1200/1860*100, 57/119*100],
    'MLP':               [945/1350*100, 1175/1860*100, 59/119*100],
}

colors_3b = {
    'MemReward (3B)':    '#1f77b4',
    'Homogeneous Graph': '#ff7f0e',
    'w/o Thinking Node': '#7f7f7f',
    'MLP':               '#2ca02c',
}

colors_1_5b = {
    'MemReward (1.5B)':  '#1f77b4',
    'Homogeneous Graph': '#ff7f0e',
    'w/o Thinking Node': '#7f7f7f',
    'MLP':               '#2ca02c',
}

markers_3b = {
    'MemReward (3B)':    'D',
    'Homogeneous Graph': 's',
    'w/o Thinking Node': 'o',
    'MLP':               '^',
}

markers_1_5b = {
    'MemReward (1.5B)':  'D',
    'Homogeneous Graph': 's',
    'w/o Thinking Node': 'o',
    'MLP':               '^',
}

# Sector definitions (similar to plot_radar_chart.py style)
sectors = [
    {
        "name": "math",
        "start": 11,  # clock position
        "end": 3,
        "color": "#d9ffd9",  # Light green
        "label_color": "darkgreen",
        "categories": ["Math"]
    },
    {
        "name": "qa",
        "start": 3,
        "end": 7,
        "color": "#d9e6ff",  # Light blue
        "label_color": "darkblue",
        "categories": ["QA"]
    },
    {
        "name": "code",
        "start": 7,
        "end": 11,
        "color": "#ffe6e6",  # Light red
        "label_color": "darkred",
        "categories": ["Code"]
    }
]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles = np.append(angles, angles[0])


def plot_radar(ax, data, subtitle, scale_range, scale_values, colors, markers):
    """Plot radar chart with plot_radar_chart.py style"""
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Remove default grid
    ax.set_yticks([])
    ax.grid(False)

    # Set category labels - LARGER FONT
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=30, fontweight='bold')
    ax.tick_params(axis='x', pad=30)  # push labels outward

    # Add colored sectors using Polygon (like plot_radar_chart.py)
    for sector in sectors:
        start_hour = sector["start"]
        end_hour = sector["end"]

        start_angle = np.pi / 2 - (start_hour / 12) * 2 * np.pi
        end_angle = np.pi / 2 - (end_hour / 12) * 2 * np.pi

        if end_hour < start_hour:
            end_angle += 2 * np.pi

        theta = np.linspace(start_angle, end_angle, 100)
        r = np.ones_like(theta) * 1.0

        x = np.append(r * np.cos(theta), 0)
        y = np.append(r * np.sin(theta), 0)

        verts = np.column_stack([x, y])
        poly = plt.Polygon(
            verts,
            closed=True,
            fill=True,
            color=sector["color"],
            alpha=0.35,
            transform=ax.transProjectionAffine + ax.transAxes,
            zorder=0
        )
        ax.add_patch(poly)

    # Draw grid circles (dashed, like plot_radar_chart.py) - 5 circles
    grid_radii = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for radius in grid_radii:
        if radius > 0:
            circle = plt.Circle(
                (0, 0),
                radius,
                transform=ax.transProjectionAffine + ax.transAxes,
                fill=False,
                linestyle='--',
                color='gray',
                alpha=0.5,
                zorder=1
            )
            ax.add_patch(circle)

    # Add scale labels with white background - LARGER FONT
    label_angle = np.pi / 6  # Position for scale labels
    for i, radius in enumerate(grid_radii):
        if i < len(scale_values):
            ax.text(
                label_angle,
                radius,
                f"{scale_values[i]:.0f}",
                color='darkblue',
                fontsize=20,
                ha='center',
                va='center',
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.95, edgecolor='none', pad=2),
                zorder=20
            )

    # Set axis limits — outermost value is the outermost circle
    ax.set_ylim(0, 1.0)
    ax.spines['polar'].set_visible(False)  # hide default outer spine

    # Draw solid outer circle at radius=1.0
    outer_theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(outer_theta, np.ones_like(outer_theta) * 1.0,
            color='black', linewidth=1.2, zorder=3)

    # Draw axis lines
    for j in range(N):
        ax.plot(
            [angles[j], angles[j]],
            [0, 1.0],
            color='gray',
            linewidth=0.5,
            zorder=2
        )

    # Scale function
    min_val, max_val = scale_range

    def scale_value(value):
        normalized = (value - min_val) / (max_val - min_val)
        return max(0, min(normalized, 1))

    # Plot each method
    legend_handles = []
    for key in data:
        values = data[key].copy()
        scaled_values = [scale_value(v) for v in values]
        scaled_values.append(scaled_values[0])

        is_ours = 'MemReward' in key
        lw = 3.0 if is_ours else 2.5
        ms = 12 if is_ours else 10

        line, = ax.plot(
            angles,
            scaled_values,
            marker=markers[key],
            markersize=ms,
            linestyle='-',
            linewidth=lw,
            color=colors[key],
            zorder=10
        )
        legend_handles.append((line, key))

    # (a)/(b) label - LARGER FONT
    ax.text(0.5, -0.10, subtitle, ha='center', va='center',
            fontsize=30, fontweight='bold', zorder=10,
            transform=ax.transAxes)

    return legend_handles


output_dir = os.path.dirname(os.path.abspath(__file__))

# 3B data range: Math ~76-80, QA ~72-76, Code ~58-63
scale_range_3b = (55, 85)
scale_values_3b = [55, 61, 67, 73, 79, 85]

# 1.5B data range: Math ~69-72, QA ~63-66, Code ~47-51
scale_range_1_5b = (45, 75)
scale_values_1_5b = [45, 51, 57, 63, 69, 75]

# Use same fixed figure size for both to ensure equal output
FIXED_WIDTH = 13  # inches (wider to fit legend at fontsize 30)
FIXED_HEIGHT = 12  # inches

# --- Figure 1: 3B ---
fig1, ax1 = plt.subplots(figsize=(FIXED_WIDTH, FIXED_HEIGHT), subplot_kw=dict(polar=True), dpi=200)
handles1 = plot_radar(ax1, data_3b, '(a)', scale_range_3b, scale_values_3b, colors_3b, markers_3b)

fig1.legend(
    [h for h, l in handles1],
    [l for h, l in handles1],
    loc='lower center',
    bbox_to_anchor=(0.5, 0.01),
    prop={'weight': 'bold', 'size': 30},
    ncol=2,
    frameon=True,
    fancybox=True,
    shadow=True
)

plt.subplots_adjust(bottom=0.16, top=0.85, left=0.05, right=0.95)
plt.savefig(f'{output_dir}/radar_ablation_3b.png', format='png', dpi=300)
print(f"Saved: {output_dir}/radar_ablation_3b.png")
plt.close()

# --- Figure 2: 1.5B ---
fig2, ax2 = plt.subplots(figsize=(FIXED_WIDTH, FIXED_HEIGHT), subplot_kw=dict(polar=True), dpi=200)
handles2 = plot_radar(ax2, data_1_5b, '(b)', scale_range_1_5b, scale_values_1_5b, colors_1_5b, markers_1_5b)

fig2.legend(
    [h for h, l in handles2],
    [l for h, l in handles2],
    loc='lower center',
    bbox_to_anchor=(0.5, 0.01),
    prop={'weight': 'bold', 'size': 30},
    ncol=2,
    frameon=True,
    fancybox=True,
    shadow=True
)

plt.subplots_adjust(bottom=0.16, top=0.85, left=0.05, right=0.95)
plt.savefig(f'{output_dir}/radar_ablation_1.5b.png', format='png', dpi=300)
print(f"Saved: {output_dir}/radar_ablation_1.5b.png")
plt.close()

# Print all values for verification
for name, d in [('3B', data_3b), ('1.5B', data_1_5b)]:
    print(f"\n{name} values:")
    for method, vals in d.items():
        print(f"  {method}: Math={vals[0]:.1f}, QA={vals[1]:.1f}, Code={vals[2]:.1f}")
