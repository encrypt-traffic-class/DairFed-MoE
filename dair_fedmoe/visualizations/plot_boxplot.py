import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from global_config import SAVED_EVALS_DIR, SAVED_VISUALIZATIONS_DIR
import pandas as pd
from global_utils import load_boxplot_data

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 20
})

def plot_boxplot(title, data_file, save_file):
    data = load_boxplot_data(data_file)
    dataset_keys = ['ISCX-VPN', 'ISCX-Tor', 'VNAT', 'USTC-TFC2016']
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create boxplot
    box_colors = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF']
    bp = ax.boxplot([group['Macro-F1'].values for name, group in df.groupby('Dataset')],
                    patch_artist=True,
                    medianprops=dict(color="red", linewidth=4),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=8),
                    widths=0.8,
                    showfliers=False)

    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for idx, (name, group) in enumerate(df.groupby('Dataset')):
        # Add jitter to x-position
        x = np.random.normal(idx + 1, 0.04, size=len(group))
        ax.scatter(x, group['Macro-F1'], color='gray', alpha=0.6, s=100)

    ax.set_title(title, pad=0)
    ax.set_ylim(0.915, 1.0)
    ax.set_xlabel('')
    ax.set_ylabel('Macro-F1 Score')
    ax.set_xticklabels(dataset_keys)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()

if __name__=='__main__':
    plot_boxplot(title='Client-wise Macro-F1 Distribution at Convergence\nacross Four Datasets', 
                 data_file=os.path.join(SAVED_EVALS_DIR, 'boxplot_data_iscx_vpn.npz'), 
                 save_file=os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-boxplot.pdf'))
