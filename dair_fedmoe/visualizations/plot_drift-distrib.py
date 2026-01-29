import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from global_config import SAVED_VISUALIZATIONS_DIR


plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 20
})


def plot_drift_distrib(datasets, alphas, K, N_per_client, save_file):

    datasets = {
        'ISCX-VPN': 12,
        'ISCX-Tor': 16,
        'VNAT': 8,
        'USTC-TFC2016': 20
    }
    alphas = [0.1, 0.5]
    K = 20
    N_per_client = 3000

    fig, axes = plt.subplots(len(datasets), len(alphas), figsize=(12, 16))
    for i, (ds_name, num_classes) in enumerate(datasets.items()):
        for j, alpha in enumerate(alphas):
            # Sample Dirichlet proportions and counts
            proportions = np.random.dirichlet([alpha] * num_classes, size=K)
            counts = proportions * N_per_client
            
            ax = axes[i, j]
            # Display heatmap: classes as rows, clients as columns
            im = ax.imshow(counts.T, aspect='auto', cmap='managua')
            ax.set_title(f'{ds_name} under Dir({alpha})')
            ax.set_xlabel('Clients')
            ax.set_ylabel('Classes')
            ax.set_xticks(np.arange(K))
            ax.set_yticks(np.arange(num_classes))
            fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    datasets = {
        'ISCX-VPN': 12,
        'ISCX-Tor': 16,
        'VNAT': 8,
        'USTC-TFC2016': 20
    }

    alphas = [0.1, 0.5]
    K = 20
    N_per_client = 3000
    save_file = os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-drif-distrib-plot.pdf')

    plot_drift_distrib(datasets, alphas, K, N_per_client, save_file)

