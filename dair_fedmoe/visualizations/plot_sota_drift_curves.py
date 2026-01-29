import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from global_utils import load_macro_f1_values
from global_config import SAVED_EVALS_DIR, SAVED_VISUALIZATIONS_DIR

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 20
})


def plot_sota_drift_curves(macro_f1_ours, macro_f1_sota_1, macro_f1_sota_2,macro_f1_sota_3, macro_f1_sota_4, title, T = 250, drift_rounds = [50, 100, 150, 200], save_file = 'fig-drift-sota-1.pdf'):

    rounds = np.arange(1, T + 1)   
    fig, ax = plt.subplots(figsize=(12, 7))

    # Macro-F1 line with different styles and markers at every 10 rounds
    marker_interval = 10
    marker_indices = np.arange(0, T, marker_interval)

    ax.plot(rounds, macro_f1_ours, color='tab:blue', linewidth=2, label='DAIR-FedMoE',
            linestyle='-', marker='o', markevery=marker_indices, markersize=6)
    ax.plot(rounds, macro_f1_sota_1, color='tab:green', linewidth=2, label='FedCCFA',
            linestyle='-', marker='o', markevery=marker_indices, markersize=6)
    ax.plot(rounds, macro_f1_sota_2, color='tab:orange', linewidth=2, label='FedDrift',
            linestyle='-.', marker='o', markevery=marker_indices, markersize=6)
    ax.plot(rounds, macro_f1_sota_3, color='tab:gray', linewidth=2, label='FedIBD',
            linestyle='-', marker='o', markevery=marker_indices, markersize=6)
    ax.plot(rounds, macro_f1_sota_4, color='tab:purple', linewidth=2, label='FairFedDrift',
            linestyle='-', marker='o', markevery=marker_indices, markersize=6)

    
    ax.set_ylabel('Macro-F1')
    ax.tick_params(axis='y')
    ax.set_xlabel('Federated Round')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Find the minimum y-value for better text placement
    min_y = min(
        macro_f1_ours.min(),
        macro_f1_sota_1.min(),
        macro_f1_sota_2.min(),
        macro_f1_sota_3.min(),
        macro_f1_sota_4.min()
    )
    text_y = min_y - 0.01  # Slightly above the minimum

    drift_str = ['Feature Drift', 'Concept Drift', 'Label Drift', 'Combined Drift']
    for dr, str in zip(drift_rounds, drift_str):
        plt.axvline(dr, color='red', linestyle='--', linewidth=2, label='Drift Event' if dr == drift_rounds[0] else "")
        plt.text(dr-2, text_y, str, rotation=90, color='red', va='bottom', ha='right')
    
    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labels1 , loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.15))
    plt.title(title)
    plt.subplots_adjust(bottom=0.18)
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()



if __name__=='__main__':

    macro_f1_ours, macro_f1_sota_1, macro_f1_sota_2, macro_f1_sota_3, macro_f1_sota_4 = load_macro_f1_values(os.path.join(SAVED_EVALS_DIR, 'macro_f1_values_iscx_vpn.npz'))
    plot_sota_drift_curves(macro_f1_ours, macro_f1_sota_1, macro_f1_sota_2, macro_f1_sota_3, macro_f1_sota_4, title='Comparing DAIR-FedMoE with SOTA on ISCX-VPN Dataset', save_file = os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-drift-sota-1.pdf'))
    
    macro_f1_ours, macro_f1_sota_1, macro_f1_sota_2, macro_f1_sota_3, macro_f1_sota_4 = load_macro_f1_values(os.path.join(SAVED_EVALS_DIR, 'macro_f1_values_iscx_tor.npz'))
    plot_sota_drift_curves(macro_f1_ours, macro_f1_sota_1, macro_f1_sota_2, macro_f1_sota_3, macro_f1_sota_4, title='Comparing DAIR-FedMoE with SOTA on ISCX-Tor Dataset', save_file = os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-drift-sota-2.pdf'))
    