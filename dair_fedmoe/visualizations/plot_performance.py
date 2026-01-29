import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from global_config import SAVED_EVALS_DIR, SAVED_VISUALIZATIONS_DIR
from global_utils import load_plot_data


plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 20
})

def plot_performance(drift_rounds, title, eval_file, save_file):

    data = load_plot_data(eval_file)
    rounds = data['round']
    macro_f1 = data['macro_f1']
    expert_count = data['expert_count']
    actions = data['actions']
    actions_new = data['actions_new']

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(rounds, macro_f1, color='tab:blue', linewidth=2, label='Macro-F1')
    ax1.set_ylabel('Macro-F1', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xlabel('Federated Round')
    ax1.grid(True, linestyle='--', alpha=0.5)


    # Expert count line
    ax2 = ax1.twinx()
    ax2.plot(rounds, expert_count, color='tab:gray', linewidth=2, label='Expert Count', zorder=1)
    ax2.set_ylabel('Active Experts')
    ax2.tick_params(axis='y')

    drift_str = ['Feature Drift', 'Concept Drift', 'Label Drift', 'Combined Drift']
    for dr, str in zip(drift_rounds, drift_str):
        plt.axvline(dr, color='red', linestyle='--', linewidth=2, label='Drift Event' if dr == drift_rounds[0] else "")
        # Add 'DRIFT' text at the bottom, right-aligned with the round number
        ax1.text(dr-2, ax1.get_ylim()[0] + 0.02 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]), str, color='red', fontsize=18, ha='right', va='bottom', backgroundcolor='white', clip_on=True)

    # Action markers
    marker_map = {'Spawn':'^', 'Prune':'s', 'Merge':'D', 'NoOp':'x'}
    color_map = {'Spawn':'green', 'Prune':'orange', 'Merge':'red', 'NoOp':'black'}

    for action in ['Spawn', 'Prune', 'Merge']:
        idx = np.where(actions_new == action)[0]
        ax2.scatter(rounds[idx], expert_count[idx],
                    marker=marker_map[action],
                    color=color_map[action],
                    edgecolors='black',
                    s=80,
                    label=action, zorder=2)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
            loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.15))

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()



if __name__=='__main__':

    plot_performance([50, 100, 150, 200],
                     'DAIR-FedMoE Performance, Expert Count, and Policy Actions on ISCX-VPN', 
                     os.path.join(SAVED_EVALS_DIR, 'performance-1.npz'),
                     os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-performance-1.pdf'))


    plot_performance([50, 100, 150, 200],
                     'DAIR-FedMoE Performance, Expert Count, and Policy Actions on ISCX-Tor',
                     os.path.join(SAVED_EVALS_DIR, 'performance-2.npz'),
                     os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-performance-2.pdf'))