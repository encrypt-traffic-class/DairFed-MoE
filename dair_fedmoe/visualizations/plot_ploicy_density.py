import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from global_config import SAVED_EVALS_DIR, SAVED_VISUALIZATIONS_DIR


plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 20
})

spawn_color = '#49beaa'
prune_color = '#ef767a'
merge_color = '#456990'

def plot_policy_density(title, log_file='policy_log_iscx_vpn.npz', save_file='fig-density-policy-iscx-vpn.pdf'):     

    data = np.load(log_file, allow_pickle=True)
    rounds = data['round']
    actions = data['actions']
    
    # Convert actions to counts per round
    unique_rounds = np.unique(rounds)
    spawn = np.zeros_like(unique_rounds, dtype=int)
    prune = np.zeros_like(unique_rounds, dtype=int)
    merge = np.zeros_like(unique_rounds, dtype=int)

    data.close()
    
    for i, round_num in enumerate(unique_rounds):
        round_actions = actions[rounds == round_num]
        spawn[i] = np.sum(round_actions == 'Spawn')
        prune[i] = np.sum(round_actions == 'Prune')
        merge[i] = np.sum(round_actions == 'Merge')

    # Smooth counts
    window = np.ones(10) / 10
    spawn_smooth = np.convolve(spawn, window, mode='same')
    prune_smooth = np.convolve(prune, window, mode='same')
    merge_smooth = np.convolve(merge, window, mode='same')

    # Ridgeline density plot
    plt.figure(figsize=(9, 5))

    plt.axvline(x=50, color='red', linestyle='--', alpha=0.9, zorder=3)
    plt.axvline(x=100, color='red', linestyle='--', alpha=0.9, zorder=3)
    plt.axvline(x=150, color='red', linestyle='--', alpha=0.9, zorder=3)
    plt.axvline(x=200, color='red', linestyle='--', alpha=0.9, zorder=3)

    plt.axhline(y=0, color=merge_color, linestyle='-', alpha=0.5, zorder=1)
    plt.axhline(y=1, color=prune_color, linestyle='-', alpha=0.5, zorder=1)
    plt.axhline(y=2.0, color=spawn_color, linestyle='-', alpha=0.5, zorder=1)

    plt.fill_between(unique_rounds, spawn_smooth + 2, 2, color=spawn_color, alpha=0.5)
    plt.fill_between(unique_rounds, prune_smooth + 1, 1, color=prune_color, alpha=0.5)
    plt.fill_between(unique_rounds, merge_smooth + 0, 0, color=merge_color, alpha=0.5)
    plt.yticks([0.5, 1.5, 2.5], ['Merge \n Density', 'Prune \n Density', 'Spawn \n Density'])
    plt.xlabel('Federated Round')
    plt.title(title)
    plt.xlim(0, 250)
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_policy_density(title='Ridgeline Density of Policy Actions for ISCX-VPN Dataset', log_file=os.path.join(SAVED_EVALS_DIR, 'policy_log_iscx_vpn.npz'), save_file=os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-density-policy-iscx-vpn.pdf'))
    plot_policy_density(title='Ridgeline Density of Policy Actions for ISCX-TOR Dataset', log_file=os.path.join(SAVED_EVALS_DIR, 'policy_log_iscx_tor.npz'), save_file=os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-density-policy-iscx-tor.pdf'))
