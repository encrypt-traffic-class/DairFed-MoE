import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from global_config import SAVED_EVALS_DIR, SAVED_VISUALIZATIONS_DIR
import matplotlib.pyplot as plt
import random
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 20
})

normal_expert_color = '#1e81b0'
active_expert_color = '#e66f2b'
spawn_action_color = '#0ead69'
prune_action_color = '#f6511d'
merge_action_color = '#2d3142'

def plot_expert_lifecycle(title, log_file, save_file):

    data = np.load(log_file, allow_pickle=True)
    rounds = data['round']
    actions = data['actions']
    expert_counts = data['expert_count']
    random.seed(42)

    # Initialize experts at round 1
    initial_count = int(expert_counts[0])
    last_round = int(rounds[-1])
    next_id = initial_count + 1

    active = [f'E{i}' for i in range(1, initial_count + 1)]
    lifespans = {e: {'start': 1, 'end': None} for e in active}

    spawn_events = []
    prune_events = []
    merge_events = []

    # Build lifecycles with random assignment
    for r, action in zip(rounds, actions):
        r = int(r)
        action = action
        if action == 'Spawn':
            new_e = f'E{next_id}'
            lifespans[new_e] = {'start': r, 'end': None}
            active.append(new_e)
            spawn_events.append((r, new_e))
            next_id += 1
        elif action == 'Prune':
            pruned = random.choice(active)
            lifespans[pruned]['end'] = r
            active.remove(pruned)
            prune_events.append((r, pruned))
        elif action == 'Merge':
            # pick two distinct experts
            src, tgt = random.sample(active, 2)
            lifespans[src]['end'] = r
            active.remove(src)  # src merged into tgt
            merge_events.append((r, src, tgt))
        # No action or other -> no change
        # Check consistency
        assert len(active) == int(expert_counts[r-1]), \
            f"Mismatch at round {r}"

    # End lifespans for remaining active experts
    for e in active:
        lifespans[e]['end'] = last_round

    # Prepare y positions
    experts = sorted(lifespans.keys(), key=lambda x: int(x[1:]))
    y_pos = {e: i for i, e in enumerate(experts, start=1)}

    # Plot with increased figure size and adjusted margins
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw lines for each expert
    for e in experts:
        start = lifespans[e]['start']
        end = lifespans[e]['end']
        # Use different color for experts that remain active at the end
        if e in active:
            ax.hlines(y_pos[e], start, end, linewidth=5, color=active_expert_color, alpha=1.0)  # Highlight color for active experts
        else:
            ax.hlines(y_pos[e], start, end, linewidth=5, color=normal_expert_color, alpha=1.0)  # Regular color for inactive experts

    # Add vertical dashed lines at specific rounds
    for round_num in [50, 100, 150, 200]:
        ax.axvline(x=round_num, color='red', linestyle='--', alpha=0.9, zorder=1)

    # Mark events
    for r, e in spawn_events:
        ax.scatter(r, y_pos[e], marker='o', s=30, zorder=3, color=spawn_action_color, alpha=1.0)
    for r, e in prune_events:
        ax.scatter(r, y_pos[e], marker='x', s=50, zorder=3, color=prune_action_color, alpha=1.0)
    for r, src, tgt in merge_events:
        # Create curved arrow
        arrow = FancyArrowPatch(
            (r, y_pos[src]),  # Start point
            (r, y_pos[tgt]-0.3),  # End point
            connectionstyle="arc3,rad=0",  # Create curved arrow
            arrowstyle='->',
            mutation_scale=15,
            color=merge_action_color,
            alpha=0.7,
            linewidth=2.0,
            zorder=3
        )
        ax.add_patch(arrow)

    # Create legend elements
    legend_elements = [
        Line2D([0], [0], color=normal_expert_color, lw=5, label='Inactive Expert'),
        Line2D([0], [0], color=active_expert_color, lw=5, label='Active Expert'),
        Line2D([0], [0], color='red', linestyle='--', lw=3, alpha=1.0, label='Drift Event'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=spawn_action_color, markersize=18, label='Spawn Event'),
        Line2D([0], [0], marker='x',  lw=0, color=prune_action_color, markerfacecolor=prune_action_color, markersize=15, label='Prune Event'),
        Line2D([0], [0], color=merge_action_color, lw=3, label='Merge Event')
]

    # Add legend
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    ax.set_xlabel('Federated Round')
    ax.set_ylabel('Expert')
    
    # Adjust y-axis labels to show only odd-numbered experts
    y_ticks = []
    y_labels = []
    for e in experts:
        expert_num = int(e[1:])  # Extract the number from expert label
        if expert_num % 2 == 1:  # Only include odd numbers
            y_ticks.append(y_pos[e])
            y_labels.append(e)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=14)
    
    # Add more space for y-axis labels
    plt.subplots_adjust(left=0.15, right=0.85)  # Adjust right margin for legend
    
    ax.set_xlim(1, last_round)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_expert_lifecycle(title='Expert Lifecycle Across 250 Federated Rounds for ISCX-VPN Dataset', 
                          log_file=os.path.join(SAVED_EVALS_DIR, 'policy_log_iscx_vpn.npz'),  
                          save_file=os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-expert-lifecycle-iscx-vpn.pdf'))
    
    
    plot_expert_lifecycle(title='Expert Lifecycle Across 250 Federated Rounds for ISCX-TOR Dataset', 
                          log_file=os.path.join(SAVED_EVALS_DIR, 'policy_log_iscx_tor.npz'), 
                          save_file=os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-expert-lifecycle-iscx-tor.pdf'))

    
    
    
    
