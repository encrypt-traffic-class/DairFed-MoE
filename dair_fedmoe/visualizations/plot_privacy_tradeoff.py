import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from global_config import SAVED_EVALS_DIR, SAVED_VISUALIZATIONS_DIR
from global_utils import load_privacy_tradeoff_data


plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 18
})

def plot_privacy_tradeoff(title, data_file, save_file):
    epsilons, macro_f1 = load_privacy_tradeoff_data(data_file)
    plt.figure(figsize=(7, 5))

    plt.vlines(epsilons, ymin=min(macro_f1) - 2.0, ymax=macro_f1, colors='red', linestyles='--', alpha=0.5)

    plt.plot(epsilons, macro_f1, marker='o', linestyle='-', linewidth=2, markersize=8, color='red')

    # Add value annotations above each point
    for x, y in zip(epsilons, macro_f1):
        plt.annotate(f'{x:.1f}', 
                    (x, y+0.1), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=14)

    plt.title('Privacy–Utility Trade-off Curve', pad=15)
    plt.xlabel('Privacy Budget $\\varepsilon$')
    plt.ylabel('Macro-F1 Score')
    plt.ylim(min(macro_f1) - 2.0, max(macro_f1) + 2.0)
    plt.xlim(min(epsilons) - 0.2, max(epsilons) + 0.2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    plot_privacy_tradeoff(title='Privacy–Utility Trade-off Curve', 
                          data_file=os.path.join(SAVED_EVALS_DIR, 'privacy_tradeoff_data.npz'), 
                          save_file=os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-privacy-tradeoff.pdf'))
