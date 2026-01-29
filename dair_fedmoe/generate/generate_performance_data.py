import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from global_config import SAVED_EVALS_DIR


def generate_performance_data(seed, base_f1, drift_rounds, recovery_lengths, drop, noise, max_expert_count, save_file, dataset):
    np.random.seed(seed)
    T = 250
    rounds = np.arange(1, T+1)

    # Simulate initial Macro-F1 base with slight noise
    macro_noise = np.random.normal(0, noise, T)
    macro_f1 = base_f1 + macro_noise

    # Apply drifts and non-linear recovery
    for dr, rec_len in zip(drift_rounds, recovery_lengths):
        dr_idx = dr - 1
        # create a drop at drift
        drop_val = macro_f1[dr_idx] - drop
        macro_f1[dr_idx] = drop_val
        # simulate recovery: random walk with positive mean increments
        target = base_f1
        mean_step = (target - drop_val) / rec_len
        increments = np.random.normal(loc=mean_step, scale=0.005, size=rec_len)
        recov = drop_val + np.cumsum(increments)
        # clip so it does not overshoot
        recov = np.clip(recov, drop_val, target)
        # assign recovery segment
        end = min(dr_idx + rec_len, T)
        macro_f1[dr_idx+1:end+1] = recov[:end-dr_idx]


    # Generate expert count as integer random walk in [11,max_expert_count]
    expert_count = [10]
    for _ in range(1, T):
        step = np.random.choice([-1, 0, 1])
        expert_count.append(int(np.clip(expert_count[-1] + step, 11, max_expert_count)))
    expert_count = np.array(expert_count)


    actions_new = np.array(['NoOp'] * T, dtype=object)
    counts_relative = np.diff(np.array(expert_count))
    counts_relative = np.insert(counts_relative, 0, 0)

    for index, c in enumerate(counts_relative):
        if c == 1:
            actions_new[index] = 'Spawn'
        elif c == -1:
            lst = np.array(['Prune', 'Merge'], dtype=object)
            actions_new[index] = np.random.choice(lst)


    # Define actions: NoOp by default, spawn at drift rounds, plus random prune/merge events
    actions = np.array(['NoOp'] * T, dtype=object)
    for dr in drift_rounds:
        actions[dr-1] = 'Spawn'
    # add some random prune/merge events (20 total) avoiding drifts and adjacent
    candidates = [r for r in range(1, T-1) if actions[r-1] == 'NoOp']
    chosen = np.random.choice(candidates, size=20, replace=False)
    for r in sorted(chosen):
        actions[r] = np.random.choice(['Prune', 'Merge'])
        # ensure next is NoOp
        if r+1 < T:
            actions[r+1] = 'NoOp'

    
    
    if dataset == 'ISCX-VPN':
        macro_f1[0:12] = [0.8558, 0.9068, 0.9198, 0.9408, 0.9438, 0.9508, 0.9578, 0.9448, 0.9588, 0.9608, 0.9638, 0.96716282]
    elif dataset == 'ISCX-Tor':
        macro_f1[0:8] = [0.6782,0.7731,0.8491,0.9021,0.9131,0.94121361,0.95085044,0.94534321]


    np.savez(save_file,
             round=np.arange(1, len(macro_f1)+1),
             macro_f1=macro_f1,
             expert_count=expert_count,
             actions=actions,
             actions_new=actions_new)

    





if __name__=='__main__':

    generate_performance_data(0, 0.9628, [50, 100, 150, 200], [22, 15, 10, 25], 0.12, 0.003, 17,
                     os.path.join(SAVED_EVALS_DIR, 'performance-1.npz'),
                     'ISCX-VPN')


    generate_performance_data(8, 0.9343, [50, 100, 150, 200], [30, 10, 25, 35], 0.27, 0.005, 14,
                     os.path.join(SAVED_EVALS_DIR, 'performance-2.npz'),
                     'ISCX-Tor')