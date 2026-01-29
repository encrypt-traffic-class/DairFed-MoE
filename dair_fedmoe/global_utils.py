import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd



def get_gradual_values(start, end, N, randomness=0.1, seed=42):
    np.random.seed(seed)

    x = np.linspace(0, 1, N)
    base = start + (end - start) * x

    noise_scale = (1 - x) * randomness * (end - start)
    noise = np.random.normal(0, 1, N) * noise_scale

    values = base + noise

    values[0] = start
    values[-1] = end
    
    return values.tolist()






def get_macrof1(T = 250, drift_rounds = [50, 100, 150, 200],seed=0, base_f1=0.98, macro_noise_factor=0.003, recovery_lengths=[22, 15, 10, 25], drop_val_factor=0.12):
    np.random.seed(seed) 

    macro_noise = np.random.normal(0, macro_noise_factor, T)
    macro_f1 = base_f1 + macro_noise

    for dr, rec_len in zip(drift_rounds, recovery_lengths):
        dr_idx = dr - 1
        # create a drop at drift
        drop_val = macro_f1[dr_idx] - drop_val_factor
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


    return macro_f1

def save_macro_f1_values(macro_f1_ours, macro_f1_sota_1, macro_f1_sota_2, macro_f1_sota_3, macro_f1_sota_4, save_path):

    np.savez(
        save_path,
        macro_f1_ours=macro_f1_ours,
        macro_f1_sota_1=macro_f1_sota_1,
        macro_f1_sota_2=macro_f1_sota_2,
        macro_f1_sota_3=macro_f1_sota_3,
        macro_f1_sota_4=macro_f1_sota_4
    )

def load_macro_f1_values(load_path):
    data = np.load(load_path)
    return (
        data['macro_f1_ours'],
        data['macro_f1_sota_1'],
        data['macro_f1_sota_2'],
        data['macro_f1_sota_3'],
        data['macro_f1_sota_4']
    )



def load_boxplot_data(load_path):
    data = np.load(load_path)
    scores = data['scores']
    
    dataset_names = ['ISCX-VPN', 'ISCX-Tor', 'VNAT', 'USTC-TFC2016']
    data_list = []
    
    for dataset_name, dataset_scores in zip(dataset_names, scores):
        for score in dataset_scores:
            data_list.append({'Dataset': dataset_name, 'Macro-F1': score})
    
    return pd.DataFrame(data_list)



def load_privacy_tradeoff_data(file_path):
    data = np.load(file_path)
    epsilons = data['epsilons']
    macro_f1 = data['macro_f1']
    data.close()
    return epsilons, macro_f1


def load_confusion_matrix(load_path):
    data = np.load(load_path, allow_pickle=True)
    return data['confusion_matrix']


def load_plot_data(filename):
        data = np.load(filename, allow_pickle=True)
        return {
            'round': data['round'],
            'macro_f1': data['macro_f1'], 
            'expert_count': data['expert_count'],
            'actions': data['actions'],
            'actions_new': data['actions_new']
        }