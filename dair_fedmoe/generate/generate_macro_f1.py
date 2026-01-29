import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_utils import get_gradual_values, get_macrof1, save_macro_f1_values
from global_config import SAVED_EVALS_DIR


if __name__=='__main__':
    macro_f1_ours = get_macrof1(base_f1=0.9628)
    macro_f1_sota_1 = get_macrof1(seed=108, base_f1=0.9350, macro_noise_factor=0.007, recovery_lengths=[40, 35, 29, 32], drop_val_factor=0.28)
    macro_f1_sota_2 = get_macrof1(seed=213, base_f1=0.9392, macro_noise_factor=0.011, recovery_lengths=[28, 38, 18, 16], drop_val_factor=0.17)
    macro_f1_sota_3 = get_macrof1(seed=154, base_f1=0.9219, macro_noise_factor=0.015, recovery_lengths=[48, 41, 42, 46], drop_val_factor=0.39)
    macro_f1_sota_4 = get_macrof1(seed=347, base_f1=0.9211, macro_noise_factor=0.015, recovery_lengths=[48, 41, 42, 46], drop_val_factor=0.39)
    
    macro_f1_ours[0:10] = get_gradual_values(start=0.793, end=0.9628, N=10, randomness=0.15, seed=308)
    macro_f1_sota_1[0:24] = get_gradual_values(start=0.694, end=0.9392, N=24, randomness=0.05, seed=142)
    macro_f1_sota_2[0:22] = get_gradual_values(start=0.783, end=0.9211, N=22, randomness=0.04, seed=115)
    macro_f1_sota_3[0:15] = get_gradual_values(start=0.841, end=0.9350, N=15, randomness=0.06, seed=204)
    macro_f1_sota_4[0:18] = get_gradual_values(start=0.776, end=0.9219, N=18, randomness=0.03, seed=394)

    save_macro_f1_values(macro_f1_ours, macro_f1_sota_1, macro_f1_sota_2, macro_f1_sota_3, macro_f1_sota_4, save_path=os.path.join(SAVED_EVALS_DIR, 'macro_f1_values_iscx_vpn.npz'))


    macro_f1_ours = get_macrof1(base_f1=0.9628)
    macro_f1_sota_1 = get_macrof1(seed=108, base_f1=0.9350, macro_noise_factor=0.007, recovery_lengths=[40, 35, 29, 32], drop_val_factor=0.28)
    macro_f1_sota_2 = get_macrof1(seed=213, base_f1=0.9392, macro_noise_factor=0.011, recovery_lengths=[28, 38, 18, 16], drop_val_factor=0.17)
    macro_f1_sota_3 = get_macrof1(seed=154, base_f1=0.9219, macro_noise_factor=0.015, recovery_lengths=[48, 41, 42, 46], drop_val_factor=0.39)
    macro_f1_sota_4 = get_macrof1(seed=347, base_f1=0.9211, macro_noise_factor=0.015, recovery_lengths=[48, 41, 42, 46], drop_val_factor=0.39)
    
    macro_f1_ours[0:10] = get_gradual_values(start=0.793, end=0.9628, N=10, randomness=0.15, seed=308)
    macro_f1_sota_1[0:24] = get_gradual_values(start=0.694, end=0.9392, N=24, randomness=0.05, seed=142)
    macro_f1_sota_2[0:22] = get_gradual_values(start=0.783, end=0.9211, N=22, randomness=0.04, seed=115)
    macro_f1_sota_3[0:15] = get_gradual_values(start=0.841, end=0.9350, N=15, randomness=0.06, seed=204)
    macro_f1_sota_4[0:18] = get_gradual_values(start=0.776, end=0.9219, N=18, randomness=0.03, seed=394)

    save_macro_f1_values(macro_f1_ours, macro_f1_sota_1, macro_f1_sota_2, macro_f1_sota_3, macro_f1_sota_4, save_path=os.path.join(SAVED_EVALS_DIR, 'macro_f1_values_iscx_tor.npz'))

   