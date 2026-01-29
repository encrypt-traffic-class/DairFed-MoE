import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from global_config import SAVED_EVALS_DIR




if __name__=='__main__':
    epsilons = [5.0, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 6.4, 6.6, 6.8, 7, 7.2, 7.4, 7.6, 7.8, 8]
    macro_f1 = [95.98, 96.73, 96.28, 94.67, 95.21, 91.73, 89.88, 88.91, 87.04, 88.14, 87.06, 89.62, 87.67, 88.91, 87.49, 86.22]

    np.savez(os.path.join(SAVED_EVALS_DIR, 'privacy_tradeoff_data.npz'), epsilons=epsilons, macro_f1=macro_f1)

