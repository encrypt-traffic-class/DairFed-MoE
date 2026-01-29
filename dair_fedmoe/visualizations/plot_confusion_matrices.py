import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from global_config import SAVED_VISUALIZATIONS_DIR, SAVED_EVALS_DIR
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from global_utils import load_confusion_matrix

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 18
})


def plot_CM(class_names, title, npz_file, save_file, fontsize=12):
    num_classes = len(class_names)
    cm = load_confusion_matrix(npz_file)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, aspect='equal', cmap='Purples')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel('Proportion', rotation=270, labelpad=15)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # Annotate each cell
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if i == j else 'black'
            ax.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center', fontsize=fontsize, color=color)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()





if __name__=="__main__":

    class_names = [
        "email", "chat", "stream", "ft", "voip", "p2p",
        "vpn_email", "vpn_chat", "vpn_stream", "vpn_ft",
        "vpn_voip", "vpn_p2p"
    ]



    plot_CM(class_names, 
            title='Confusion Matrix for ISCX-VPN', 
            npz_file=os.path.join(SAVED_EVALS_DIR, 'confusion_matrix_iscx_vpn.npz'),
            save_file=os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-cm-iscx-vpn.pdf'))
 


    class_names = [
        "browse", "email", "chat", "audio", "video",
        "ft", "voip", "p2p",
        "tor_browse", "tor_email", "tor_chat", "tor_audio",
        "tor_video", "tor_ft", "tor_voip", "tor_p2p"
        ]
    
    plot_CM(class_names, 
            title='Confusion Matrix for ISCX-TOR', 
            npz_file=os.path.join(SAVED_EVALS_DIR, 'confusion_matrix_iscx_tor.npz'),
            save_file=os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-cm-iscx-tor.pdf'))
    

    class_names = [
    "stream", "voip", "ft", "p2p", 
    "vpn_stream", "vpn_voip", "vpn_ft", "vpn_p2p"
    ]

    plot_CM(class_names, 
            title='Confusion Matrix for VNAT', 
            npz_file=os.path.join(SAVED_EVALS_DIR, 'confusion_matrix_vnat.npz'),
            save_file=os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-cm-vnat.pdf'))
    
    
    
    class_names = [
    "Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus",
    "BitTorrent", "Facetime", "FTP", "Gmail", "MySQL", "Outlook", "Skype", "SMB", "Weibo", "WW"
    ]
    
    plot_CM(class_names,
            title='Confusion Matrix for USTC-TFC2016', 
            npz_file=os.path.join(SAVED_EVALS_DIR, 'confusion_matrix_ustc.npz'),
            save_file=os.path.join(SAVED_VISUALIZATIONS_DIR, 'fig-cm-ustc.pdf'),
            fontsize=8)
    

