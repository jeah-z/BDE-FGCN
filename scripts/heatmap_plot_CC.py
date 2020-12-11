import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(
        "Prepares distribution plots for weight, logP, SA, and QED\n"
    )
    parser.add_argument(
        '--test', type=str, default='train_ifgs.csv', help='Path to the test set'
    )
    parser.add_argument(
        '--config', '-c', type=str, default='distribution_config_FG.csv',
        help='Path to the config csv with `name` and `path` columns. '
             '`name` is a model name, and '
             '`path` is a path to generated samples`'
    )
    parser.add_argument(
        '--n_jobs', type=int, default=1,
        help='number of processes to use'
    )
    parser.add_argument(
        '--img_folder', type=str, default='images/',
        help='Store images in this folder'
    )
    return parser


parser = get_parser()
config, unknown = parser.parse_known_args()
if len(unknown) != 0:
    raise ValueError("Unknown argument "+unknown[0])

os.makedirs(config.img_folder, exist_ok=True)

data = pd.read_csv('graph_bits_CC.csv')
print(data)
sns.heatmap(data, cmap='viridis')
plt.xlabel('Fragments', fontsize=14)
plt.ylabel('Bonds', fontsize=14)
title = 'Encoding of Multi-level C-C Bond Chemical Environments'
plt.title(title)
plt.xticks(rotation=30)
# plt.yticks(range(9), ['M0: C-C', 'M1: C-C', 'M2: C-H',
#                       'M3: C-H', 'M4: O-H', 'M5: O-H', 'M6: C-N', 'M7: C-Br', 'M8: C-Cl'])
plt.yticks(rotation=45)
plt.rc('font', family='Times New Roman', size=12, weight='bold')

plt.tight_layout()
plt.savefig(
    os.path.join(config.img_folder, title+'.pdf')
)
plt.savefig(
    os.path.join(config.img_folder, title+'.png'),
    dpi=250
)
plt.close()
plt.show()
