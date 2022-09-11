import json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import AutoMinorLocator


def get_result_path(dataset, alg):
    return f'experiments/results/time/{dataset}_{alg}.json'


def get_alg_plotting_style(alg):
    if alg == 'iai':
        shape = 'D'
        color = 'hotpink'
        alpha = 0.75
    elif alg == 'evtree':
        shape = 'x'
        color = 'black'
        alpha = 0.75
    elif alg == 'guide':
        shape = 's'
        color = 'forestgreen'
        alpha = 0.75
    elif alg == 'cart':
        shape = 'D'
        color = 'darkorange'
        alpha = 0.75
    else:
        shape = '^'
        color = 'royalblue'
        alpha = 0.9

    return shape, color, alpha


def plot_time_dataset_depth(dataset, depth):
    # helper function
    def get_trees(file_path, d):
        with open(file_path) as f:
            results = json.load(f)

        alg_results = {}
        for tree in results[f'depth_{d}']:
            num_leaves = tree['num_leaves']
            duration = tree['time']
            if num_leaves > 30: continue

            if num_leaves in alg_results:
                alg_results[num_leaves].append(duration)
            else:
                alg_results[num_leaves] = [duration]

        x = []
        y = []
        y_low = []
        y_high = []

        for num_leaves in alg_results:
            x.append(num_leaves)
            y_iqr = np.quantile(alg_results[num_leaves], [0, 0.5, 1])
            y.append(y_iqr[1])
            y_low.append(y_iqr[1] - y_iqr[0])
            y_high.append(y_iqr[2] - y_iqr[1])

        return x, y, y_low, y_high

    fig = plt.figure(figsize=(8, 5.5), dpi=80)
    plt.rc('font', size=18)
    x_max = 0
    for alg in ['iai', 'evtree', 'guide', 'cart', 'osrt']:
        result_path = get_result_path(dataset, alg)
        x, y, y_low, y_high = get_trees(result_path, depth)
        shape, color, alpha = get_alg_plotting_style(alg)
        plt.errorbar(x, y, yerr=[y_low, y_high], label=alg,
                     marker=shape, markersize=10, c=color, alpha=alpha, linewidth=1, linestyle='none')
        x_max = max(max(x) if len(x) else 0, x_max)

    if x_max > 15:
        x_major_locator = MultipleLocator(5)
    else:
        x_major_locator = MultipleLocator(int(x_max / 4))

    plt.gca().xaxis.set_major_locator(x_major_locator)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())

    plt.xlabel('Number of Leaves')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Number of Leaves\n ' + dataset + ', max depth: ' + str(depth))
    # if dataset == 'yacht' and depth == 9:
    #     plt.legend(loc='upper right')
    # if depth == 2:
    #     plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.tight_layout()
    fig_path = Path(f'figures/time/{dataset}_depth_{depth}.png')
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(str(fig_path))
    plt.close()