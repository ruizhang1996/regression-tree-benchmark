import os
import json
import time
from typing import List, Any

import numpy as np
import pandas as pd
# from processors.mapping import data_mapping
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.pyplot import MultipleLocator, MaxNLocator
from matplotlib.ticker import AutoMinorLocator


# from processors.model.tree_classifier import TreeClassifier
# import processors.utils as draw_osrt


def get_result_path(dataset, alg):
    return f'experiments/results/loss/{dataset}_{alg}.json'


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


def plot_loss_dataset_depth(dataset, depth):
    # helper function
    def get_trees(file_path, d):
        with open(file_path) as f:
            results = json.load(f)

        alg_results = {}
        for tree in results[f'depth_{d}']:
            num_leaves = tree['num_leaves']
            train_mse = tree['train_mse']

            if num_leaves < 2 or num_leaves > 30:
                continue
            if depth > 3 and num_leaves < 4:
                continue  # comment this line if want full range
            if 'yacht' not in file_path and depth > 6 and num_leaves <= 10:
                continue  # comment this line if want full range

            if num_leaves in alg_results:
                alg_results[num_leaves].append(train_mse)
            else:
                alg_results[num_leaves] = [train_mse]

        x = []
        y = []
        y_low = []
        y_high = []

        for num_leaves in alg_results:
            x.append(num_leaves)
            loss = np.sqrt(alg_results[num_leaves])
            y_iqr = np.quantile(loss, [0, 0.5, 1])
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
    plt.ylabel('Training Loss (RMSE)')
    plt.title('Training Loss vs Number of Leaves\n ' + dataset + ', max depth: ' + str(depth))
    plt.legend(loc='upper right')
    plt.tight_layout()
    fig_path = Path(f'figures/loss/{dataset}_depth_{depth}.png')
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
