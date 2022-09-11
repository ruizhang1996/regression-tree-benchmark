import os
import json
import time
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
    return f'experiments/results/controllability/{dataset}_{alg}.json'


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


def plot_controllability_dataset_depth(dataset, depth):
    # helper function
    def get_trees(file_path, d):
        with open(file_path) as f:
            results = json.load(f)

        x = []
        y = []
        x_low = []
        x_high = []
        y_low = []
        y_high = []

        for reg in results[f'depth_{d}']:
            reg_num_leaves = []
            reg_train_mse = []
            for tree in results[f'depth_{d}'][reg]:
                num_leaves = tree['num_leaves']
                train_mse = tree['train_mse']

                reg_num_leaves.append(num_leaves)
                reg_train_mse.append(np.sqrt(train_mse))

            x_iqr = np.quantile(reg_num_leaves, [0, 0.5, 1])
            y_iqr = np.quantile(reg_train_mse, [0, 0.5, 1])
            point = (
                x_iqr[1], y_iqr[1],
                x_iqr[1] - x_iqr[0], x_iqr[2] - x_iqr[1],
                y_iqr[1] - y_iqr[0], y_iqr[2] - y_iqr[1])

            x.append(point[0])
            y.append(point[1])
            x_low.append(point[2])
            x_high.append(point[3])
            y_low.append(point[4])
            y_high.append(point[5])

        return x, y, x_low, x_high, y_low, y_high

    fig = plt.figure(figsize=(8, 5.5), dpi=80)
    plt.rc('font', size=18)
    x_max = 0
    for alg in ['iai', 'evtree', 'osrt']:
        result_path = get_result_path(dataset, alg)
        x, y, x_low, x_high, y_low, y_high = get_trees(result_path, depth)
        shape, color, alpha = get_alg_plotting_style(alg)
        plt.errorbar(x, y, xerr=[x_low, x_high], yerr=[y_low, y_high], label=alg,
                     marker=shape, markersize=10, c=color, alpha=alpha, linewidth=1, linestyle='none')
        x_max = max(max(x) if len(x) else 0, x_max)

    if x_max > 60:
        x_major_locator = MultipleLocator(20)
    elif x_max > 20:
        x_major_locator = MultipleLocator(10)
    else:
        x_major_locator = MultipleLocator(4)

    plt.gca().xaxis.set_major_locator(x_major_locator)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('Number of Leaves')
    plt.ylabel('Training Loss (RMSE)')
    plt.title('Variance of Runs\n ' + dataset + ', max depth: ' + str(depth))
    plt.legend(loc='upper right')
    plt.tight_layout()
    fig_path = Path(f'figures/controllability/{dataset}_depth_{depth}.png')
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path)
    plt.close()