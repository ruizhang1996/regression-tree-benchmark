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
    return f'experiments/results/scalability/{dataset}_{alg}.json'


def get_alg_plotting_style(alg):
    if alg == 'iai':
        shape = 'D'
        color = 'hotpink'
        alpha = 0.75
    elif alg == 'evtree':
        shape = 'x'
        color = 'black'
        alpha = 0.9
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
        alpha = 1

    return shape, color, alpha


def plot_scalability_dataset_depth(dataset):
    # helper function
    def get_trees(file_path):
        with open(file_path) as f:
            results = json.load(f)

        x = []
        y = []
        for tree in results['results']:
            num_samples = tree['num_samples']
            duration = tree['time']
            x.append(num_samples)
            y.append(duration)

        return x, y

    fig = plt.figure(figsize=(8, 5.5), dpi=80)
    plt.rc('font', size=18)
    x_max = 0
    for alg in ['iai', 'evtree', 'guide', 'cart', 'osrt']:
        result_path = get_result_path(dataset, alg)
        x, y = get_trees(result_path)
        shape, color, alpha = get_alg_plotting_style(alg)
        plt.plot(x, y, label=alg, marker=shape, markersize=10, c=color, alpha=alpha, linewidth=1, linestyle='dashed')
        x_max = max(max(x) if len(x) else 0, x_max)

    plt.axhline(1800, linestyle="solid", color="k", label='timeout')
    plt.gca().set_xscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Number of Samples\n ' + dataset + ', max depth: 5')
    plt.legend(loc='best')
    plt.tight_layout()
    fig_path = Path(f'figures/scalability/{dataset}.png')
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path)
    plt.close()