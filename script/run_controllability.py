import numpy as np
import pandas as pd
import re
import os
import json
from subprocess import check_output, TimeoutExpired
from .processors.mapping import data_mapping
import tempfile
from pathlib import Path
from .processors.osrt_processor import run_osrt


def run_variance_iai(dataset):
    result_path = f'experiments/results/controllability/{dataset}_iai.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return
    train_path = data_mapping[dataset]['dataset_path']

    iai_results = {}

    txt_pattern = {
        "loss": (r"Train Loss: ([\d.]+)", float),
        "complexity": (r"Number of Leaves: ([\d.]+)", int),
        "time": (r"Training Duration: ([\d.]+) seconds", float),
        "test_loss": (r"Test Loss: ([\d.]+)", float)
    }

    def parse_iai_output(iai_output):
        out = {}
        for i in txt_pattern:
            res = re.search(txt_pattern[i][0], iai_output, re.M)
            if res:
                out[i] = txt_pattern[i][1](res.group(1))
            else:
                out[i] = None
        return out

    lambs = [0.0005, 0.001, 0.005, 0.01, 0.05]

    for depth in range(4, 10):
        if f'depth_{depth}' not in iai_results:
            iai_results[f'depth_{depth}'] = {}

        for reg in lambs:
            if reg not in iai_results[f'depth_{depth}']:
                iai_results[f'depth_{depth}'][reg] = []

            for j in range(10):
                try:
                    iai_out = check_output(
                        ["python3", 'script/processors/iai_processor.py', train_path, str(depth), str(reg)],
                        timeout=1800
                    ).decode()
                except TimeoutExpired:
                    print(f"IAI TIMED OUT. Dataset: {dataset}, Depth: {depth}, Regularization: {reg}")
                    continue

                iai_out_fields = parse_iai_output(iai_out)
                num_leaves = iai_out_fields['complexity']
                train_mse = iai_out_fields['loss']
                iai_results[f'depth_{depth}'][reg].append({'num_leaves': num_leaves, 'train_mse': train_mse})

    with open(result_path, 'w') as f:
        json.dump(iai_results, f)


def run_variance_evtree(dataset):
    result_path = f'experiments/results/controllability/{dataset}_evtree.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return
    train_path = data_mapping[dataset]['dataset_path']
    evtree_results = {}
    txt_pattern = {
        "loss": (r"Train Loss: ([\d.]+)", float),
        "complexity": (r"Number of Leaves: ([\d.]+)", int),
        "time": (r"Training Duration: ([\d.]+) seconds", float),
        "test_loss": (r"Test Loss: ([\d.]+)", float)
    }

    def parse_evtree_output(evtree_output):
        out = {}
        for i in txt_pattern:
            res = re.search(txt_pattern[i][0], evtree_output, re.M)
            if res:
                out[i] = txt_pattern[i][1](res.group(1))
            else:
                out[i] = None
        return out

    with tempfile.TemporaryDirectory(dir='/tmp') as tmpdirname:
        # create proper format file for evtree
        old = pd.read_csv(train_path)
        new = pd.DataFrame()
        for col in old.columns[:-1]:
            new[col] = np.where(old[col] == 1, 'yes', 'no')

        data = pd.concat([new, old.iloc[:, -1]], axis=1)
        tmpdirname = Path(tmpdirname)
        train_path = tmpdirname / 'evtree_train.csv'
        data.to_csv(train_path, index=False)

        lambs = [0.05, 0.08, 0.1, 0.5, 1]
        for depth in range(4, 10):
            if f'depth_{depth}' not in evtree_results:
                evtree_results[f'depth_{depth}'] = {}
            for reg in lambs:
                if reg not in evtree_results[f'depth_{depth}']:
                    evtree_results[f'depth_{depth}'][reg] = []

                for j in range(10):
                    try:
                        evtree_out = check_output(
                            ["Rscript", 'script/processors/evtree_processor.R', train_path, str(depth), str(reg)], timeout=1800
                        ).decode()
                    except TimeoutExpired:
                        print(f"EVTREE TIMED OUT. Dataset: {dataset}, Depth: {depth}, Regularization: {reg}")
                        continue

                    evtree_out_fields = parse_evtree_output(evtree_out)
                    num_leaves = evtree_out_fields['complexity']
                    train_mse = evtree_out_fields['loss']
                    evtree_results[f'depth_{depth}'][reg].append({'num_leaves': num_leaves, 'train_mse': train_mse})

    with open(result_path, 'w') as f:
        json.dump(evtree_results, f)


def run_variance_osrt(dataset):
    result_path = f'experiments/results/controllability/{dataset}_osrt.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return
    train_path = data_mapping[dataset]['dataset_path']
    osrt_results = {}
    lambs = [0.0005, 0.001, 0.005, 0.01, 0.05]
    for depth in range(4, 10):
        if f'depth_{depth}' not in osrt_results:
            osrt_results[f'depth_{depth}'] = {}
        for reg in lambs:
            if reg not in osrt_results[f'depth_{depth}']:
                osrt_results[f'depth_{depth}'][reg] = []

            for j in range(10):
                try:
                    osrt_out = run_osrt(train_path, None, depth, reg, 1800)
                except TimeoutExpired:
                    print(f"OSRT TIMED OUT. Dataset: {dataset}, Depth: {depth}, Regularization: {reg}")
                    continue

                num_leaves = osrt_out['complexity']
                train_mse = osrt_out['loss']
                osrt_results[f'depth_{depth}'][reg].append({'num_leaves': num_leaves, 'train_mse': train_mse})
    with open(result_path, 'w') as f:
        json.dump(osrt_results, f)


def run_variance_experiments(dataset):
    run_variance_iai(dataset)
    run_variance_evtree(dataset)
    run_variance_osrt(dataset)

