import numpy as np
import pandas as pd
import re
import os
import json
from subprocess import check_output, TimeoutExpired
from .processors.mapping import data_mapping
import tempfile
from pathlib import Path
from .processors.guide_processor import run_guide
from .processors.osrt_processor import run_osrt


def run_time_cart(dataset):
    txt_pattern = {
        "loss": (r"Train Loss: ([\d.]+)", float),
        "complexity": (r"Number of Leaves: ([\d.]+)", int),
        "time": (r"Training Duration: ([\d.]+) seconds", float),
        "test_loss": (r"Test Loss: ([\d.]+)", float)
    }

    def parse_cart_output(cart_output):
        out = {}
        for i in txt_pattern:
            res = re.search(txt_pattern[i][0], cart_output, re.M)
            if res:
                out[i] = txt_pattern[i][1](res.group(1))
            else:
                out[i] = None
        return out

    result_path = f'experiments/results/time/{dataset}_cart.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return

    train_path = data_mapping[dataset]['dataset_path']

    df = pd.read_csv(train_path)
    num_samples, num_features = df.shape
    cart_results = {
        'dataset_name': dataset,
        'num_samples': num_samples,
        'num_features': num_features}

    for depth in range(2, 10):
        try:
            cart_out = check_output(
                ["python3", 'script/processors/cart_processor.py', train_path, str(depth)], timeout=1800
            ).decode()
        except TimeoutExpired:
            print(f"CART TIMED OUT. Dataset: {dataset}, Depth: {depth}")
            continue

        cart_out_fields = parse_cart_output(cart_out)
        num_leaves = cart_out_fields['complexity']
        duration = cart_out_fields['time']
        cart_results[f'depth_{depth}'] = [{'num_leaves': num_leaves, 'time': duration}]

    with open(result_path, 'w') as f:
        json.dump(cart_results, f)


def run_time_guide(dataset):
    result_path = f'experiments/results/time/{dataset}_guide.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return

    train_path = data_mapping[dataset]['dataset_path']

    df = pd.read_csv(train_path)
    num_samples, num_features = df.shape
    guide_results = {
        'dataset_name': dataset,
        'num_samples': num_samples,
        'num_features': num_features}

    for depth in range(2, 10):
        try:
            guide_out = run_guide(train_path, None, depth, 1800)
        except TimeoutExpired:
            print(f"GUIDE TIMED OUT. Dataset: {dataset}, Depth: {depth}")
            continue
        num_leaves = guide_out['complexity']
        duration = guide_out['time']
        guide_results[f'depth_{depth}'] = [{'num_leaves': num_leaves, 'time': duration}]

    with open(result_path, 'w') as f:
        json.dump(guide_results, f)


def run_time_iai(dataset):
    result_path = f'experiments/results/time/{dataset}_iai.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return
    train_path = data_mapping[dataset]['dataset_path']

    df = pd.read_csv(train_path)
    num_samples, num_features = df.shape
    iai_results = {
        'dataset_name': dataset,
        'num_samples': num_samples,
        'num_features': num_features}

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
    lambs = list(np.concatenate([[0.0001, 0.0002, 0.0005],
                                np.arange(0.001, 0.01, 0.001), np.arange(0.01, 0.11, 0.025), [0.1, 0.2, 0.5]]))
    for depth in range(2, 10):
        for reg in lambs:
            try:
                iai_out = check_output(
                    ["python3", 'script/processors/iai_processor.py', train_path, str(depth), str(reg)], timeout=1800
                ).decode()
            except TimeoutExpired:
                print(f"IAI TIMED OUT. Dataset: {dataset}, Depth: {depth}, Regularization: {reg}")
                continue

            iai_out_fields = parse_iai_output(iai_out)
            num_leaves = iai_out_fields['complexity']
            duration = iai_out_fields['time']

            if f'depth_{depth}' not in iai_results:
                iai_results[f'depth_{depth}'] = [{'num_leaves': num_leaves, 'time': duration}]
            else:
                iai_results[f'depth_{depth}'].append({'num_leaves': num_leaves, 'time': duration})

    with open(result_path, 'w') as f:
        json.dump(iai_results, f)


def run_time_evtree(dataset):
    result_path = f'experiments/results/time/{dataset}_evtree.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return
    train_path = data_mapping[dataset]['dataset_path']

    df = pd.read_csv(train_path)
    num_samples, num_features = df.shape
    evtree_results = {
        'dataset_name': dataset,
        'num_samples': num_samples,
        'num_features': num_features}

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
        lambs = list(np.concatenate([np.arange(0.1, 1, 0.01), np.arange(1, 2.1, 0.1)]))
        for depth in range(2, 10):
            for reg in lambs:
                try:
                    evtree_out = check_output(
                        ["Rscript", 'script/processors/evtree_processor.R', train_path, str(depth), str(reg)], timeout=1800
                    ).decode()
                except TimeoutExpired:
                    print(f"EVTREE TIMED OUT. Dataset: {dataset}, Depth: {depth}, Regularization: {reg}")
                    continue

                evtree_out_fields = parse_evtree_output(evtree_out)
                num_leaves = evtree_out_fields['complexity']
                duration = evtree_out_fields['time']

                if f'depth_{depth}' not in evtree_results:
                    evtree_results[f'depth_{depth}'] = [{'num_leaves': num_leaves, 'time': duration}]
                else:
                    evtree_results[f'depth_{depth}'].append({'num_leaves': num_leaves, 'time': duration})

    with open(result_path, 'w') as f:
        json.dump(evtree_results, f)


def run_time_osrt(dataset):
    result_path = f'experiments/results/time/{dataset}_osrt.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return
    train_path = data_mapping[dataset]['dataset_path']

    df = pd.read_csv(train_path)
    num_samples, num_features = df.shape
    osrt_results = {
        'dataset_name': dataset,
        'num_samples': num_samples,
        'num_features': num_features}

    lambs = list(np.concatenate([[0.0001, 0.0002, 0.0005],
                                np.arange(0.001, 0.01, 0.001), np.arange(0.01, 0.11, 0.025), [0.1, 0.2, 0.5]]))
    for depth in range(2, 10):
        for reg in lambs:
            try:
                osrt_out = run_osrt(train_path, None, depth, reg, 1800)
            except TimeoutExpired:
                print(f"OSRT TIMED OUT. Dataset: {dataset}, Depth: {depth}, Regularization: {reg}")
                continue

            num_leaves = osrt_out['complexity']
            duration = osrt_out['time']

            if f'depth_{depth}' not in osrt_results:
                osrt_results[f'depth_{depth}'] = [{'num_leaves': num_leaves, 'time': duration}]
            else:
                osrt_results[f'depth_{depth}'].append({'num_leaves': num_leaves, 'time': duration})

    with open(result_path, 'w') as f:
        json.dump(osrt_results, f)


def run_time_experiments(dataset):
    run_time_cart(dataset)
    run_time_guide(dataset)
    run_time_iai(dataset)
    run_time_evtree(dataset)
    run_time_osrt(dataset)
