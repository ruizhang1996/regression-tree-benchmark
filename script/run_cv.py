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
from sklearn.model_selection import KFold


def run_cv_cart(dataset, train_paths, test_paths):
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

    result_path = f'experiments/results/cv/{dataset}_cart.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return

    data_path = data_mapping[dataset]['dataset_path']
    df = pd.read_csv(data_path)
    num_samples, num_features = df.shape
    cart_results = {
        'dataset_name': dataset,
        'num_samples': num_samples,
        'num_features': num_features}

    for depth in range(2, 6):
        if f'depth_{depth}' not in cart_results:
            cart_results[f'depth_{depth}'] = {}

        for fold in range(0, 5):
            try:
                cart_out = check_output(
                    ["python3", 'script/processors/cart_processor.py', train_paths[fold],
                     str(depth), '-t', test_paths[fold]], timeout=300
                ).decode()
            except TimeoutExpired:
                print(f"CART TIMED OUT. Dataset: {dataset}, Depth: {depth}, Fold: {fold + 1}")
                continue

            cart_out_fields = parse_cart_output(cart_out)
            num_leaves = cart_out_fields['complexity']
            train_mse = cart_out_fields['loss']
            test_mse = cart_out_fields['test_loss']
            if f'fold_{fold + 1}' not in cart_results[f'depth_{depth}']:
                cart_results[f'depth_{depth}'][f'fold_{fold + 1}'] = [{'regularization': None, 'num_leaves': num_leaves,
                                                                       'train_mse': train_mse, 'test_mse': test_mse}]
            else:
                cart_results[f'depth_{depth}'][f'fold_{fold + 1}'].append(
                    {'regularization': None, 'num_leaves': num_leaves,
                     'train_mse': train_mse, 'test_mse': test_mse})

    with open(result_path, 'w') as f:
        json.dump(cart_results, f)


def run_cv_guide(dataset, train_paths, test_paths):
    result_path = f'experiments/results/cv/{dataset}_guide.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return

    data_path = data_mapping[dataset]['dataset_path']

    df = pd.read_csv(data_path)
    num_samples, num_features = df.shape
    guide_results = {
        'dataset_name': dataset,
        'num_samples': num_samples,
        'num_features': num_features}

    for depth in range(2, 6):
        if f'depth_{depth}' not in guide_results:
            guide_results[f'depth_{depth}'] = {}
        for fold in range(0, 5):
            try:
                guide_out = run_guide(train_paths[fold], test_paths[fold], depth, 300)
            except TimeoutExpired:
                print(f"GUIDE TIMED OUT. Dataset: {dataset}, Depth: {depth}, Fold: {fold + 1}")
                continue
            num_leaves = guide_out['complexity']
            train_mse = guide_out['loss']
            test_mse = guide_out['test_loss']
            if f'fold_{fold + 1}' not in guide_results[f'depth_{depth}']:
                guide_results[f'depth_{depth}'][f'fold_{fold + 1}'] = [
                    {'regularization': None, 'num_leaves': num_leaves,
                     'train_mse': train_mse, 'test_mse': test_mse}]
            else:
                guide_results[f'depth_{depth}'][f'fold_{fold + 1}'].append(
                    {'regularization': None, 'num_leaves': num_leaves,
                     'train_mse': train_mse, 'test_mse': test_mse})

    with open(result_path, 'w') as f:
        json.dump(guide_results, f)


def run_cv_iai(dataset, train_paths, test_paths):
    result_path = f'experiments/results/cv/{dataset}_iai.json'
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

    lambs = [0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001]
    for depth in range(2, 6):
        if f'depth_{depth}' not in iai_results:
            iai_results[f'depth_{depth}'] = {}

        for fold in range(0, 5):
            for reg in lambs:
                try:
                    iai_out = check_output(
                        ["python3", 'script/processors/iai_processor.py',
                         train_paths[fold], str(depth), str(reg), '-t', test_paths[fold]], timeout=300
                    ).decode()
                except TimeoutExpired:
                    print(f"IAI TIMED OUT. Dataset: {dataset}, Depth: {depth}, Regularization: {reg}, Fold: {fold + 1}")
                    continue

                iai_out_fields = parse_iai_output(iai_out)
                num_leaves = iai_out_fields['complexity']
                train_mse = iai_out_fields['loss']
                test_mse = iai_out_fields['test_loss']

                if f'fold_{fold + 1}' not in iai_results[f'depth_{depth}']:
                    iai_results[f'depth_{depth}'][f'fold_{fold + 1}'] = [
                        {'regularization': reg, 'num_leaves': num_leaves,
                         'train_mse': train_mse, 'test_mse': test_mse}]
                else:
                    iai_results[f'depth_{depth}'][f'fold_{fold + 1}'].append({'regularization': reg,
                                                                              'num_leaves': num_leaves,
                                                                              'train_mse': train_mse,
                                                                              'test_mse': test_mse})

    with open(result_path, 'w') as f:
        json.dump(iai_results, f)


def run_cv_evtree(dataset, train_paths, test_paths):
    result_path = f'experiments/results/cv/{dataset}_evtree.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return
    data_path = data_mapping[dataset]['dataset_path']

    df = pd.read_csv(data_path)
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
        evtree_train_paths = []
        evtree_test_paths = []
        for fold in range(5):
            old_train = pd.read_csv(train_paths[fold])
            new_train = pd.DataFrame()
            for col in old_train.columns[:-1]:
                new_train[col] = np.where(old_train[col] == 1, 'yes', 'no')

            data_train = pd.concat([new_train, old_train.iloc[:, -1]], axis=1)
            tmpdirname = Path(tmpdirname)
            data_train.to_csv(tmpdirname / os.path.basename(train_paths[fold]), index=False)
            evtree_train_paths[fold] = str(tmpdirname / os.path.basename(train_paths[fold]))

            old_test = pd.read_csv(test_paths[fold])
            new_test = pd.DataFrame()
            for col in old_test.columns[:-1]:
                new_test[col] = np.where(old_test[col] == 1, 'yes', 'no')

            data_test = pd.concat([new_test, old_test.iloc[:, -1]], axis=1)
            data_test.to_csv(tmpdirname / os.path.basename(test_paths[fold]), index=False)
            evtree_test_paths[fold] = str(tmpdirname / os.path.basename(test_paths[fold]))

        lambs = list(np.arange(0.1, 2.1, 0.1))
        for depth in range(2, 6):
            if f'depth_{depth}' not in evtree_results:
                evtree_results[f'depth_{depth}'] = {}
            for fold in range(0, 5):
                for reg in lambs:
                    try:
                        evtree_out = check_output(
                            ["Rscript", 'script/processors/evtree_processor.R',
                             evtree_train_paths[fold], str(depth), str(reg), '-t', evtree_test_paths[fold]], timeout=300
                        ).decode()
                    except TimeoutExpired:
                        print(
                            f"EVTREE TIMED OUT. Dataset: {dataset}, Depth: {depth}, Regularization: {reg}, Fold: {fold + 1}")
                        continue

                    evtree_out_fields = parse_evtree_output(evtree_out)
                    num_leaves = evtree_out_fields['complexity']
                    train_mse = evtree_out_fields['loss']
                    test_mse = evtree_out_fields['test_loss']

                    if f'fold_{fold + 1}' not in evtree_results[f'depth_{depth}']:
                        evtree_results[f'depth_{depth}'][f'fold_{fold + 1}'] = [
                            {'regularization': reg, 'num_leaves': num_leaves,
                             'train_mse': train_mse, 'test_mse': test_mse}]
                    else:
                        evtree_results[f'depth_{depth}'][f'fold_{fold + 1}'].append({'regularization': reg,
                                                                                     'num_leaves': num_leaves,
                                                                                     'train_mse': train_mse,
                                                                                     'test_mse': test_mse})

    with open(result_path, 'w') as f:
        json.dump(evtree_results, f)


def run_cv_osrt(dataset, train_paths, test_paths):
    result_path = f'experiments/results/cv/{dataset}_osrt.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return
    data_path = data_mapping[dataset]['dataset_path']

    df = pd.read_csv(data_path)
    num_samples, num_features = df.shape
    osrt_results = {
        'dataset_name': dataset,
        'num_samples': num_samples,
        'num_features': num_features}

    lambs = [0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001]
    for depth in range(2, 6):
        if f'depth_{depth}' not in osrt_results:
            osrt_results[f'depth_{depth}'] = {}

        for fold in range(0, 5):
            for reg in lambs:
                try:
                    osrt_out = run_osrt(train_paths[fold], test_paths[fold], depth, reg, 300)
                except TimeoutExpired:
                    print(f"OSRT TIMED OUT. Dataset: {dataset}, Depth: {depth}, Regularization: {reg}")
                    continue

                num_leaves = osrt_out['complexity']
                train_mse = osrt_out['loss']
                test_mse = osrt_out['test_loss']

                if f'fold_{fold + 1}' not in osrt_results[f'depth_{depth}']:
                    osrt_results[f'depth_{depth}'][f'fold_{fold + 1}'] = [
                        {'regularization': reg, 'num_leaves': num_leaves,
                         'train_mse': train_mse, 'test_mse': test_mse}]
                else:
                    osrt_results[f'depth_{depth}'][f'fold_{fold + 1}'].append({'regularization': reg,
                                                                               'num_leaves': num_leaves,
                                                                               'train_mse': train_mse,
                                                                               'test_mse': test_mse})

    with open(result_path, 'w') as f:
        json.dump(osrt_results, f)


def run_cv_experiments(dataset):
    data_path = data_mapping[dataset]['dataset_path']

    with tempfile.TemporaryDirectory(dir='/tmp') as tmpdirname:
        # train and test split
        df = pd.read_csv(data_path)
        root_ext = os.path.basename(data_path).split('.')
        tmpdirname = Path(tmpdirname)
        train_paths = []
        test_paths = []
        for i, e in enumerate(KFold(n_splits=5, shuffle=True, random_state=666).split(df)):
            train_index = e[0]
            test_index = e[1]
            df.iloc[train_index].to_csv(tmpdirname / f'{root_ext[0]}_train_fold_{i + 1}.csv', index=False)
            df.iloc[test_index].to_csv(tmpdirname / f'{root_ext[0]}_test_fold_{i + 1}.csv', index=False)
            train_paths.append(str(tmpdirname / f'{root_ext[0]}_train_fold_{i + 1}.csv'))
            test_paths.append(str(tmpdirname / f'{root_ext[0]}_test_fold_{i + 1}.csv'))

        run_cv_cart(dataset, train_paths, test_paths)
        run_cv_guide(dataset, train_paths, test_paths)
        run_cv_iai(dataset, train_paths, test_paths)
        run_cv_evtree(dataset, train_paths, test_paths)
        run_cv_osrt(dataset, train_paths, test_paths)
