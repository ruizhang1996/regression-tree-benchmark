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


def run_scalability_cart(dataset):
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

    result_path = f'experiments/results/scalability/{dataset}_cart.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return

    num_samples = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    with tempfile.TemporaryDirectory(dir='/tmp') as tmpdirname:
        tmpdirname = Path(tmpdirname)
        data_path = data_mapping[dataset]['dataset_path']
        df = pd.read_csv(data_path)
        num_samples.append(df.shape[0])
        cart_results = {'results': []}
        for num_sample in num_samples:
            sub_df = df.iloc[:num_sample, :]
            sub_sample_path = tmpdirname / f'{num_sample}.csv'
            sub_df.to_csv(sub_sample_path, index=False)

            try:
                cart_out = check_output(
                    ["python3", 'script/processors/cart_processor.py', str(sub_sample_path), str(5)], timeout=1800
                ).decode()
                cart_out_fields = parse_cart_output(cart_out)
                duration = cart_out_fields['time']
            except TimeoutExpired:
                print(f"CART TIMED OUT. Dataset: {dataset}, Samples: {num_sample}")
                duration = 1800

            cart_results['results'].append({'num_samples': num_sample, 'time': duration})

    with open(result_path, 'w') as f:
        json.dump(cart_results, f)


def run_scalability_guide(dataset):
    result_path = f'experiments/results/scalability/{dataset}_guide.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return

    num_samples = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    with tempfile.TemporaryDirectory(dir='/tmp') as tmpdirname:
        tmpdirname = Path(tmpdirname)
        data_path = data_mapping[dataset]['dataset_path']
        df = pd.read_csv(data_path)
        num_samples.append(df.shape[0])
        guide_results = {'results': []}
        for num_sample in num_samples:
            sub_df = df.iloc[:num_sample, :]
            sub_sample_path = tmpdirname / f'{num_sample}.csv'
            sub_df.to_csv(sub_sample_path, index=False)
            try:
                guide_out = run_guide(sub_sample_path, None, 5, 1800)
                duration = guide_out['time']
            except TimeoutExpired:
                print(f"GUIDE TIMED OUT. Dataset: {dataset}, Samples: {num_sample}")
                duration = 1800

            guide_results['results'].append({'num_samples': num_sample, 'time': duration})

    with open(result_path, 'w') as f:
        json.dump(guide_results, f)


def run_scalability_iai(dataset):
    result_path = f'experiments/results/scalability/{dataset}_iai.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return

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

    num_samples = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    with tempfile.TemporaryDirectory(dir='/tmp') as tmpdirname:
        tmpdirname = Path(tmpdirname)
        data_path = data_mapping[dataset]['dataset_path']
        df = pd.read_csv(data_path)
        num_samples.append(df.shape[0])
        iai_results = {'results': []}
        for num_sample in num_samples:
            sub_df = df.iloc[:num_sample, :]
            sub_sample_path = tmpdirname / f'{num_sample}.csv'
            sub_df.to_csv(sub_sample_path, index=False)
            try:
                iai_out = check_output(
                    ["python3", 'script/processors/iai_processor.py', sub_sample_path, str(5), str(0.01)], timeout=1800
                ).decode()
                iai_out_fields = parse_iai_output(iai_out)
                duration = iai_out_fields['time']
            except TimeoutExpired:
                print(f"IAI TIMED OUT. Dataset: {dataset}, Samples: {num_sample}")
                duration = 1800

            iai_results['results'].append({'num_samples': num_sample, 'time': duration})

    with open(result_path, 'w') as f:
        json.dump(iai_results, f)


def run_scalability_evtree(dataset):
    result_path = f'experiments/results/scalability/{dataset}_evtree.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return

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

    num_samples = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    with tempfile.TemporaryDirectory(dir='/tmp') as tmpdirname:
        tmpdirname = Path(tmpdirname)
        data_path = data_mapping[dataset]['dataset_path']
        df = pd.read_csv(data_path)
        num_samples.append(df.shape[0])
        evtree_results = {'results': []}
        for num_sample in num_samples:
            sub_df = df.iloc[:num_sample, :]
            sub_sample_path = tmpdirname / f'{num_sample}.csv'
            sub_df_new = pd.DataFrame()
            for col in sub_df.columns[:-1]:
                sub_df_new[col] = np.where(sub_df[col] == 1, 'yes', 'no')

            data = pd.concat([sub_df_new, sub_df.iloc[:, -1]], axis=1)
            data.to_csv(sub_sample_path, index=False)
            try:
                evtree_out = check_output(
                    ["Rscript", 'script/processors/evtree_processor.R', sub_sample_path, str(5), str(0.2)], timeout=1800
                ).decode()
                evtree_out_fields = parse_evtree_output(evtree_out)
                duration = evtree_out_fields['time']
            except TimeoutExpired:
                print(f"EVTREE TIMED OUT. Dataset: {dataset}, Samples: {num_sample}")
                duration = 1800

            evtree_results['results'].append({'num_samples': num_sample, 'time': duration})

    with open(result_path, 'w') as f:
        json.dump(evtree_results, f)


def run_scalability_osrt(dataset):
    result_path = f'experiments/results/scalability/{dataset}_osrt.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        return

    num_samples = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    with tempfile.TemporaryDirectory(dir='/tmp') as tmpdirname:
        tmpdirname = Path(tmpdirname)
        data_path = data_mapping[dataset]['dataset_path']
        df = pd.read_csv(data_path)
        num_samples.append(df.shape[0])
        osrt_results = {'results': []}
        for num_sample in num_samples:
            sub_df = df.iloc[:num_sample, :]
            sub_sample_path = tmpdirname / f'{num_sample}.csv'
            sub_df.to_csv(sub_sample_path, index=False)

            try:
                osrt_out = run_osrt(sub_sample_path, None, 5, 0.01, 1800)
                duration = osrt_out['time']
            except TimeoutExpired:
                print(f"OSRT TIMED OUT. Dataset: {dataset}, Samples: {num_sample}")
                duration = 1800

            osrt_results['results'].append({'num_samples': num_sample, 'time': duration})

    with open(result_path, 'w') as f:
        json.dump(osrt_results, f)


def run_scalability_experiments(dataset):
    run_scalability_cart(dataset)
    run_scalability_guide(dataset)
    run_scalability_iai(dataset)
    run_scalability_evtree(dataset)
    run_scalability_osrt(dataset)
