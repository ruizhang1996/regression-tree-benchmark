import numpy as np
import pandas as pd
import re
import os
import json
from subprocess import check_output, TimeoutExpired, run, CalledProcessError
from .mapping import data_mapping
import tempfile
from pathlib import Path
import sys

example_input_file_path = 'experiments/configurations/test.in'

txt_pattern = {
        "train_mse": (r"train_mse: ([\d.]+)", float),
        "test_mse": (r"test_mse: ([\d.]+)", float),
    }


def parse_guide_output(guide_output):
    out = {}
    for i in txt_pattern:
        res = re.search(txt_pattern[i][0], guide_output, re.M)
        if res:
            out[i] = txt_pattern[i][1](res.group(1))
        else:
            out[i] = None
    return out


def run_guide(train_dataset_path, test_dataset_path, depth, timeout):
    with tempfile.TemporaryDirectory(dir='/tmp') as tmpdirname:
        # step1: create input file
        # step1.1 create description file
        tmpdirname = Path(tmpdirname)
        dsc_path = str(tmpdirname / "dsc.txt")

        train_dataset_path = Path(train_dataset_path)
        df = pd.read_csv(str(train_dataset_path))

        with open(dsc_path, 'w') as f:
            f.write('\'' + str(train_dataset_path) + '\'\n')
            f.write('NA\n')
            f.write('2\n')
            for i, col in enumerate(df.columns[:-1]):
                f.write(str(i + 1) + ' ' + '\'' + str(col) + '\'' + ' C\n')

            f.write(str(i + 2) + ' \'' + str(df.columns[-1]) + '\' D\n')

        # step1.2 create input file
        input_file_path = str(tmpdirname / "input.in")
        with open(example_input_file_path) as r:
            lines = r.readlines()
            with open(input_file_path, 'w') as f:
                for i, line in enumerate(lines):
                    if i == 3:
                        line = line.replace("test.out", str(tmpdirname / "test.out"))
                    elif i == 11:
                        line = line.replace("dsc.txt", dsc_path)
                    elif i == 17:
                        line = str(depth) + '    (max. no. split levels)\n'
                    elif i == 24:
                        line = line.replace("tree.R", str(tmpdirname / "tree.R"))

                    f.write(line)

        # step2: run guide command
        with open(input_file_path, 'rb') as f:
            if sys.platform == 'darwin':
                run('./guide_mac', input=f.read(), timeout=timeout)
            elif sys.platform == 'linux':
                run('./guide_linux', input=f.read(), timeout=timeout)
            elif sys.platform == 'win32' or sys.platform == 'cygwin' or sys.platform == 'msys':
                run('guide', input=f.read(), timeout=timeout)

        # step3 modify and run Rscript
        guide_tree_path = str(tmpdirname / "guide_tree.R")
        with open(str(tmpdirname / "tree.R")) as r:
            with open(guide_tree_path, 'w') as f:
                lines = r.readlines()
                for line in lines:
                    if line.startswith('newdata'):
                        new_line = line.replace('\"newdata.csv\"', f'\"{str(train_dataset_path)}\"')
                        f.write(new_line)
                    else:
                        f.write(line)

                f.write('target_col <- tail(names(newdata), n=1)\n')
                f.write(
                    'train_mse <- eval(parse(text = paste(\'mean((as.numeric(newdata$\', target_col, \') - pred)^2)\', sep = \"\")))\n')
                if test_dataset_path:
                    test_dataset_path = str(Path(test_dataset_path).resolve())
                    f.write(f'testdata <- read.csv(\"{test_dataset_path}\",header=TRUE,colClasses=\"character\")\n')
                    f.write(
                        'test_mse <- eval(parse(text = paste(\'mean((as.numeric(testdata$\', target_col, \') - pred)^2)\', sep = \"\")))\n')
                f.write('cat(\"train_mse:\", train_mse)\n')
                if test_dataset_path:
                    f.write('cat(\"test_mse:\", test_mse)\n')

        guide_output = check_output(
            ["Rscript", guide_tree_path], timeout=1800
        ).decode()

        out = {}
        guide_out_fields = parse_guide_output(guide_output)
        train_mse = guide_out_fields['train_mse']
        test_mse = guide_out_fields['test_mse']
        out['loss'] = train_mse
        if test_mse:
            out['test_loss'] = test_mse

        with open(str(tmpdirname / "test.out")) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(' Number of terminal nodes of final tree: '):
                    num_leaves = int(line.split(' Number of terminal nodes of final tree: ')[1][:-1])
                    break

            duration = float(lines[-1].split(' Elapsed time in seconds: ')[1][:-1])

        out['complexity'] = num_leaves
        out['time'] = duration

        return out