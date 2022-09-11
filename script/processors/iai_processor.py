import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from julia.api import Julia
Julia(compiled_modules=False, sysimage='/Users/ruizhang/Downloads/sys.dylib')
from interpretableai import iai


def run_iai(train_dataset_path, test_dataset_path, d, lamb):
    data = pd.read_csv(train_dataset_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y_var = np.var(y)

    if test_dataset_path:
        test_data = pd.read_csv(test_dataset_path)
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        test_var = np.var(y_test)

    start_time = time.time()
    ort = iai.OptimalTreeRegressor(
            random_seed=1,
            num_threads=1,
            normalize_y=True,
            show_progress=False,
            max_depth=d,
            cp=lamb
        )
    ort.fit(X, y)
    duration = time.time() - start_time
    r2 = ort.score(X, y, criterion='mse')
    mse = y_var*(1-r2)
    num_leaves = int(ort.get_num_nodes()/2) + 1
    print('Train Loss:', mse)
    print('Number of Leaves:', num_leaves)
    print('Training Duration:', duration, 'seconds')
    if test_dataset_path:
        r2_test = ort.score(X_test, y_test, criterion='mse')
        mse_test = test_var * (1 - r2_test)
        print('Test Loss:', mse_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=str, help="train path")
    parser.add_argument("depth", type=int, help="depth limit")
    parser.add_argument("reg", type=float, help="regularization")
    parser.add_argument("-t", "--test", type=str, help="test path")

    args = parser.parse_args()
    train_path = args.train
    depth = args.depth
    reg = args.reg
    test_path = args.test

    run_iai(train_path, test_path, depth, reg)