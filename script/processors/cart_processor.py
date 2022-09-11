import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def run_cart(train_dataset_path, test_dataset_path, d):
    data = pd.read_csv(train_dataset_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if test_dataset_path:
        test_data = pd.read_csv(test_dataset_path)
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

    start_time = time.time()
    cart = DecisionTreeRegressor(random_state=1, max_depth=d)
    cart.fit(X, y)
    duration = time.time() - start_time
    train_mse = mean_squared_error(y, cart.predict(X))
    num_leaves = cart.get_n_leaves()
    print('Train Loss:', train_mse)
    print('Number of Leaves:', num_leaves)
    print('Training Duration:', duration, 'seconds')
    if test_dataset_path:
        test_mse = mean_squared_error(y_test, cart.predict(X_test))
        print('Test Loss:', test_mse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=str, help="train path")
    parser.add_argument("depth", type=int, help="depth limit")
    parser.add_argument("-t", "--test", type=str, help="test path")

    args = parser.parse_args()
    train_path = args.train
    depth = args.depth
    test_path = args.test

    run_cart(train_path, test_path, depth)


