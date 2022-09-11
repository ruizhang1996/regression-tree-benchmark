from .model.tree_classifier import TreeClassifier
import pandas as pd
import re
import json
import tempfile
from subprocess import check_output, TimeoutExpired
from pathlib import Path
from sklearn.metrics import mean_squared_error
import sys

with open("experiments/configurations/config_template.json") as f:
    config_template = json.load(f)

txt_pattern = {
    "loss_normalizer": (r"loss_normalizer: ([\d.]+)", float),
    "loss": (r"Loss: ([\d.]+)", float),
    "complexity": (r"Complexity: ([\d.]+)", float),
    "time": (r"Training Duration: ([\d.]+) seconds", float),
}


def parse_gosdt_output(gosdt_output):
    out = {}
    for i in txt_pattern:
        out[i] = txt_pattern[i][1](
            re.search(txt_pattern[i][0], gosdt_output, re.M).group(1)
        )
    return out


def compute_mse(model, X, y, loss_normalizer):
    return mean_squared_error(y, model.predict(X) * loss_normalizer)


def run_osrt(train_dataset_path, test_dataset_path, depth_limit, regularization, timeout):
    df = pd.read_csv(train_dataset_path)
    X_train = df[df.columns[:-1]].to_numpy()
    y_train = df[df.columns[-1]].to_numpy()

    if test_dataset_path is None:
        X_test = None
        y_test = None
    else:
        df = pd.read_csv(test_dataset_path)
        X_test = df[df.columns[:-1]].to_numpy()
        y_test = df[df.columns[-1]].to_numpy()

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)

        model_output_path = str(tmpdirname / "output_models.json")
        config_path = str(tmpdirname / "config.json")

        config = config_template.copy()
        config["regularization"] = regularization
        config["depth_budget"] = depth_limit + 1
        config["model"] = model_output_path

        if X_train.shape[0] > 1000:
            config["k_cluster"] = False  # for large dataset only

        with open(config_path, "w") as f:
            json.dump(config, f)

        if sys.platform == 'darwin':
            gosdt_out = check_output(
                ["gosdt", train_dataset_path, config_path], timeout=timeout
            ).decode()
        elif sys.platform == 'linux':
            gosdt_out = check_output(
                ["./gosdt", train_dataset_path, config_path], timeout=timeout
            ).decode()
        elif sys.platform == 'win32' or sys.platform == 'cygwin' or sys.platform == 'msys':
            gosdt_out = check_output(
                ["./gosdt", train_dataset_path, config_path], timeout=timeout
            ).decode()

        with open(model_output_path) as f:
            models = json.load(f)

    if len(models) == 0:
        return

    out = {}
    gosdt_out_fields = parse_gosdt_output(gosdt_out)
    loss_normalizer = gosdt_out_fields["loss_normalizer"]
    gosdt_time = gosdt_out_fields["time"]

    mse_train = compute_mse(TreeClassifier(models[0]), X_train, y_train, loss_normalizer)
    num_leaves = TreeClassifier(models[0]).leaves()
    if X_test is not None:
        mse_test = compute_mse(TreeClassifier(models[0]), X_test, y_test, loss_normalizer)
        for i in range(1, len(models)):
            model = TreeClassifier(models[i])
            tmp = compute_mse(model, X_test, y_test, loss_normalizer)
            if tmp < mse_test:
                mse_test = tmp

    if X_test is not None:
        out["test_loss"] = mse_test

    out["time"] = gosdt_time
    out["complexity"] = num_leaves
    out["loss"] = mse_train

    return out
