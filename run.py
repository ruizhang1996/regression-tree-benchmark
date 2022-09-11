import argparse
from script.run_loss import run_loss_experiments
from script.run_time import run_time_experiments
from script.run_cv import run_cv_experiments
from script.run_controllability import run_variance_experiments
from plotting.plot_loss import plot_loss_dataset_depth
from plotting.plot_controllability import plot_controllability_dataset_depth
from plotting.plot_time import plot_time_dataset_depth
from plotting.plot_cv import plot_cv_dataset_depth
from plotting.plot_scalability import plot_scalability_dataset_depth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", type=str, help="benchmarking command")
    parser.add_argument('-d', '--datasets', default=[], nargs='+')

    args = parser.parse_args()
    command = args.cmd
    datasets = args.datasets

    if command == 'loss':
        if len(datasets) == 0:
            datasets = ['airfoil', 'airquality', 'real-estate', 'seoul-bike', 'servo', 'sync', 'yacht', 'insurance']

        for dataset in datasets:
            run_loss_experiments(dataset)
            for depth in range(2, 10):
                plot_loss_dataset_depth(dataset, depth)

    elif command == 'controllability':
        if len(datasets) == 0:
            datasets = ['real-estate', 'servo']

        for dataset in datasets:
            run_variance_experiments(dataset)
            for depth in range(4, 10):
                plot_controllability_dataset_depth(dataset, depth)

    elif command == 'time':
        if len(datasets) == 0:
            datasets = ['airfoil', 'real-estate', 'servo', 'sync', 'yacht', 'insurance']

        for dataset in datasets:
            run_time_experiments(dataset)
            for depth in range(2, 10):
                plot_time_dataset_depth(dataset, depth)

    elif command == 'cv':
        if len(datasets) == 0:
            datasets = ['airfoil', 'sync', 'servo', 'seoul-bike', 'insurance', 'enb-heat']

        for dataset in datasets:
            run_cv_experiments(dataset)
            for depth in range(2, 6):
                plot_cv_dataset_depth(dataset, depth, test=True)
                plot_cv_dataset_depth(dataset, depth, test=False)

    elif command == 'scalability':

        datasets = ['household']

        for dataset in datasets:
            plot_scalability_dataset_depth(dataset)

    else:
        print('INVALID COMMAND')
