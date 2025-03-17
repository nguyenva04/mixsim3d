import os
import argparse

from drp.utils.config import Config

NUM_WORKERS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", help="path to drp dataset")
    parser.add_argument(
        "-e",
        "--epochs",
        help="number of epochs in learning process",
        type=int,
        default=11,
    )
    parser.add_argument(
        "-b", "--batch_size", help="batch size in learning process", type=int, default=8
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="learning rate for optimizer",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "-wd", "--weight_decay", help="regularization", type=float, default=1e-6
    )

    parser.add_argument(
        "-n", "--num_workers", default=NUM_WORKERS, type=int, help="number of devices"
    )

    parser.add_argument(
        "-c",
        "--config",
        help="training config file",
        type=str,
        default=Config.default_path(),
    )
    parser.add_argument(
        "-p",
        "--path_state",
        help="path state dict for fine tuning",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-r", "--run_id", help="mlflow run to continue", type=str, default=None
    )
    parser.add_argument('-nd', '--nodes', default=1,
                        type=int, metavar='N')

    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')

    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    return parser.parse_args()
