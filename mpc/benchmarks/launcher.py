#!/usr/bin/env python3

# pyre-unsafe

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run tfe_benchmarks example in multiprocess mode:

$ python3 examples/tfe_benchmarks/launcher.py --multiprocess

To run tfe_benchmarks example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
      --ssh_key_file=$HOME/.aws/fair-$USER.pem \
      --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
      --aux_files=examples/tfe_benchmarks/tfe_benchmarks.py \
      examples/tfe_benchmarks/launcher.py
"""

import argparse
import logging
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from multiprocess_launcher import MultiProcessLauncher

parser = argparse.ArgumentParser(description="CrypTen Benchmarks")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)

parser.add_argument(
    "--epochs", default=20, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument("--momentum", default=0.99, type=float, metavar="M", help="momentum")

parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)

parser.add_argument(
    "--loss-func",
    default="ce",
    type=str,
    help="choose loss function: mse or ce(default is ce)",
)

parser.add_argument(
    "--multiprocess",
    default=False,
    action="store_true",
    help="Run example in multiprocess mode",
)


def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    # pyre-fixme[21]: Could not find module `tfe_benchmarks`.
    from mpc_benchmarks import run_benchmarks  # @manual

    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    run_benchmarks(
        args.epochs,
        args.start_epoch,
        args.batch_size,
        args.lr,
        args.momentum,
        args.loss_func,
        args.seed,
    )


def main(run_experiment):
    args = parser.parse_args()

    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)
