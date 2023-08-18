"""Fits an ensemble of neural networks

usage: mnist_nn.py [-h] [--gpu GPU] [--results RESULTS] [--seed SEED] n_bootstrap n_tuning

Fits an ensemble of neural networks. Each neural network is obtained by using
a bootstrap (within fold) sample and doing a random search of tuning parameters.
The tuning parameter with the best 5-fold cross validation loss is used.

The test error rate is reported for each neural network along with the ensemble
error rate

positional arguments:
  n_bootstrap        Number of bootstrap samples
  n_tuning           Number of tuning parameters to search

options:
  -h, --help         show this help message and exit
  --gpu GPU          What GPUs to use. Indicate individual GPUs using integers separated by
                     a comma, eg 0,1,2. Or provide 'all' to use all available GPUs
  --results RESULTS  Where to save figures and results, defaults to here
  --seed SEED        Integer seed for random number generation
"""

import argparse
import logging
import multiprocessing
import os
import platform
import time

import torch

import mnist
import mnist_nn.evaluate
import mnist_nn.train

if __name__ == "__main__":
    import mnist_mpi
    COMM = mnist_mpi.COMM
    multiprocessing.set_start_method("forkserver")

DOWNLOAD_PATH = "../"

SEED = 169514335042358758700408908237337905507


def get_args():
    parser = argparse.ArgumentParser(
        description="Fits an ensemble of neural networks"
    )
    parser.add_argument("n_bootstrap", type=int,
                        help="Number of bootstrap samples")
    parser.add_argument("n_tuning", type=int,
                        help="Number of tuning parameters to search")
    parser.add_argument(
        "--gpu",
        help="What GPUs to use. Indicate individual GPUs using integers "
             "seperated by a comma, eg 0,1,2. Or provide 'all' to use all "
             "available GPUs")
    parser.add_argument(
        "--results",
        help="Where to save figures and results, defaults to here"
    )
    parser.add_argument(
        "--seed", type=int,
        help="Integer seed for random number generation"
    )
    args = parser.parse_args()
    return args


def main(args):

    n_bootstrap = args.n_bootstrap
    n_tuning = args.n_tuning
    gpu_ids = args.gpu
    figure_path = args.results
    seed = args.seed
    if gpu_ids is None:
        gpu_ids = [0]
    elif gpu_ids == "all":
        gpu_ids = range(torch.cuda.device_count())
    else:
        gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    if figure_path is None:
        figure_path = "."
    if seed is None:
        seed = SEED

    rank = COMM.Get_rank()

    # check figure_path is a directory
    # create directory if there isn't one
    if not os.path.isdir(figure_path) and rank == 0:
        os.mkdir(figure_path)

    logging.info("Rank %i on %s with GPU %s", rank, platform.node(), gpu_ids)

    # load the data
    training_data, test_data = mnist.get_data(mnist.TRANSFORM)

    # train ensemble
    time_start = time.perf_counter()
    models, results = mnist_nn.train.train_ensemble(
        COMM, gpu_ids, training_data, n_bootstrap, n_tuning, seed)

    gathered_results = mnist_nn.train.gather_results(COMM, results)

    if rank == 0:
        time_taken = time.perf_counter() - time_start
        print(f"Time taken {time_taken} s")

    # final evaluation of the ensemble of models
    device = torch.device("cuda")
    mnist_nn.evaluate.eval_ensemble(COMM, device, gathered_results, models,
                                    training_data, test_data, figure_path)
    mnist_nn.evaluate.eval_errn(COMM, device, models, mnist.TRANSFORM,
                                figure_path)

if __name__ == "__main__":
    main(get_args())
