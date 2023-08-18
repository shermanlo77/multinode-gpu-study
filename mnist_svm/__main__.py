"""Fits a SVM with k-fold validation and grid search on the MNIST dataset

usage: mnist_svm.py [-h] [--gpu GPU] [--batch BATCH] [--results RESULTS] n_tuning

Investigates the validation error on a 2D grid of penalty parameters and kernel
parameters. The validation error was obtained using 5-fold cross validation.
The error rate is reported and plotted.

Afterwards, the best penalty and kernel parmaeters are used for a final fit. The
test set error rate is reported.

positional arguments:
  n_tuning           Number of tuning parameters

options:
  -h, --help         show this help message and exit
  --gpu GPU          What GPUs to use. Indicate individual GPUs using integers separated by
                     a comma, eg 0,1,2. Or provide 'all' to use all available GPUs. Defaults
                     to device 0
  --batch BATCH      Number of tuning parameters to validate for a worker before a newone
                     instantiates. Defaults to no re-instantiation
  --results RESULTS  Where to save figures and results, defaults to here
"""

import argparse
import logging
import multiprocessing
import platform
import os

import cupy
import numpy as np

import mnist_svm.evaluate
import mnist_svm.data
import mnist_svm.error
import mnist_svm.model
import mnist_svm.train


if __name__ == "__main__":
    import mnist_mpi
    COMM = mnist_mpi.COMM
    multiprocessing.set_start_method("forkserver")


def get_args():
    """Get and return args"""
    parser = argparse.ArgumentParser(
        description="Fits a SVM with k-fold validation and grid search on the "
                    "MNIST dataset"
    )
    parser.add_argument("n_tuning", type=int,
                        help="Number of tuning parameters")
    parser.add_argument(
        "--gpu",
        help="What GPUs to use. Indicate individual GPUs using integers "
             "seperated by a comma, eg 0,1,2. Or provide 'all' to use all "
             "available GPUs. Defaults to device 0"
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Number of tuning parameters to validate for a worker before a new"
             "one instantiates. Defaults to no re-instantiation"
    )
    parser.add_argument(
        "--results",
        help="Where to save figures and results, defaults to here"
    )
    args = parser.parse_args()
    return args


def main(args):

    n_tuning = args.n_tuning
    gpu_ids = args.gpu
    n_batch = args.batch
    figure_path = args.results
    if gpu_ids is None:
        gpu_ids = [0]
    elif gpu_ids == "all":
        gpu_ids = range(cupy.cuda.runtime.getDeviceCount())
    else:
        gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    if figure_path is None:
        figure_path = "."

    rank = COMM.Get_rank()

    # check figure_path is a directory
    # create directory if there isn't one
    if not os.path.isdir(figure_path) and rank == 0:
        os.mkdir(figure_path)

    logging.info("Rank %i on %s with GPU %s", rank, platform.node(), gpu_ids)

    training_image, training_response, test_image, test_response = (
        mnist_svm.data.get_data())

    # parameters to do grid search on
    log_penalty_parameters = np.linspace(-3, 6, n_tuning)
    log_kernel_parameters = np.linspace(-6, -2, n_tuning)

    validation_error, best_param = mnist_svm.train.grid_search_cv(
        COMM, gpu_ids, training_image, training_response, n_batch,
        log_penalty_parameters, log_kernel_parameters)

    # main thread only
    if rank != 0:
        return

    # save the validation error in case want to replot them
    np.save(os.path.join(figure_path, "validation_error.npy"), validation_error)
    mnist_svm.evaluate.plot_validation_error(
        figure_path, log_penalty_parameters, log_kernel_parameters,
        validation_error)

    # fit the model using the best parameter on the entire training data
    mnist_svm.evaluate.report_test_error(
        gpu_ids[0], best_param, training_image, training_response,
                      test_image, test_response)


if __name__ == "__main__":
    main(get_args())
