"""Training the model using multiple GPUs

Does grid search k-fold cross validation. Using hybrid mpi4py (to distribute one
process per node) and multiprocessing (uses GPUs on a node).

Ensure __main__ only has set the multiprocessing start method to be either spawn
or forkserver.
"""

import itertools
import logging
import multiprocessing
import queue
import time

import cupy
from cupy import cuda
import numpy as np

import mnist_svm.error
import mnist_svm.model

# Number of folds when doing k-fold CV
K_FOLD = 5


def grid_search_cv(comm, gpu_ids, training_image, training_response, n_batch,
                   log_penalty_parameters, log_kernel_parameters):
    """Grid search cross-validation

    For every log penalty and log kernel parameter pair, do 5 fold cross
    validation and report the validation error. Also return the penalty
    and kernel parameter with the lowest validation error

    Uses mpi to split the grid of parameters to grid on between workers

    Args:
        comm (mpi4py.MPI.Intracomm): MPI.COMM_WORLD()
        gpu_ids (list): list of gpu ids
        training_image (np.ndarray): dim n_training_image, n_pixel
        training_response (np.ndarray): dim n_training_image, contains class of
            each image as ints
        n_batch (int): number of tuning parameters to validate for a worker
            before a new one instantiates
        log_penalty_parameters (np.ndarray): vector of log_10 penalty parameters
            to do grid search on
        log_kernel_parameters (np.ndarray): vector of log_10 kernel parameters
            to do grid search on

    Returns:
        np.ndarray: only for RANK 0, validation error, dim for each kernel
            parameter, for each penalty parameter
        list: only for RANK 0, length 2, penalty parameter and kernel parameter
            with the lowest validation error
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    penalty_parameters = np.power(10, log_penalty_parameters)
    kernel_parameters = np.power(10, log_kernel_parameters)

    # create grid to search on
    penalty_grid, kernel_grid = np.meshgrid(penalty_parameters,
                                            kernel_parameters)
    penalty_grid = penalty_grid.flatten()
    kernel_grid = kernel_grid.flatten()
    parameter_grid = zip(penalty_grid, kernel_grid)

    # split tuning parameters between each process
    rank_parameter_grid = []
    parameter_split_index = np.round(
        np.linspace(0, len(penalty_parameters)*len(kernel_parameters), size+1))
    parameter_split_index = parameter_split_index.astype(np.int32)
    rank_parameter_grid = itertools.islice(parameter_grid,
                                           parameter_split_index[rank],
                                           parameter_split_index[rank+1])
    rank_parameter_grid = list(rank_parameter_grid)

    # benchmark and do k fold validation
    time_start = time.perf_counter()
    mpi_validation_error = k_fold_validation(
        rank, gpu_ids, training_image, training_response, rank_parameter_grid,
        n_batch)
    time_taken = time.perf_counter() - time_start

    validation_error = comm.gather(mpi_validation_error, root=0)

    if rank == 0:
        print(f"Time taken for k-fold validation: {time_taken} s")
    else:
        return None, None

    # form a matrix from all collected validation errors
    validation_error = np.concatenate(validation_error, 1)
    validation_error_reshape = [
        validation_error[i].reshape(
            len(log_kernel_parameters), len(log_penalty_parameters))
        for i in range(K_FOLD)
    ]

    validation_error = np.asarray(validation_error_reshape)

    # select the parameter with the best validation error
    min_index = np.argmin(np.mean(validation_error, 0))
    best_param = [penalty_grid[min_index], kernel_grid[min_index]]

    return validation_error, best_param


def k_fold_validation(rank, gpu_ids, full_train_image, full_train_response,
                      tuning_grid, n_batch):
    """Do k fold cross validation and return the validation errors

    Do k fold cross validation and return the validation error for each fold
    and tuning parameter

    Uses multiprocessing to use multiple GPUs

    Args:
        rank (int): mpi rank
        gpu_ids (list): list of gpu ids
        full_train_image (np.ndarray): training images, dim n_train_image,
            n_pixel
        full_train_response (np.ndarray): training responses, dim n_train_image
        tuning_grid (list): nested list of tuning parameters, each element
            is a list of size two containing the penalty and kernel parameter
        n_batch (int): number of tuning parameters to validate for a worker
            before a new one instantiates

    Returns:
        np.ndarray: dim n_fold, len(tuning_parameters), contains the validation
            error rate for each fold and tuning parameter
    """

    if n_batch is None:
        n_batch = len(tuning_grid)

    # queue for storing the error rate for each tuning parameter
    # each item is a list of length 2 containing the tuning parameter index and
    # the error rate
    output_queue = multiprocessing.Queue()
    validation_error = np.empty((K_FOLD, len(tuning_grid)))
    # queue of GPUs ids whom have finished their validation batch
    is_finish_queue = multiprocessing.Queue()

    # for each fold
    for i_fold, (train_image, train_response, valid_image, valid_response) in (
        enumerate(k_folds(full_train_image, full_train_response, K_FOLD))):

        # put all tuning parameters in a queue
        tuning_queue = multiprocessing.Queue()
        for i, param in enumerate(tuning_grid):
            tuning_queue.put([i, param])

        # perpare threads
        trainer_list = [
            Trainer(rank, gpu_id, train_image, train_response,
                    valid_image, valid_response, tuning_queue,
                    output_queue, n_batch, is_finish_queue)
            for gpu_id in gpu_ids
        ]
        # start all threads
        for trainer in trainer_list:
            trainer.start()

        # main thread handles all workers
        # if a thread/GPU finishes their batch, init a new worker
        # init a new worker clears GPU memory
        #
        # also note all items put in the queue by a child should be consumed
        # see documentation for multiprocessing.Queue
        while True:
            gpu_id = is_finish_queue.get()
            flush_output_queue(i_fold, validation_error, output_queue)
            trainer_list[gpu_id].join()
            if not tuning_queue.empty():
                trainer_list[gpu_id].close()
                trainer = Trainer(rank, gpu_id, train_image, train_response,
                                  valid_image, valid_response, tuning_queue,
                                  output_queue, n_batch, is_finish_queue)
                trainer_list[gpu_id] = trainer
                trainer.start()
            else:
                n_gpu = 1
                while n_gpu < len(gpu_ids):
                    gpu_id = is_finish_queue.get()
                    flush_output_queue(i_fold, validation_error, output_queue)
                    trainer_list[gpu_id].join()
                    n_gpu += 1
                break

        # synch here
        for trainer in trainer_list:
            trainer.join()
        for trainer in trainer_list:
            trainer.close()

        # collect validation errors from all workers
        while True:
            try:
                i_param, error = output_queue.get(False)
            except queue.Empty:
                break
            validation_error[i_fold, i_param] = error

        logging.info("Rank %i done fold %i", rank, i_fold)

    return validation_error


def flush_output_queue(i_fold, validation_error, output_queue):
    """Extract validation errors from children and store them

    Args:
        i_fold (int): fold number
        validation_error (np.ndarray): n_fold x n_tuning_parameter matrix,
            modified to contain the most result validation results
        output_queue (mutiprocessing.Queue): where all children put their
            validation errors
    """
    while True:
        try:
            i_param, error = output_queue.get(False)
        except queue.Empty:
            break
        validation_error[i_fold, i_param] = error


def k_folds(full_training_image, full_training_response, n_fold):
    """Iterator and yield the training and test data for each fold

    Split the data into k folds. Each iteration allocates a fold as the test set
    and the rest as the training. This cycles for each iteration.

    Args:
        full_training_image (np.ndarray): training images, dim n_image, n_pixel
        full_training_response (np.ndarray): training responses, dim n_image
        n_fold (int): number of folds

    Yields:
        np.ndarray: training_image, dim <n_training_image, n_pixel
        np.ndarray: training_response, dim <n_training_image
        np.ndarray: validation_image, dim <n_training_image, n_pixel
        np.ndarray: validation_response, dim <n_training_image
    """

    # split into equally sized folds
    fold_linspace = np.linspace(0, len(full_training_image), n_fold+1)
    fold_linspace = np.round(fold_linspace).astype(np.int32)

    # list of lists of length 2, for indexing each fold
    fold_index = [
        fold_linspace[i_fold-1:i_fold+1].tolist()
        for i_fold in range(1, len(fold_linspace))
    ]

    for fold_validation in fold_index:
        # get the validation set
        validation_index = slice(fold_validation[0], fold_validation[1])
        validation_image = full_training_image[validation_index]
        validation_response = full_training_response[validation_index]

        # go through each fold and collect the folds used for training
        training_image = np.empty((0, validation_image.shape[1]))
        training_response = np.empty(0)

        for fold_training in fold_index:
            if fold_training is fold_validation:
                continue
            training_index = slice(fold_training[0], fold_training[1])
            training_image = np.concatenate(
                [training_image, full_training_image[training_index]])
            training_response = np.concatenate(
                [training_response, full_training_response[training_index]])

        yield (training_image, training_response,
               validation_image, validation_response)


class Trainer(multiprocessing.Process):
    """Worker for fitting the model tuning parameters and reporting the error

    Worker going through all tuning parameters in the queue. It fits the model
    using the tuning parameter and puts the error rate in the output queue.

    Attributes:
        _rank (int): mpi rank
        _gpu (int): which GPU to use
        _training_image (np.ndarray): dim n_training_image, n_pixel
        _training_response (np.ndarray): dim n_training_image
        _test_image (np.ndarray): dim n_test_image, n_pixel
        _test_response (np.ndarray): dim n_test_image
        _tuning_queue (multiprocessing.Queue): tuning parameters to fit,
            this is shared with all workers
        _output_queue (multiprocessing.Queue): output queue, to put the
            resulting error. Each item is a list of length 2 containing the
            tuning parameter index and the error rate
        _n_batch (int): number of tuning parameters to validate
        _is_finish_queue (multiprocessing.Queue): a queue to place gpu id
            when finished
    """

    def __init__(self, rank, gpu, training_image, training_response, test_image,
                 test_response, tuning_queue, output_queue, n_batch,
                 is_finish_queue):
        """For fitting the model tuning parameters and reporting the error

        Args:
            See attributes
        """
        super().__init__()
        self._rank = rank
        self._gpu = gpu
        self._training_image = training_image
        self._training_response = training_response
        self._test_image = test_image
        self._test_response = test_response
        self._tuning_queue = tuning_queue
        self._output_queue = output_queue
        self._n_batch = n_batch
        self._is_finish_queue = is_finish_queue

    def run(self):
        with cuda.Device(self._gpu):
            logging.info("Start rank %i GPU %i", self._rank, self._gpu)
            self._fit()
            logging.info("Finish rank %i GPU %i", self._rank, self._gpu)
        self._is_finish_queue.put(self._gpu)

    def _fit(self):
        """Fit the model for all tuning parameters and report the error rate

        Fit the model for all tuning parameters and report the error rate in
        the output queue. This continues until there are no more tuning
        parameters
        """
        training_image = cupy.asarray(self._training_image)
        training_response = cupy.asarray(self._training_response)
        test_image = cupy.asarray(self._test_image)
        test_response = cupy.asarray(self._test_response)
        for _ in range(self._n_batch):
            try:
                index, tuning = self._tuning_queue.get(False)
            except queue.Empty:
                break
            model = mnist_svm.model.get_model(tuning)
            model.fit(training_image, training_response)
            error = mnist_svm.error.get_error(model, test_image, test_response)
            self._output_queue.put([index, error])
