"""Training an ensemble using multiple GPUs

Does bagging with random search and k-fold cross validation, using hybrid mpi4py
(to distribute one process per node) and multiprocessing (uses GPUs on a node).

Ensure __main__ only has set the multiprocessing start method to be either spawn
or forkserver.
"""

import logging
import math
import multiprocessing
import queue

import numpy as np
from numpy import random
import pandas as pd
import torch

import mnist_nn.error
import mnist_nn.model
import mnist_nn.sampling

# maximum number of cycles through the training set when fitting
N_MAX_EPOCH = 20
# stop fitting when the loss does not decrease by this factor
TOLERANCE = 0.8
# number of data points used for each gradient step
N_BATCH = 10
K_FOLD = 5


def train_ensemble(comm, gpu_ids, training_data, n_bootstrap, n_tuning, seed):
    """Train an ensemble of models

    For each bootstrap sample, do a random search for the best parameter. For
    each parameter, the validation error was obtained using k-fold CV. For each
    bootstrap sample, the parameter with the best validation error is chosen.
    This function then returns the best model found for each boostrap sample.

    mpi4py is used to distribute seeds across mpi processes. multiprocessing is
    used to use multiple GPUs on a node. Each bootstrap sample has its own seed.
    Each process will work through all of the seeds/bootstrap samples

    Args:
        comm (mpi4py.MPI.Intracomm): MPI.COMM_WORLD()
        gpu_ids (list): list of GPU devices to use
        training_data (torchvision.datasets)
        n_bootstrap (int): number of bootstrap samples
        n_tuning (int): for each bootstrap sample, the number of tuning
            parameters to randomly try out
        seed (int): seed for random number generation

    Returns:
        dict: dictionary of models for each bootstrap sample for this MPI rank
            key: int to identify the bootstrap sample
            value: (Net) best model found for this bootstrap sample
        pandas.core.frame.DataFrame: dataframe of specs for each model for this
            MPI rank
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    seeds = random.SeedSequence(seed).spawn(n_bootstrap)

    # each bootstrap sample has a seed
    # distribute seed between all mpi processes
    seed_index_rank = np.round(np.linspace(0, n_bootstrap, size+1))
    seed_index_rank = seed_index_rank.astype(np.int32)
    seed_queue = multiprocessing.Queue()
    seeds_rank = seeds[seed_index_rank[rank]: seed_index_rank[rank+1]]
    for seed_i in seeds_rank:
        seed_queue.put((seed_i.spawn_key[0], seed_i))

    # queue to put results in and the found model
    output_queue = multiprocessing.Queue()
    # barrier required to ensure all objects have been retrived from the queue
    # before the Process dies
    # torch.nn.Module objects in a queue do break when the Process dies
    exit_barrier = multiprocessing.Barrier(len(gpu_ids)+1)
    workers = [
        RandomSearch(rank, gpu_id, training_data, n_tuning, seed_queue,
                     output_queue, exit_barrier)
        for gpu_id in gpu_ids
    ]

    for worker in workers:
        worker.start()

    models = {}
    columns = [
        "n_conv_layer",
        "kernel_size",
        "n_hidden_layer",
        "learning_rate",
        "momentum",
    ]
    results = pd.DataFrame(columns=columns)

    # main thread waits and retrives results from workers
    for _ in range(len(seeds_rank)):
        net_id, net, best_param = output_queue.get()
        results.loc[net_id, "n_conv_layer"] = best_param.n_conv_layer
        results.loc[net_id, "kernel_size"] = best_param.kernel_size
        results.loc[net_id, "n_hidden_layer"] = best_param.n_hidden_layer
        results.loc[net_id, "learning_rate"] = best_param.learning_rate
        results.loc[net_id, "momentum"] = best_param.momentum
        models[net_id] = net

    # barrier ensures processes are alive when retriving from the queue
    exit_barrier.wait()
    for worker in workers:
        worker.join()

    return models, results


def train_model(device, data_loader, param, loss):
    """Train a model given parameters

    Args:
        device (torch.device): device to run on
        data_loader (torch.utils.data.DataLoader): contains the training data
        param (Parameters): parameters to use
        loss (torch.nn.CrossEntropyLoss): loss function

    Returns:
        Net: the fitted model
    """
    net = mnist_nn.model.Net(param.n_conv_layer, param.kernel_size,
                             param.n_hidden_layer)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=param.learning_rate,
                                momentum=param.momentum)
    net.train(True)
    loss_before = math.inf

    for _ in range(N_MAX_EPOCH):
        loss_epoch = 0.0
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_i = loss(outputs, labels)
            loss_i.backward()
            optimizer.step()
            loss_epoch += loss_i
        if loss_epoch > loss_before * TOLERANCE:
            break
        loss_before = loss_epoch

    net.eval()
    return net


def gather_results(comm, results):
    """Gather the results from train_ensemble() from all MPI processes

    Args:
        comm (mpi4py.MPI.Intracomm): MPI.COMM_WORLD()
        results (pandas.core.frame.DataFrame): dataframe of specs for each model
            for this MPI rank, output of train_ensemble()

    Returns:
        pandas.core.frame.DataFrame: results from all MPI processes if rank 0,
            else None
    """
    rank = comm.Get_rank()
    gathered_results = comm.gather(results, root=0)
    if rank == 0:
        results = pd.concat(gathered_results)
        return results
    return None


class RandomSearch(multiprocessing.Process):
    """For each bootstrap, random search tuning parameters and fit Net

    For each seed, do a bootstrap (within fold) on the training data. Then
    randomly search of tuning parameters and do k-fold cross validation to
    evaluate the tuning parameter using the loss function. The best tuning
    parameters is used to do a fit using the entire training data

    This continues until all seeds in the Queue have been completed

    Attributes:
        _rank (int): mpi rank
        _gpu_id (int): device number
        _data (torch.utils.data.Dataset): training data, not bootstrapped
        _n_tuning (int): number of random tuning parameters to search for
        _seeds_queue (multiprocessing.Queue): queue of seeds to use
        _output_queue (multiprocessing.Queue): queue to put fitted models
            and results into
        _exit_barrier (multiprocessing.Barrier): barrier to wait on when
            finished
    """

    def __init__(self, rank, gpu_id, data, n_tuning, seeds_queue, output_queue,
                 exit_barrier):
        """
        Args:
            See attributes
        """
        super().__init__()
        self._rank = rank
        self._gpu_id = gpu_id
        self._data = data
        self._n_tuning = n_tuning
        self._seeds_queue = seeds_queue
        self._output_queue = output_queue
        self._exit_barrier = exit_barrier

    def run(self):
        device = torch.device(f"cuda:{self._gpu_id}")
        torch.backends.cudnn.deterministic = True

        # calculate fold boundaries
        fold_linspace = np.linspace(0, len(self._data), K_FOLD+1)
        fold_linspace = np.round(fold_linspace).astype(np.int32)
        while True:
            # continue fitting until there are no more seeds
            try:
                seed_id, seed = self._seeds_queue.get(False)
                rng = random.default_rng(seed)
                torch.manual_seed(
                    int.from_bytes(rng.bytes(8), byteorder='big', signed=True))
                torch.cuda.manual_seed(
                    int.from_bytes(rng.bytes(8), byteorder='big', signed=True))
                bootstrap = mnist_nn.sampling.Bootstrap(self._data,
                                                        fold_linspace, rng)
                net, best_param = self._random_search(device, bootstrap, rng)
                net.cpu()
                self._output_queue.put((seed_id, net, best_param))
                logging.info("Rank %i GPU %i done seed %i",
                             self._rank, self._gpu_id, seed_id)
            except queue.Empty:
                break
        self._exit_barrier.wait()

    def _random_search(self, device, bootstrap, rng):
        """For a bootstrap sample, do random search of tuning parameters

        For a bootstrap sample, do random search of tuning parameters. For each
        parameters, they are evaluated using k-fold cross validation.
        Afterwards, the best parameter is used to fit the model onto and
        returned

        Args:
            device (torch.device): device to run on
            bootstrap (Bootstrap): the bootstrapped dataset to fit onto
            rng (numpy.random._generator.Generator)

        Returns:
            Net: the fitted model using the best parameters found
            Parameters: the best parameters found
        """
        best_param = None
        best_loss = math.inf
        for _ in range(self._n_tuning):
            # random sample of parameters
            param = mnist_nn.model.random_parameter(rng)
            loss = self._k_fold_validation(device, bootstrap, K_FOLD, param)
            if loss < best_loss:
                best_loss = loss
                best_param = param

        # train model using the best parameters found
        loss_fun = mnist_nn.error.get_loss_func()
        training_loader = torch.utils.data.DataLoader(bootstrap,
                                                      batch_size=N_BATCH)
        net = train_model(device, training_loader, best_param, loss_fun)
        return net, best_param

    def _k_fold_validation(self, device, bootstrap, k_fold, param):
        """Given a bootstrap dataset and param, return the validation loss

        Args:
            device (torch.device): device to run on
            bootstrap (Bootstrap): the bootstrapped dataset to validate on
            k_fold (int): number of folds
            param (Parameters): the parameter to use

        Returns:
            float: the mean loss over folds
        """
        losses = [
            self._get_validation_loss(device, bootstrap, k_fold, fold_id, param)
            for fold_id in range(k_fold)
        ]
        return np.mean(losses)

    def _get_validation_loss(self, device, bootstrap, k_fold, fold_id, param):
        """Work out the validation loss for a fold

        Args:
            device (torch.device): device to run on
            bootstrap (Bootstrap): the bootstrapped dataset to validate on
            k_fold (int): number of folds
            fold_id (int): which fold to use as validation set
            param (Parameters): the parameter to use

        Returns:
            torch.Tensor: validation loss
        """
        loss_fun = mnist_nn.error.get_loss_func()

        training_fold = list(range(k_fold))
        training_fold.remove(fold_id)

        training_data = bootstrap.get_fold(training_fold)
        training_loader = torch.utils.data.DataLoader(
            training_data, batch_size=N_BATCH)
        net = train_model(device, training_loader, param, loss_fun)

        validation_data = bootstrap.get_fold([fold_id])
        validation_data = torch.utils.data.DataLoader(
            validation_data, batch_size=N_BATCH)
        loss = torch.tensor(0.0).to(device)
        with torch.no_grad():
            for data in validation_data:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss += loss_fun(outputs, labels)

        return loss.cpu()
