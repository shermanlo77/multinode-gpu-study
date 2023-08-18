import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
from scipy import stats
import torch

ERRN_FILE = os.path.join("mnist_nn", "errn.png")


def eval_ensemble(comm, device, results, models, training_data, test_data,
                  figure_path):
    """Evaluate the ensemble of models on the training and test set

    For each model in the ensemble, evaluate the error rate to get the training
    and test error. Also works out the error rate when using majority vote from
    all models in the ensemble.

    For a sample of data in the test data, plot the distribution of certainity
    for each label as a box plot.

    Args:
        comm (mpi4py.MPI.Intracomm): MPI.COMM_WORLD()
        device (torch.device): device to run on
        results (pandas.core.frame.DataFrame): if rank 0, dataframe of specs for
            each model and modified to include the training and test error
        models (dict): dictionary of models for each bootstrap sample
            key: int to identify the bootstrap sample
            value: (Net) best model found for this bootstrap sample
        training_data (torchvision.datasets)
        test_data (torchvision.datasets)
        figure_path (str): where to save the results
    """
    rank = comm.Get_rank()

    # report the training error for each model
    predictions, probs = models_predict(comm, device, models, training_data)
    if rank == 0:
        training_error = get_error_rate(predictions, training_data)
        results["training_error"] = training_error

    # report the test error and get predictions and probabilites
    predictions, probs = models_predict(comm, device, models, test_data)
    if rank == 0:
        test_error = get_error_rate(predictions, test_data)
        results["test_error"] = test_error

    if rank != 0:
        return

    results.to_csv(os.path.join(figure_path, "results.csv"))

    # report the ensemble test error, using majority vote
    predictions = stats.mode(predictions, 0).mode.flatten()
    error_rate = [
        predictions[i_data] != data[1]
        for i_data, data in enumerate(test_data)
    ]
    error_rate = np.mean(error_rate)
    print(f"ensemble test error: {error_rate}")

    # box plot certainity and their digit
    for i in range(100):
        plot_uncertainity(probs[:, :, i], f"datapoint {i}",
                          os.path.join(figure_path, f"prob_{i}.pdf"))
        plot_datapoint(test_data[i][0][0],
                       os.path.join(figure_path, f"test_{i}.png"))


def eval_errn(comm, device, models, transform, figure_path):
    """Evaluate the ensemble of models on erroneous data

    Evaluate the model using one erroneous data. Plots the uncertainity of each
    class prediction and the erroneous data

    Args:
        comm (mpi4py.MPI.Intracomm): MPI.COMM_WORLD()
        device (torch.device): device to run on
        models (dict): dictionary of models for each bootstrap sample
            key: int to identify the bootstrap sample
            value: (Net) best model found for this bootstrap sample
        transform (torchvision.transforms.Compose): to normalise the image
        figure_path (str): where to save the results
    """
    rank = comm.Get_rank()

    # get erroneous data
    inputs = PIL.Image.open(ERRN_FILE)
    inputs = PIL.ImageOps.grayscale(inputs)
    inputs = transform(inputs)

    # plot box plot certainity for the erroneous data
    _, probs = models_predict(comm, device, models, [[inputs, None]])

    if rank != 0:
        return

    probs = probs[:, :, 0]

    plot_uncertainity(probs, "erroneous data",
                      os.path.join(figure_path, "prob_errn.pdf"))
    plot_datapoint(inputs[0].cpu(),
                   os.path.join(figure_path, "test_errn.png"))


def models_predict(comm, device, models, test_data):
    """Get predictions for each ensemble of models from all MPI ranks

    Get the prediction (digit) and certainity (for each digit) for each model
    from all MPI ranks

    Return value only available for rank 0, else returns (None, None)

    Args:
        comm (mpi4py.MPI.Intracomm): MPI.COMM_WORLD()
        device (torch.device): device to run on
        models (dict): dictionary of models for each bootstrap sample
            key: (int) to identify the bootstrap sample or model
            value: (Net) best model found for this bootstrap sample
        test_data (torchvision.datasets): dataset to prediction on

    Returns:
        np.ndarray: the predicted labels, ndim for each model, for each data
            point
        np.ndarray: the certainity of each label prediction, n dim for each
            model, for each label, for each data point
    """
    predictions, probs = rank_models_predict(device, models, test_data)
    predictions, probs = gather_model_predictions(comm, predictions, probs)
    return predictions, probs


def rank_models_predict(device, models, test_data):
    """Get predictions for each ensemble of models for this MPI rank

    Get the prediction (digit) and certainity (for each digit) for each model
    in this MPI rank

    Args:
        device (torch.device): device to run on
        models (dict): dictionary of models for each bootstrap sample
            key: (int) to identify the bootstrap sample
            value: (Net) best model found for this bootstrap sample
        test_data (torchvision.datasets): dataset to prediction on

    Returns:
        dict: the predicted labels,
            key: (int) to identify the bootstrap sample or model in this rank
            value: (np.ndarray), 1d, n_dim for each data
        dict: the certainity of each label prediction
            key: (int) to identify the bootstrap sample or model in this rank
            value: (np.ndarray) 2d, n_dim or each label, for each data point
    """

    with torch.no_grad():

        predictions = {}
        probs = {}

        for net_id, net in models.items():
            net.to(device)

            predictions_i = np.empty(len(test_data))
            probs_i = np.empty((10, len(test_data)))

            for i_data, data in enumerate(test_data):
                inputs, _ = data
                inputs = inputs[None, :, :, :]
                inputs = inputs.to(device)
                output = net(inputs)

                # transform into target, ie digit
                prediction = torch.argmax(output)
                predictions_i[i_data] = prediction.cpu()

                # transform into probability
                prob = torch.nn.functional.softmax(output, 1)
                probs_i[:, i_data] = prob.cpu()

            predictions[net_id] = predictions_i
            probs[net_id] = probs_i

    return predictions, probs


def gather_model_predictions(comm, predictions, probs):
    """Gather the return value of rank_models_predict() from all ranks

    Return value only available for rank 0, else returns (None, None)

    Args:
        comm (mpi4py.MPI.Intracomm): MPI.COMM_WORLD()
        predictions (dict): the predited labels, return value of
            models_predict()
        probs (dict): the certainity of each label prediction, return value of
            models_predict()

    Returns:
        np.ndarray: the predicted labels, ndim for each model, for each data
            point
        np.ndarray: the certainity of each label prediction, n dim for each
            model, for each label, for each data point
    """

    rank = comm.Get_rank()

    gathered_predictions = comm.gather(predictions, root=0)
    gathered_probs = comm.gather(probs, root=0)

    if rank != 0:
        return None, None

    predictions_dict = {}
    probs_dict = {}

    for predictions_i in gathered_predictions:
        predictions_dict.update(predictions_i)
    for probs_i in gathered_probs:
        probs_dict.update(probs_i)

    n_model = len(predictions_dict)
    n_data = len(predictions_dict[0])

    predictions_np = np.empty((n_model, n_data))
    probs_np = np.empty((n_model, 10, n_data))

    for i_model, predictions_i in predictions_dict.items():
        predictions_np[i_model] = predictions_i

    for i_model, probs_i in probs_dict.items():
        probs_np[i_model] = probs_i


    return predictions_np, probs_np


def get_error_rate(predictions, test_data):
    """Evaluate prediction of the ensemble of models

    Evaluate the error rate for each model in an ensemble given test data

    Args:
        predictions (np.ndarray): the predicted labels, ndim for each model, for
            each data point
        test_data (torchvision.datasets): dataset to prediction on

    Returns:
        np.ndarray: the predicted labels, ndim for each model, for each data
            point
        np.ndarray: the certainity of each label prediction, n dim for each
            model, for each label, for each data point
    """
    error_rate = np.empty((len(predictions), len(test_data)))

    for i_data, data in enumerate(test_data):
        _, label = data
        label = torch.tensor(label, dtype=int).item()
        error_rate[:, i_data] = predictions[:, i_data] != label

    error_rate = np.mean(error_rate, axis=1)
    return error_rate


def plot_uncertainity(boxplot_data, title, file_name):
    """Plots uncertainity

    Plots the uncertainity of each class prediction

    Args:
        boxplot_data (np.ndarray): certainity of each class, dim for each
            prediction, for each class
        file_name (str): where to save the plot
    """
    plt.figure()
    plt.boxplot(boxplot_data, positions=range(10))
    plt.ylim([-0.05, 1.05])
    plt.ylabel("certainity of prediction")
    plt.xlabel("prediction")
    plt.title(title)
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def plot_datapoint(data_i, file_name):
    """Plot the image"""
    plt.imshow(data_i, cmap='gray')
    plt.axis('off')
    plt.savefig(file_name)
    plt.close()
