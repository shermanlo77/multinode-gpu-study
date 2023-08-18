"""Functions for evaluating the training process"""

import os

from cupy import cuda
import matplotlib.pyplot as plt

import mnist_svm.error
import mnist_svm.model

def plot_validation_error(figure_path, log_penalty_parameters,
                          log_kernel_parameters, validation_error):
    """Plot validation error as boxplots

    Args:
        figure_path (str): where to save the figures
        log_penalty_parameters (np.ndarray): vector of log_10 penalty parameters
            to do grid search on
        log_kernel_parameters (np.ndarray): vector of log_10 kernel parameters
            to do grid search on
        validation_error (np.ndarray): dim for each kernel parameter, for each
            penalty parameter
    """
    # plot validation error
    for i, log_kernel in enumerate(log_kernel_parameters):
        plt.figure()
        plt.title(f"log gamma = {log_kernel}")
        plt.boxplot(validation_error[:, i, :],
                    positions=log_penalty_parameters,
                    widths=0.1,
                    manage_ticks=False)
        plt.ylabel("validation error")
        plt.xlabel("log penalty parameter")
        plt.savefig(os.path.join(figure_path, f"validation_error_{i}.pdf"),
                    bbox_inches="tight")
        plt.close()


def report_test_error(gpu_id, best_param, training_image, training_response,
                      test_image, test_response):
    """Train the model on the entire training set and report test error

    Args:
        gpu_id (int): which GPU to use
        best_param (list): length 2, penalty parameter and kernel parameter to
            use
        training_image (np.ndarray): dim n_training_image, n_pixel
        training_response (np.ndarray): dim n_training_image, contains class of
            each image as ints
        test_image (np.ndarray): dim n_test_image, n_pixel
        test_response (np.ndarray): dim n_test_image, contains class of
            each image as ints
    """
    with cuda.Device(gpu_id):
        model = mnist_svm.model.get_model(best_param)
        model.fit(training_image, training_response)

        # report the best error
        error = (
            mnist_svm.error.get_error(model, test_image, test_response) * 100)
        print(f"Best penalty parameter: {best_param[0]}")
        print(f"Best kernel parameter: {best_param[1]}")
        print(f"Test error: {error}%")
