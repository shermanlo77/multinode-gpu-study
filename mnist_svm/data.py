"""For important and standarising the MNIST dataset"""

import numpy as np

import mnist

# Data is normalised to be centred at 0 with std 1
CENTRE = 127.5
SCALE = 73.61

def get_data():
    """Get the MNIST dataset

    Get the MNIST dataset, normalised to be centred at 0 with std 1.

    Returns:
        np.ndarray: training images, dim n_training_image, n_pixel
        np.ndarray: training responses, dim n_training_image, contains class of
            each image as ints
        np.ndarray: test images with dim n_test_image, n_pixel
        np.ndarray: test responses with dim n_test_image, contains class of each
            image as ints
    """
    training_data, test_data = mnist.get_data()

    training_image = []
    test_image = []

    training_response = []
    test_response = []

    for data in training_data:
        response = data[1]
        training_image.append(np.asarray(data[0].getdata()))
        training_response.append(response)

    for data in test_data:
        response = data[1]
        test_image.append(np.asarray(data[0].getdata()))
        test_response.append(response)

    training_image = np.asarray(training_image, dtype=np.float64)
    test_image = np.asarray(test_image, dtype=np.float64)

    training_response = np.asarray(training_response, dtype=np.int32)
    test_response = np.asarray(test_response, dtype=np.int32)

    training_image -= CENTRE
    test_image -= CENTRE

    training_image /= SCALE
    test_image /= SCALE

    return (training_image, training_response, test_image, test_response)
