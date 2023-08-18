"""Provides the prediction error rate"""

import cupy

def get_error(model, image, response):
    """Get error rate given a model

    Use the model to predict the class for each image. The error rate is the
    proportion of predictions which are incorrect

    Args:
        model: object with the method predict()
        image (np.ndarray): dim n_image, n_pixel
        response (np.ndarray): dim n_image, contains class of each image as
            ints

    Returns:
        float: error rate
    """
    xp = cupy.get_array_module(image, response)
    predict = model.predict(image)
    is_equal = xp.equal(predict, response)
    return xp.mean(xp.logical_not(is_equal))
