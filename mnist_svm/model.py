"""The model"""

from cuml import svm


# Model to fit
MODEL = svm.SVC


def get_model(tuning):
    """Model to fit

    Args:
        tuning (list): contains the penalty and kernel parameters

    Returns:
        Object which can fit() and predict()
    """
    return MODEL(C=tuning[0],
                 gamma=tuning[1],
                 multiclass_strategy="ovr")
