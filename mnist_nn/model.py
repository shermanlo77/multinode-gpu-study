"""The model and parameters (tuning and hyper) to tune

Also provides random parameters for random search
"""

import math

import torch


class Net(torch.nn.Module):
    """Neural network
    """

    def __init__(self, n_conv_layer, kernel_size, n_hidden_layer):
        super().__init__()
        self._conv_layer = torch.nn.Conv2d(1, n_conv_layer, kernel_size)
        self._hidden_layer = torch.nn.Linear(
            n_conv_layer * (28-kernel_size+1)**2,
            n_hidden_layer
        )
        self._output_layer = torch.nn.Linear(n_hidden_layer, 10)
        self._activation = torch.nn.ReLU()

    def forward(self, x):
        x = self._conv_layer(x)
        x = torch.flatten(x, 1)
        x = self._activation(x)
        x = self._hidden_layer(x)
        x = self._activation(x)
        x = self._output_layer(x)
        return x


class Parameters:
    """Parameters to tune
    """

    def __init__(self, n_conv_layer, kernel_size, n_hidden_layer,
                 learning_rate, momentum):
        self.n_conv_layer = n_conv_layer
        self.kernel_size = kernel_size
        self.n_hidden_layer = n_hidden_layer
        self.learning_rate = learning_rate
        self.momentum = momentum

    def __str__(self):
        string = (
            f"n_conv_layer: {self.n_conv_layer}\n"
            f"kernel_size: {self.kernel_size}\n"
            f"n_hidden_layer: {self.n_hidden_layer}\n"
            f"learning_rate: {self.learning_rate}\n"
            f"momentum: {self.momentum}"
        )
        return string


def random_parameter(rng):
    """Return a random parameter

    Return a random parameter, to be used in random search
    """

    param = Parameters(
        rng.integers(1, 1000),
        rng.integers(2, 11),
        rng.integers(1, 2000),
        math.pow(10, rng.random()*4 - 5),
        math.pow(10, rng.random()*2 - 2)
    )

    return param
