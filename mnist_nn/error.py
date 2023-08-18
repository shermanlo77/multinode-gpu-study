"""The loss function to optimise"""

import torch


def get_loss_func():
    """Return a loss function to use"""
    return torch.nn.CrossEntropyLoss()
