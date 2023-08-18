"""Bootstrapping within folds

Wrapper classes for torch.utils.data.Dataset, to be used for bagging (which
uses bootstrapping) and k-fold cross validation
"""

import numpy as np
import torch


class Bootstrap(torch.utils.data.Dataset):
    """Wrapper of Dataset, bootstrap within folds

    Use the method get_fold() to get data from fold(s)

    Attributes:
        _dataset (torch.utils.data.Dataset): the original dataset
        _index (numpy.ndarray): array of ints, corresponds to which data point
            to use for each position
        _index_in_fold (list): list of numpy.ndarray for each fold, index for
            each fold
    """

    def __init__(self, dataset, fold_boundary, rng):
        """Bootstrap within folds

        Args:
            dataset (torch.utils.data.Dataset): the dataset to bootstrap
            fold_boundary (numpy.ndarray): ascending integers, index boundary of
                each fold, of length n_fold + 1
            rng (numpy.random._generator.Generator)
        """
        super().__init__()
        self._dataset = dataset
        self._index = np.asarray([], dtype=np.int32)
        self._index_in_fold = []

        # shuffle the data
        shuffle_index = rng.permutation(len(dataset))

        # for each fold, bootstrap
        self._index_in_fold = [
            shuffle_index[
                self._random_sample_within_fold(rng, i_fold, fold_boundary)
            ]
            for i_fold in range(len(fold_boundary) - 1)
        ]

        self._index = np.ravel(self._index_in_fold)

    def get_fold(self, fold_index):
        """Extract fold(s) from this bootstrap

        Args:
            fold_index (list): list of ints, which fold(s) to use

        Returns:
            Fold
        """
        index = np.asarray([], dtype=np.int32)
        for i in fold_index:
            index = np.concatenate([index, self._index_in_fold[i]])
        return Fold(self._dataset, index)

    def _random_sample_within_fold(self, rng, i_fold, fold_boundary):
        """Generate index of random sample within a fold

        Args:
            rng (numpy.random._generator.Generator)
            i_fold (int): fold number to bootstrap on
            fold_boundary (numpy.ndarray): ascending integers, index boundary of
                each fold, of length n_fold + 1

        Returns:
            numpy.ndarray: array of random integers
        """
        index_fold = rng.integers(
            fold_boundary[i_fold],
            fold_boundary[i_fold+1],
            fold_boundary[i_fold+1] - fold_boundary[i_fold]
        )
        return index_fold

    def __getitem__(self, index):
        return self._dataset[self._index[index]]

    def __len__(self):
        return len(self._dataset)


class Fold(torch.utils.data.Dataset):
    """Wrapper of Dataset, subset of a dataset

    Attributes:
        _data (torch.utils.data.Dataset): the data to subset on
        _index (numpy.ndarray): array of ints, index to subset on
    """

    def __init__(self, data, index):
        """
        Args:
            data (torch.utils.data.Dataset): the data to subset on
            index (numpy.ndarray): array of ints, index to subset on
        """
        super().__init__()
        self._data = data
        self._index = index

    def __getitem__(self, index):
        return self._data[self._index[index]]

    def __len__(self):
        return len(self._index)
