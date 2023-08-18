"""Get the MNIST Dataset

Run this script to download the data and save it on disk. This should be done
before running any scripts or jobs
"""

import torchvision

DOWNLOAD_PATH = "."

TRANSFORM = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(0.5, 1)
        ]
    )


def get_data(transform=None):
    """Get the MNIST dataset

    Returns:
        torchvision.datasets: training data
        torchvision.datasets: test data
    """
    training_data = torchvision.datasets.MNIST(
        DOWNLOAD_PATH, True, transform, download=True)
    test_data = torchvision.datasets.MNIST(
        DOWNLOAD_PATH, False, transform, download=True)

    return training_data, test_data


if __name__ == "__main__":
    get_data()
