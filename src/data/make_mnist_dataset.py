# -*- coding: utf-8 -*-
"""
This module contains the code to create the MNIST dataset.
"""
import os

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from src.config.logger_initialization import setup_custom_logger
from src.config.paths_configuration import raw_path

logger = setup_custom_logger(__name__)


def load_mnist_data(batch_size_train: int = 64, num_workers: int = 0,
                    transform: object = None) -> (data.DataLoader, data.DataLoader):
    """
    Load MNIST data and return a test and train DataLoader.
    Args:
        batch_size_train (int): batch size for training
        num_workers (int): number of workers for data loading
        transform (callable, optional): Optional transform to be applied on a sample.

    Returns:
        (DataLoader, DataLoader): train and test DataLoader
    """
    logger.info("Load MNIST data")
    # create data folder:
    os.makedirs(raw_path, exist_ok=True)

    # define transforms:
    if transform is None:
        transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomRotation(5),
                                        transforms.Normalize(
                                            (0.1307,), (0.3081,))])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(
                                                 (0.1307,), (0.3081,))])

    # load data:
    train_data = datasets.MNIST(root=raw_path,
                                train=True,
                                download=True,
                                transform=transform)
    test_data = datasets.MNIST(root=raw_path,
                               train=False,
                               download=True,
                               transform=test_transform)

    # create data loaders:
    train_loader = data.DataLoader(train_data,
                                   batch_size=batch_size_train,
                                   shuffle=True,
                                   num_workers=num_workers)

    test_loader = data.DataLoader(test_data,
                                  batch_size=1000,
                                  shuffle=True,
                                  num_workers=num_workers)

    logger.info("Load MNIST dataset from torchvision and save to raw folder.")
    logger.info(f"Train dataset size: {train_loader.dataset.__len__():,}")
    logger.info(f"Test dataset size: {test_loader.dataset.__len__():,}")
    logger.info(f"Images shape: {train_loader.dataset[0][0].shape}")

    return train_loader, test_loader


if __name__ == "__main__":
    train, test = load_mnist_data()
    print(f"Train batch number: {train.__len__()}")
    print(f"Test batch number: {test.__len__()}")
    print(f"Images size: {train.dataset[0][0].shape}")
