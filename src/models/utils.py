# pylint: disable=import-error
"""Utils for model module."""
from torch import nn


def kaiming_init(module):
    """
    Kaiming initialization.
    Args:
        module (nn.Module): Module to be initialized.

    Returns:
        Module with Kaiming initialization.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight.data, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0)


def normal_init(module, mean, std):
    """
    Normal initialization.
    Args:
        module (nn.Module): Module to be initialized.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        Module with normal initialization.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        module.weight.data.normal_(mean, std)
        if module.bias.data is not None:
            module.bias.data.zero_()
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.weight.data.fill_(1)
        if module.bias.data is not None:
            module.bias.data.zero_()
