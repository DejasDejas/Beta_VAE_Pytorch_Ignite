# -*- coding: utf-8 -*-
# pylint: disable = logging-fstring-interpolation, import-error
"""
Utils for the trainer module and the runner module.
"""
import random
import numpy as np
from torch import manual_seed
from torch.cuda import manual_seed_all
from torch.cuda import is_available, device_count, get_device_name
from torch.nn import DataParallel
from src.config.logger_initialization import setup_custom_logger

logger = setup_custom_logger(__name__)


def str2bool(_string: str) -> bool:
    """
    Convert a string to a boolean.
    Args:
        _string (str): The string to convert.

    Returns:
        bool: The boolean.

    Raises:
        ValueError: If the string is not 'True' or 'False'.

    Notes: codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with
    -argparse
    """
    if _string.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    if _string.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    raise ValueError('Not a valid boolean string')


def gpu_config(model):
    """
    Configure the GPU with all the available GPUs.
    If there is only one GPU, the model is not parallelized and is trained on the GPU.
    If there are more than one GPU, the model is parallelized and trained on the GPUs.
    Args:
        model (nn.Module): The model to configure.

    Returns:
        model (nn.Module): The model with the GPU configured.
        device (torch.device): The device.

    """
    if is_available():
        device = 'cuda'
        gpu_count = device_count()
        if gpu_count > 1:
            logger.info(f"use {gpu_count} gpu who named:")
            for i in range(gpu_count):
                logger.info(get_device_name(i))
            model = DataParallel(model, device_ids=list(range(gpu_count)))
        else:
            logger.info(f"use 1 gpu who named: {get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("no gpu available !")
    model.to(device)
    return model, device


def randomness_seed(seed):
    """
    Set the randomness seed.
    If manual seed is not specified, choose a random one and communicate it to the user
    Args:
        seed (int): The seed to set.
    """
    seed = seed or random.randint(1, 10000)
    manual_seed(seed)
    manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Using manual seed: {seed}")
