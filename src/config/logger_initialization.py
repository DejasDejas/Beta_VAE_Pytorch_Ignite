# -*- coding: utf-8 -*-
"""
This module contains the logger configuration for the application.
"""
import logging


def setup_custom_logger(name: str) -> logging.Logger:
    """
    Set up a custom logger.

    Args:
        name (str): name of the logger

    Returns:
        Logger object: logger initialized with default configuration and name parameter

    Raises:
        ValueError: if name is not a string

    Notes:
        This is a note

    Examples:
        >>> from src.config.logger_initialization import setup_custom_logger
        >>> log = setup_custom_logger(__name__)
        >>> log.info("test")
    """
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    _log = setup_custom_logger(__name__)
    _log.info("test")
