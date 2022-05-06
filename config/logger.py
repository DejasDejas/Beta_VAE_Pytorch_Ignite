"""Module logging to define the loggers configuration."""
import logging


def setup_custom_logger(name):
    """
    Set up a custom logger. \n
    :param name: (str) logger name to create. \n
    :return: Logger object initialized with default configuration and name param. \n
    Use example: \n
        import config.log as log \n
        logger = log.setup_custom_logger(__name__) \n
        logger.info('This is a test')
    """
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger
