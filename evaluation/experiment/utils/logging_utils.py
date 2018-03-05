import logging as core_logging
import sys

def setup(config):
    """setup a logger"""
    logger = core_logging.getLogger('experiment')
    formatter = core_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_stdout = core_logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(config['logger']['level'])
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)

    if 'path' in config['logger']:
        handler_file = core_logging.FileHandler(config['logger']['path'])
        handler_file.setLevel(config['logger']['level'])
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    logger.setLevel(config['logger']['level'])

    return logger
