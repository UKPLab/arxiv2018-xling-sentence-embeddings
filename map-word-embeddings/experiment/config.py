from os import path

import yaml

from experiment.utils.data import merge_dicts


def load_config(config_path, default_config_path=None):
    """Loads and reads a yaml configuration file which extends the default_config of this project

    :param config_path: path of the configuration file to load
    :type config_path: str
    :param default_config_path: the path of the default configuration file. If not set, the function will use the
                                default_config.yaml file in the root directory.
    :type default_config_path: str
    :return: the configuration dictionary
    :rtype: dict
    """
    if default_config_path is None:
        default_config_path = path.join(path.dirname(__file__), '..', 'default_config.yaml')
    with open(config_path, 'r') as user_config_file, open(default_config_path, 'r') as default_config_file:
        return read_config(user_config_file.read(), default_config_file.read())


def read_config(config_str, default_config_str=None):
    """Reads a yaml configuration string which extends the default_config string, if there is one

    :param config_str: the configuration to read
    :type config_str: str
    :param default_config_str: the default configuration. If set to None, the default configuration will be empty
    :type default_config_path: str
    :return: the configuration dictionary
    :rtype: dict
    """
    user_config = yaml.load(config_str)
    default_config_str = yaml.load(default_config_str) if default_config_str else dict()
    return merge_dicts(default_config_str, user_config)
