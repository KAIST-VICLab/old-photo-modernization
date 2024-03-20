"""
Ref:
- https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts # noqa: E501
"""

from collections import OrderedDict

import yaml
import os


def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    """
    Load yaml file in ordered manner (wrapper for all python versions)

    :param stream: file to load
    :type stream: file
    :param Loader: loader class to load yaml file
    :type Loader: class
    :param object_pairs_hook: hook to load the objects
    :type object_pairs_hook: class
    :return: yaml file
    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(stream, OrderedLoader)


def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    """
    Ordered dumper wrapper comptabile with all python versions
    """

    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items()
        )

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def parse_config(config_path: os.path):
    with open(config_path, mode="r") as f:
        configs = ordered_load(
            f, Loader=yaml.SafeLoader
        )  # Loader can be changed depending on the pytorch version

    return configs


def dump_config(configs: OrderedDict, yaml_path: os.path):
    with open(yaml_path, mode="w") as f:
        ordered_dump(configs, f, Dumper=yaml.SafeDumper)
