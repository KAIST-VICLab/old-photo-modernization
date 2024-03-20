import importlib
import typing
from collections import OrderedDict
from .base import BaseEngine


def find_engine_class(engine_filename: str, engine_class: str) -> typing.Callable:
    engine_file_module = "src.engine.{}".format(engine_filename)
    enginelib = importlib.import_module(engine_file_module)
    engine = None
    for name, cls in enginelib.__dict__.items():
        if name == engine_class:
            engine = cls

    if engine is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseEngine with a class name "
            "that matches %s in lowercase" % (engine_filename, engine_class)
        )

    return engine


def build_engine(global_config: OrderedDict) -> BaseEngine:
    engine_config = global_config["engine"]
    engine_cls = find_engine_class(engine_config["file"], engine_config["class"])
    engine_instance = engine_cls(global_config)
    return engine_instance
