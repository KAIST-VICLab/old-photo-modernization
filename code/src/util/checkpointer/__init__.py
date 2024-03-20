from .checkpointer import Checkpointer
from collections import OrderedDict


def build_checkpointer(engine_config: OrderedDict):
    return Checkpointer(engine_config["checkpointer"])
