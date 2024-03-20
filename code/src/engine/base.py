from abc import ABC, abstractmethod
from collections import OrderedDict


class BaseEngine(ABC):
    def __init__(self, global_config: OrderedDict):
        self.global_config = global_config
        self.engine_config = global_config["engine"]
        self.datasets_config = global_config["datasets"]
        self.model_config = global_config["model"]

    @abstractmethod
    def run(self):
        pass
