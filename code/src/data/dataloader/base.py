import importlib

from abc import ABC, abstractmethod
from collections import OrderedDict
import typing
import logging

log = logging.getLogger(__name__)


def find_dataset_class(dataset_file: str, dataset_class: str) -> typing.Callable:
    dataset_filename = "src.data.dataset." + dataset_file
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    for name, cls in datasetlib.__dict__.items():
        if name == dataset_class:
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with"
            " class name that matches %s in lowercase"
            % (dataset_filename, dataset_class)
        )

    return dataset


class BaseDataLoader(ABC):
    def __init__(
        self,
        dataloader_config: OrderedDict,
        dataset_config: OrderedDict,
        seed: int,
        world_size: int = 1,
        global_rank: int = -1,
    ):
        self.dataloader_config = dataloader_config
        self.dataset_config = dataset_config
        self.seed = seed
        # distributed params
        self.world_size = world_size
        self.global_rank = global_rank

        dataset_class = find_dataset_class(
            self.dataset_config["file"], self.dataset_config["class"]
        )
        self.dataset = dataset_class(self.dataset_config)

        log.info("dataset {} was created".format(type(self.dataset).__name__))

        dataloader_params = self.dataloader_config["params"]
        self.dataloader = self.build_dataloader(dataloader_params)

    @abstractmethod
    def build_dataloader(self, dataloader_params: OrderedDict):
        return None

    def set_epoch(self, epoch: int):
        self.dataset.current_epoch = epoch

    def load_data(self):
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        for _, data in enumerate(self.dataloader):
            yield data
