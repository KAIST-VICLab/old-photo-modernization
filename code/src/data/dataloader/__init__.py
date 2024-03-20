import importlib
from collections import OrderedDict
import typing
from .base import BaseDataLoader


def find_dataloader_class(
    dataloader_file: str, dataloader_class: str
) -> typing.Callable:
    dataloader_module = "src.data.dataloader." + dataloader_file
    dataloaderlib = importlib.import_module(dataloader_module)

    dataloader = None
    for name, cls in dataloaderlib.__dict__.items():
        if name == dataloader_class:
            dataloader = cls

    if dataloader is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataLoader with"
            " class name that matches %s in lowercase"
            % (dataloader_module, dataloader_class)
        )

    return dataloader


def build_dataloader(
    dataloader_config: OrderedDict,
    dataset_config: OrderedDict,
    seed: int,
    world_size: int = 1,
    global_rank: int = -1,
) -> BaseDataLoader:
    dataloader_file = dataloader_config["file"]
    dataloader_class = dataloader_config["class"]
    dataloader_cls = find_dataloader_class(dataloader_file, dataloader_class)
    dataloader = dataloader_cls(
        dataloader_config, dataset_config, seed, world_size, global_rank
    )
    dataloader = dataloader.load_data()
    return dataloader
