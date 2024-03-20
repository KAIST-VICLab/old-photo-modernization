from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from collections import OrderedDict


class BaseDataset(Dataset, ABC):
    def __init__(self, dataset_config: OrderedDict):
        self.dataset_config = dataset_config
        self.root_dir = dataset_config["params"]["root_dir"]
        self.current_epoch = 0

    @abstractmethod
    def __getitem__(self, index) -> dict:
        return {}

    @abstractmethod
    def __len__(self) -> int:
        return 0
