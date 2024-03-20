from .base import BaseDataLoader
from ..util.worker import seed_worker
from torch.utils.data import DataLoader, RandomSampler
from collections import OrderedDict

import torch

# https://stackoverflow.com/questions/74738608/the-relationship-between-dataloader-sampler-and-generator-in-pytorch
# https://stackoverflow.com/questions/67196075/pytorch-dataloader-uses-identical-random-transformation-across-each-epoch


class SingleDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataloader_config: OrderedDict,
        dataset_config: OrderedDict,
        seed: int,
        world_size: int = 1,
        global_rank: int = -1,
    ):
        super().__init__(
            dataloader_config, dataset_config, seed, world_size, global_rank
        )

    def build_dataloader(self, dataloader_params: OrderedDict):
        # Knowledge-Optimization: use pin memory if you want faster dataloading,
        # but it must be paired with non_blocking in .to() function

        if dataloader_params["use_sampler"]:
            sampler = RandomSampler(self.dataset)

            dataloader = DataLoader(
                self.dataset,
                sampler=sampler,
                batch_size=dataloader_params["batch_size"],
                num_workers=dataloader_params["n_workers"],
                drop_last=dataloader_params["drop_last"],
                pin_memory=dataloader_params["pin_memory"],
                worker_init_fn=seed_worker,  # to prevent every worker loading the same data
            )
        else:
            # important to prevent dataloader bug where every worker load the same data (every epoch maybe)
            g = torch.Generator()  # need more validation of the usage
            g.manual_seed(self.seed)

            dataloader = DataLoader(
                self.dataset,
                batch_size=dataloader_params["batch_size"],
                shuffle=dataloader_params["use_shuffle"],
                num_workers=dataloader_params["n_workers"],
                drop_last=dataloader_params["drop_last"],
                pin_memory=dataloader_params["pin_memory"],
                worker_init_fn=seed_worker,  # prevent every worker to load the same data
                generator=g,
            )
        return dataloader

    def set_epoch(self, epoch: int):
        self.dataset.current_epoch = epoch

    def load_data(self):
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        for _, data in enumerate(self.dataloader):
            yield data
