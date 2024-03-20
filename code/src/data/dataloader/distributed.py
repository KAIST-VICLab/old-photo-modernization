from .base import BaseDataLoader

from collections import OrderedDict
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from ..util.worker import seed_worker


class DistributedDataLoader(BaseDataLoader):
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
        world_size = self.world_size
        global_rank = self.global_rank

        sampler = DistributedSampler(
            self.dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=dataloader_params["use_shuffle"],
        )

        dataloader = DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=dataloader_params["batch_size"],
            num_workers=dataloader_params["n_workers"],
            drop_last=dataloader_params["drop_last"],
            pin_memory=dataloader_params["pin_memory"],
            worker_init_fn=seed_worker,
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
