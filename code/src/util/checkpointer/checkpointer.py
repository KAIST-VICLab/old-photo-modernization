import os

import torch
from collections import OrderedDict
import logging

log = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, config: OrderedDict):
        self.config = config

        self.root_dir = self.config["root_dir"]
        self.epoch_freq = self.config["epoch_freq"]
        self.iter_freq = self.config["iter_freq"]
        log.info("Creating checkpointer")

    def time2checkpoint(
        self, current_epoch: int, current_iter: int, check_by: str = "iter"
    ) -> bool:
        if check_by == "both":
            if (
                current_epoch % self.epoch_freq == 0
                or current_iter % self.iter_freq == 0
            ):
                return True
            else:
                return False
        elif check_by == "epoch":
            if current_epoch % self.epoch_freq == 0:
                return True
            else:
                return False
        elif check_by == "iter":
            if current_iter % self.iter_freq == 0:
                return True
            else:
                return False
        else:
            raise NotImplementedError

    def save(
        self,
        step: str,
        model_state: dict,
        model_internal_state: dict,
        engine_state: dict,
        summary: dict,
    ):
        checkpoint = {
            "model": model_state,  # model state
            "model_internal_state": model_internal_state,  # other important internal states of the model
            "engine": engine_state,  # for run
            "summary": summary,  # for utility information
        }
        checkpoint_path = os.path.join(self.root_dir, f"checkpoint_{step}.pt")
        torch.save(checkpoint, checkpoint_path)

    def load(self, step: str) -> dict:
        checkpoint_path = os.path.join(self.root_dir, f"checkpoint_{step}.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint

    def direct_load(self, checkpoint_path):
        checkpoint = torch.load(
            os.path.join(self.root_dir, checkpoint_path), map_location="cpu"
        )
        return checkpoint
