from .base import BaseLogger
from collections import OrderedDict
import logging
import os

# Source: https://gist.github.com/scarecrow1123/967a97f553697743ae4ec7af36690da6
# fail due to the nature of torch.distributed.launch
# can be used later if we spawn it manually


class RankFilter(logging.Filter):
    def __init__(self, rank: int = -1):
        super().__init__()
        self._rank = str(rank)

    def filter(self, record):
        record.rank = self._rank
        return True


def init_logger(logger_config, verbose, rank: int = 0):
    logging_root_dir = logger_config["root_dir"]

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(logging_root_dir, "run.log"))
    stream_handler = logging.StreamHandler()

    # TODO: check verbosity, it's better to add this variable in the engine
    if verbose:
        file_handler.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.DEBUG)
    else:
        file_handler.setLevel(logging.INFO)
        stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "Rank:%(rank)s::%(asctime)s::%(name)s::%(levelname)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    file_handler.addFilter(RankFilter(rank=rank))
    stream_handler.addFilter(RankFilter(rank=rank))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def build_logger(engine_config: OrderedDict) -> BaseLogger:
    logger = BaseLogger(engine_config["logger"])
    return logger
