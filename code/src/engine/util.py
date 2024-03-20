import random
import numpy as np
import torch

from torch import distributed
from collections import OrderedDict


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_cudnn_benchmark(bool_value):
    # Knowledge-Optimization: only set to true if the size of image not varied
    # if size of the input can vary, then this can create a bottleneck when enabled
    torch.backends.cudnn.benchmark = bool_value
    # if the size of image vary, then it's better not to use this


def set_cudnn_deterministic(bool_value):
    # same with torch.use_deterministic_algorithms(bool_value)
    torch.backends.cudnn.deterministic = bool_value


def init_device(gpu_ids):
    if gpu_ids is not None:
        assert len(gpu_ids) <= torch.cuda.device_count()
        device = torch.device("cuda:{}".format(gpu_ids[0]) if gpu_ids else "cpu")
    else:
        device = torch.device("cpu")
    return device


# Code from DINO: https://github.com/facebookresearch/dino/blob/main/utils.py#L467
# Code from iColorIT: https://github.com/pmh9960/iColoriT/blob/main/utils.py


def is_dist_avail_and_initialized():
    if not distributed.is_available():
        return False
    if not distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    :param is_master: whether the process is in the master process
    :return:
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(distributed_params: OrderedDict):
    # launched with torch.distributed.launch
    torch.cuda.set_device(distributed_params["gpu"])
    distributed_params["dist_backend"] = "nccl"

    print(
        "| distributed init (rank {}): {} | gpu: {}".format(
            distributed_params["rank"],
            distributed_params["dist_url"],
            distributed_params["gpu"],
        ),
        flush=True,
    )
    print(distributed_params)
    distributed.init_process_group(
        backend=distributed_params["dist_backend"],
        init_method=distributed_params["dist_url"],
        world_size=distributed_params["world_size"],
        rank=distributed_params["rank"],
    )
    distributed.barrier()
    setup_for_distributed(distributed_params["rank"] == 0)
