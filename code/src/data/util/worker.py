import numpy as np
import random
import torch


def seed_worker(worker_id):
    # initial_seed is different depending on the worker id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # important since to remove data duplication issue
    random.seed(worker_seed)
