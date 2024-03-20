from torch.optim import lr_scheduler
from torch import nn

import numpy as np


def get_scheduler(optimizer, scheduler_config):
    if scheduler_config is None:
        scheduler = nn.Identity()
    else:
        lr_policy = scheduler_config["policy"]
        scheduler_params = scheduler_config["params"]
        if lr_policy == "linear":
            start_epoch = scheduler_params["start_epoch"]
            n_epochs = scheduler_params["n_epochs"]
            n_epochs_decay = scheduler_params["n_epochs_decay"]

            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + start_epoch - n_epochs) / float(
                    n_epochs_decay + 1
                )
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == "step":
            lr_decay_iters = scheduler_params["lr_decay_iters"]
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=lr_decay_iters, gamma=0.1
            )
        elif lr_policy == "plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
            )
        elif lr_policy == "cosine":
            n_epochs = scheduler_params["n_epochs"]

            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=0
            )
        else:
            raise NotImplementedError
    return scheduler


# from DINO
def cosine_scheduler_values(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def step_scheduler_values(
    base_value,
    epochs,
    step_size,
    gamma,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    all_schedule = warmup_schedule
    for epoch in range(warmup_epochs, epochs):
        scale = gamma ** (epoch // step_size)
        schedule = np.zeros(niter_per_ep) + (base_value * scale)
        all_schedule = np.concatenate((all_schedule, schedule))
    assert len(all_schedule) == epochs * niter_per_ep
    return all_schedule
