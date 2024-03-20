import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)


def diagnose_network(net: nn.Module, name: str = "network"):
    """
    Calculate and print the mean of average absolute of gradients
    :param net: the network
    :param name: the name of the network
    :return: None
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad))
            count += 1
    mean = mean / count if count > 0 else 0.0
    log.info(f"DiagnoseNetwork - name: {name}, mean: {mean:.4f}")


def print_numpy(numpy_array: np.array, val: bool = True, shape: bool = False):
    if shape:
        log.info(f"shape: {numpy_array.shape}")
    if val:
        log.info(
            f"mean = {np.mean(numpy_array):.3f}, min = {np.min(numpy_array):.3f}, "
            f"max = {np.max(numpy_array):.3f}, median = {np.median(numpy_array):.3f}, "
            f"std = {np.std(numpy_array):.3f}"
        )


def print_tensor(tensor: torch.Tensor, val: bool = True, shape: bool = False):
    if shape:
        log.info(f"shape: {tensor.size()}")
    if val:
        log.info(
            f"mean = {torch.mean(tensor):.3f}, min = {torch.min(tensor):.3f}, "
            f"max = {torch.max(tensor):.3f}, median = {torch.median(tensor):.3f}, "
            f"std = {torch.std(tensor):.3f}"
        )


def plot_grad_flow(net: nn.Module):
    named_parameters = net.named_parameters()

    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (
            p is not None
            and (p.requires_grad)
            and ("bias" not in n)
            and p.grad is not None
        ):
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()
