# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------
from torch import optim

import json
import torch

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_parameter_groups(
    model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None
):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            # TODO: check
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(
    optim_config,
    network,
    get_num_layer=None,
    get_layer_scale=None,
    filter_bias_and_bn=True,
    skip_list=None,
):
    optim_name = optim_config["name"]
    weight_decay = optim_config["weight_decay"]

    optim_params = optim_config["params"]

    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(network, "no_weight_decay"):
            skip = network.no_weight_decay()
        parameters = get_parameter_groups(
            network, weight_decay, skip, get_num_layer, get_layer_scale
        )
        weight_decay = 0.0
    else:
        parameters = network.parameters()

    if "fused" in optim_name:
        assert (
            has_apex and torch.cuda.is_available()
        ), "APEX and CUDA required for fused optimizers"

    print("optimizer settings:", optim_params)

    if optim_name == "adamw":
        optimizer = optim.AdamW(parameters, **optim_params)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer


def create_combined_optimizers(
    optim_config,
    network_list: list,
    get_num_layer_list: list = None,
    get_layer_scale_list: list = None,
    filter_bias_and_bn: bool = True,
    skip_lists: list = None,
):
    optim_name = optim_config["name"]
    weight_decay = optim_config["weight_decay"]

    optim_params = optim_config["params"]

    total_parameters = []
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if get_num_layer_list is None:
            get_num_layer_list = [None] * len(network_list)
        if get_layer_scale_list is None:
            get_layer_scale_list = [None] * len(network_list)
        if skip_lists is None:
            skip_lists = [None] * len(network_list)
        for network, skip_list, get_num_layer, get_layer_scale in zip(
            network_list, skip_lists, get_num_layer_list, get_layer_scale_list
        ):
            if skip_list is not None:
                skip = skip_list
            elif hasattr(network, "no_weight_decay"):
                skip = network.no_weight_decay()
            parameters = get_parameter_groups(
                network, weight_decay, skip, get_num_layer, get_layer_scale
            )
            weight_decay = 0.0
            total_parameters.extend(parameters)
    else:
        for network in network_list:
            parameters = network.parameters()
            total_parameters.extend(parameters)

    if "fused" in optim_name:
        assert (
            has_apex and torch.cuda.is_available()
        ), "APEX and CUDA required for fused optimizers"

    print("optimizer settings:", optim_params)

    if optim_name == "adamw":
        optimizer = optim.AdamW(total_parameters, **optim_params)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer
