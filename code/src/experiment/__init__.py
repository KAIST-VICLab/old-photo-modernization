"""
Experiment related stuff which acts as a bridge between main and
    other stuff related to experiments:
"""
import logging
import os
import torch
import sys

from collections import OrderedDict
from pathlib import Path

from src.config.parser import dump_config
from src.util.checkpointer.code import dump_code
from src.util.logger import init_logger
import src
from datetime import datetime

log = logging.getLogger(__name__)


def init_experiment(global_config: OrderedDict):
    if (
        "RANK" in os.environ and "WORLD_SIZE" in os.environ
    ) or "SLURM_PROCID" in os.environ:
        init_distributed_experiment(global_config)
    else:
        init_local_experiment(global_config)


def init_local_experiment(global_config: OrderedDict):
    """
    initialize experiment by creating various directory for logger,
    visualizer, checkpointer, and evaluator
    :param: config: loaded configuration
    :type: config: OrderedDict
    :rtype: None
    """
    distributed_params = global_config["engine"].get("distributed_params", {})
    # launched naively with `python main.py` -> distributed engine will also support a single GPU
    if torch.cuda.is_available():
        print("Will run the code on one GPU.")
        distributed_params["rank"] = global_config["engine"]["gpu_id"]
        distributed_params["gpu"] = global_config["engine"]["gpu_id"]
        distributed_params["world_size"] = 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        distributed_params["is_distributed"] = False
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    experiment_root_dir = global_config["experiment"]["root_dir"]
    # Logger is initialized after engine building
    if global_config["experiment"]["add_timestamp"]:
        print("Adding timestamp to the experiment directory")
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M")
        parent_path, old_dir_name = os.path.split(experiment_root_dir)
        new_dir_name = "_".join([current_datetime, old_dir_name])
        experiment_root_dir = os.path.join(parent_path, new_dir_name)
    else:
        print("Using the same root directory for the experiment")

    checkpoint_dir = os.path.join(experiment_root_dir, "checkpoint")
    log_dir = os.path.join(experiment_root_dir, "log")
    visualization_dir = os.path.join(experiment_root_dir, "visualization")
    evaluation_dir = os.path.join(experiment_root_dir, "evaluation")

    global_config["engine"]["checkpointer"]["root_dir"] = checkpoint_dir
    global_config["engine"]["logger"]["root_dir"] = log_dir
    global_config["engine"]["visualizer"]["root_dir"] = visualization_dir
    global_config["engine"]["evaluator"]["root_dir"] = evaluation_dir

    if global_config["engine"]["image_writer"]["root_dir"] is None:
        image_writer_dir = "outputs"
    else:
        image_writer_dir = global_config["engine"]["image_writer"]["root_dir"]
    image_writer_dir = os.path.join(experiment_root_dir, image_writer_dir)
    global_config["engine"]["image_writer"]["root_dir"] = image_writer_dir

    yaml_filepath = os.path.join(experiment_root_dir, "config.yaml")

    zipped_code_path = os.path.join(experiment_root_dir, "code")
    src_dir_path = os.path.dirname(src.__file__)

    print("Creating directory for utility module in local machine")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(visualization_dir).mkdir(parents=True, exist_ok=True)
    Path(evaluation_dir).mkdir(parents=True, exist_ok=True)
    Path(image_writer_dir).mkdir(parents=True, exist_ok=True)
    dump_config(global_config, yaml_filepath)
    dump_code(in_dir_path=src_dir_path, out_file_path=zipped_code_path)

    init_logger(global_config["engine"]["logger"], global_config["engine"]["verbose"])


def init_distributed_experiment(global_config: OrderedDict):
    """
    initialize experiment by creating various directory for logger,
    visualizer, checkpointer, and evaluator
    :param: config: loaded configuration
    :type: config: OrderedDict
    :rtype: None
    """
    # Distributed params
    distributed_params = global_config["engine"]["distributed_params"]
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        distributed_params["rank"] = int(os.environ["RANK"])
        distributed_params["world_size"] = int(os.environ["WORLD_SIZE"])
        distributed_params["gpu"] = int(os.environ["LOCAL_RANK"])
        distributed_params["is_distributed"] = True
    # launched with submitit on a slurm cluster
    elif "SLURM_PROCID" in os.environ:
        distributed_params["rank"] = int(os.environ["SLURM_PROCID"])
        distributed_params["gpu"] = (
            distributed_params["rank"] % torch.cuda.device_count()
        )
        distributed_params["is_distributed"] = True

    experiment_root_dir = global_config["experiment"]["root_dir"]
    # Logger is initialized after engine building
    if global_config["experiment"]["add_timestamp"]:
        print("Adding timestamp to the experiment directory")
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M")
        parent_path, old_dir_name = os.path.split(experiment_root_dir)
        new_dir_name = "_".join([current_datetime, old_dir_name])
        experiment_root_dir = os.path.join(parent_path, new_dir_name)
        torch.cuda.synchronize()
    else:
        print("Using the same root directory for the experiment")

    rank = distributed_params["rank"]
    checkpoint_dir = os.path.join(experiment_root_dir, "checkpoint")
    log_dir = os.path.join(experiment_root_dir, f"log_rank{rank}")
    visualization_dir = os.path.join(experiment_root_dir, f"visualization_rank{rank}")
    evaluation_dir = os.path.join(experiment_root_dir, f"evaluation_rank{rank}")

    global_config["engine"]["checkpointer"]["root_dir"] = checkpoint_dir
    global_config["engine"]["logger"]["root_dir"] = log_dir
    global_config["engine"]["visualizer"]["root_dir"] = visualization_dir
    global_config["engine"]["evaluator"]["root_dir"] = evaluation_dir

    # Only support single machine
    if global_config["engine"]["image_writer"]["root_dir"] is None:
        image_writer_dir = f"outputs_{rank}"
    else:
        image_writer_dir = global_config["engine"]["image_writer"]["root_dir"]
    image_writer_dir = os.path.join(experiment_root_dir, image_writer_dir)
    global_config["engine"]["image_writer"]["root_dir"] = image_writer_dir

    yaml_filepath = os.path.join(experiment_root_dir, "config.yaml")

    zipped_code_path = os.path.join(experiment_root_dir, "code")
    src_dir_path = os.path.dirname(src.__file__)

    print("Creating directory for utility module in local machine")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(visualization_dir).mkdir(parents=True, exist_ok=True)
    Path(evaluation_dir).mkdir(parents=True, exist_ok=True)
    Path(image_writer_dir).mkdir(parents=True, exist_ok=True)

    if rank == 0:
        dump_config(global_config, yaml_filepath)
        dump_code(in_dir_path=src_dir_path, out_file_path=zipped_code_path)

    init_logger(
        global_config["engine"]["logger"], global_config["engine"]["verbose"], rank=rank
    )
