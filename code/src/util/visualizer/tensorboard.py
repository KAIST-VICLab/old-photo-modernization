import torch
from torch.utils.tensorboard import SummaryWriter
from .base import BaseVisualizer
from collections import OrderedDict
import os
import numpy as np
from datetime import datetime

# from src.util.writer.image import tensor2im


class TensorboardVisualizer(BaseVisualizer):
    def __init__(self, instance_config: OrderedDict, visualizer_config: OrderedDict):
        super(TensorboardVisualizer, self).__init__(instance_config, visualizer_config)

        self.name = self.instance_config["name"]
        current_datetime = datetime.now().strftime("run%Y%m%d_%H%M")
        self.instance_root_dir = os.path.join(
            self.visualizer_config["root_dir"], self.name, current_datetime
        )

        self.writer = SummaryWriter(log_dir=self.instance_root_dir)

    def get_global_step(self, step: str) -> int:
        global_step = int(step.split("_")[-1][2:])  # total iteration
        return global_step

    def add_scalar(
        self, tag: str, scalar_value: float, step: str = None, wall_time=None
    ):
        global_step = self.get_global_step(step)
        self.writer.add_scalar(tag, scalar_value, global_step, wall_time)

    def add_scalars(
        self, tag: str, scalar_dict: dict, step: str = None, wall_time=None
    ):
        # better looking than using direct add_scalars
        for key, value in scalar_dict.items():
            self.add_scalar(tag + "/" + key, value, step, wall_time)
        # global_step = self.get_global_step(step)
        # self.writer.add_scalars(tag, scalar_dict, global_step, wall_time)

    def add_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: str,
        bins: str = "tensorflow",
        wall_time=None,
    ):
        global_step = self.get_global_step(step)
        self.writer.add_histogram(tag, values, global_step, bins, wall_time)

    def add_image(
        self,
        tag: str,
        img_tensor: np.ndarray,
        step: str = None,
        wall_time=None,
        dataformats: str = "CHW",
    ):
        global_step = self.get_global_step(step)
        self.writer.add_image(
            tag, img_tensor, global_step, wall_time, dataformats=dataformats
        )

    def add_images(  # for batch images
        self, tag: str, img_tensor: torch.Tensor, step: str = None, wall_time=None
    ):
        global_step = self.get_global_step(step)
        self.writer.add_images(tag, img_tensor, global_step, wall_time)

    def add_image_dict(
        self, tag: str, img_tensor_dict: dict, step: str = None, wall_time=None
    ):
        for key, img_tensor in img_tensor_dict.items():
            # can support batch visualization
            # image_numpy = tensor2im(img_tensor)
            # if image_numpy.ndim == 2:
            #     image_numpy = np.expand_dims(
            #         image_numpy, axis=-1
            #     )  # tensorboard only support ndim=3
            self.add_image(tag + "/" + key, img_tensor[0], step, wall_time)

    def add_video(
        self,
        tag: str,
        vid_tensor: torch.Tensor,
        fps: int = None,
        step: str = None,
        wall_time=None,
    ):
        global_step = self.get_global_step(step)
        self.writer.add_video(tag, vid_tensor, global_step, fps, wall_time)

    def add_graph(
        self,
        model: torch.nn.Module,
        input_to_model: torch.Tensor,
        verbose: bool = False,
    ):
        self.writer.add_graph(model, input_to_model, verbose)

    def add_embedding(
        self,
        mat: torch.Tensor,
        metadata: list,
        label_img: torch.Tensor,
        step: str,
        tag: str,
    ):
        global_step = self.get_global_step(step)
        self.writer.add_embedding(mat, metadata, label_img, global_step, tag)

    def add_hparams_summary(
        self, hparam_dict: dict, metric_dict: dict, run_name: str = None
    ):
        self.writer.add_hparams(hparam_dict, metric_dict, run_name=run_name)

    def close(self):
        self.writer.flush()
        self.writer.close()
