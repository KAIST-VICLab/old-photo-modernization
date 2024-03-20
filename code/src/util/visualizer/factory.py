from .base import BaseVisualizer
from .html import HTMLVisualizer
from .tensorboard import TensorboardVisualizer
from collections import OrderedDict
import torch
import logging

log = logging.getLogger(__name__)


class MultiInstanceVisualizer(BaseVisualizer):
    def __init__(self, instance_config: OrderedDict, visualizer_config: OrderedDict):
        super().__init__(instance_config, visualizer_config)

        self.visualizer_list = []
        instances = self.visualizer_config["instances"]
        for instance in instances:
            log.info(f"Creating visualizer instance: {instance}")
            visualizer = build_single_visualizer(instance, visualizer_config)
            self.visualizer_list.append(visualizer)

    def add_scalar(
        self, tag: str, scalar_value: float, step: str = None, wall_time=None
    ):
        for visualizer in self.visualizer_list:
            visualizer.add_scalar(
                tag=tag, scalar_value=scalar_value, step=step, wall_time=wall_time
            )

    def add_scalars(
        self, tag: str, scalar_dict: dict, step: str = None, wall_time=None
    ):
        for visualizer in self.visualizer_list:
            visualizer.add_scalars(
                tag=tag, scalar_dict=scalar_dict, step=step, wall_time=wall_time
            )

    # diagnostic model.weight and model.grad
    def add_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: str,
        bins: str = "tensorflow",
        wall_time=None,
    ):
        for visualizer in self.visualizer_list:
            visualizer.add_histogram(
                tag=tag, values=values, step=step, bins=bins, wall_time=wall_time
            )

    def add_image(
        self, tag: str, img_tensor: torch.Tensor, step: str = None, wall_time=None
    ):
        for visualizer in self.visualizer_list:
            visualizer.add_image(
                tag=tag, img_tensor=img_tensor, step=step, wall_time=wall_time
            )

    def add_images(  # for batch images
        self, tag: str, img_tensor: torch.Tensor, step: str = None, wall_time=None
    ):
        for visualizer in self.visualizer_list:
            visualizer.add_images(
                tag=tag, img_tensor=img_tensor, step=step, wall_time=wall_time
            )

    def add_image_dict(
        self, tag: str, img_tensor_dict: dict, step: str = None, wall_time=None
    ):
        for visualizer in self.visualizer_list:
            visualizer.add_image_dict(
                tag=tag, img_tensor_dict=img_tensor_dict, step=step, wall_time=wall_time
            )

    def add_video(
        self,
        tag: str,
        vid_tensor: torch.Tensor,
        fps: int = None,
        step: str = None,
        wall_time=None,
    ):
        for visualizer in self.visualizer_list:
            visualizer.add_video(
                tag=tag, vid_tensor=vid_tensor, fps=fps, step=step, wall_time=wall_time
            )

    # diagnostic: model
    def add_graph(
        self,
        model: torch.nn.Module,
        input_to_model: torch.Tensor,
        verbose: bool = False,
    ):
        for visualizer in self.visualizer_list:
            visualizer.add_graph(
                model=model, input_to_model=input_to_model, verbose=verbose
            )

    # diagnostic: data
    def add_embedding(
        self,
        mat: torch.Tensor,
        metadata: list,
        label_img: torch.Tensor,
        step: str,
        tag: str,
    ):
        for visualizer in self.visualizer_list:
            visualizer.add_embedding(mat, metadata, label_img, step, tag)

    # hyper parameters
    def add_hparams_summary(
        self, hparam_dict: dict, metric_dict: dict, run_name: str = None
    ):
        for visualizer in self.visualizer_list:
            visualizer.add_hparams_summary(
                hparam_dict=hparam_dict, metric_dict=metric_dict, run_name=run_name
            )

    def close(self):
        for visualizer in self.visualizer_list:
            visualizer.close()


def build_single_visualizer(
    instance: str, visualizer_config: OrderedDict
) -> BaseVisualizer:
    if instance == "html":
        instance_config = visualizer_config[instance]
        return HTMLVisualizer(instance_config, visualizer_config)
    elif instance == "tensorboard":
        instance_config = visualizer_config[instance]
        return TensorboardVisualizer(instance_config, visualizer_config)
    else:
        raise NotImplementedError
