from abc import ABC, abstractmethod
from collections import OrderedDict
import torch


class BaseVisualizer(ABC):
    def __init__(self, instance_config: OrderedDict, visualizer_config: OrderedDict):
        self.instance_config = instance_config
        self.visualizer_config = visualizer_config

    def time2visualize(self, current_epoch: int, current_iter: int) -> bool:
        if current_iter % self.visualizer_config["iter_freq"] == 0:
            return True
        else:
            return False

    @abstractmethod
    def add_scalar(
        self, tag: str, scalar_value: float, step: str = None, wall_time=None
    ):
        pass

    @abstractmethod
    def add_scalars(
        self, tag: str, scalar_dict: dict, step: str = None, wall_time=None
    ):
        pass

    @abstractmethod
    def add_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: str,
        bins: str = "tensorflow",
        wall_time=None,
    ):
        pass

    @abstractmethod
    def add_image(
        self, tag: str, img_tensor: torch.Tensor, step: str = None, wall_time=None
    ):
        pass

    @abstractmethod
    def add_images(  # for batch images
        self, tag: str, img_tensor: torch.Tensor, step: str = None, wall_time=None
    ):
        pass

    @abstractmethod
    def add_image_dict(
        self, tag: str, img_tensor_dict: dict, step: str = None, wall_time=None
    ):
        pass

    @abstractmethod
    def add_video(
        self,
        tag: str,
        vid_tensor: torch.Tensor,
        fps: int = None,
        step: str = None,
        wall_time=None,
    ):
        pass

    @abstractmethod
    def add_graph(
        self,
        model: torch.nn.Module,
        input_to_model: torch.Tensor,
        verbose: bool = False,
    ):
        pass

    # someday: for visualizing embedding data better
    @abstractmethod
    def add_embedding(
        self,
        mat: torch.Tensor,
        metadata: list,
        label_img: torch.Tensor,
        step: str,
        tag: str,
    ):
        pass

    @abstractmethod
    def add_hparams_summary(
        self, hparam_dict: dict, metric_dict: dict, run_name: str = None
    ):
        pass

    @abstractmethod
    def close(self):
        pass
