from collections import OrderedDict
from .image import ImagePSNR, ImageSSIM
import logging

log = logging.getLogger(__name__)

REQUIRED_PARAMS = {"ImagePSNR": [], "ImageSSIM": []}

REQUIRED_CALL_PARAMS = {
    "ImagePSNR": ["image_out", "image_gt"],
    "ImageSSIM": ["image_out", "image_gt"],
}


def assert_param(instance: str, params: OrderedDict):
    for required_param in REQUIRED_PARAMS[instance]:
        assert required_param in params


def build_metric(metric_config: OrderedDict):
    return MultiInstanceMetric(metric_config)


def build_single_metric(instance: str, params: OrderedDict):
    if instance == "ImagePSNR":
        return ImagePSNR()
    elif instance == "ImageSSIM":
        return ImageSSIM()
    else:
        raise NotImplementedError


class MultiInstanceMetric:
    def __init__(self, metric_config: OrderedDict):
        self.metric_config = metric_config
        self.metric_params = metric_config["params"]

        self.instance_list = metric_config["instances"]
        self.metric_list = []
        log.info("Creating performance evaluator with multi instance metric")
        for instance in self.instance_list:
            log.info(f"Creating metric instance: {instance}")
            self.metric_list.append(build_single_metric(instance, self.metric_params))

    def __call__(self, output: dict) -> dict:
        # TODO: assert in case needed
        metric_dict = {}
        for metric in self.metric_list:
            value = metric(output)
            name = str(metric)
            metric_dict[name] = value
        return metric_dict
