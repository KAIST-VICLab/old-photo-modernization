from .factory import MultiInstanceVisualizer
from collections import OrderedDict


def build_visualizer(engine_config: OrderedDict):
    visualizer = MultiInstanceVisualizer({}, engine_config["visualizer"])
    return visualizer
