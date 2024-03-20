from .performance import PerformanceEvaluator
from collections import OrderedDict


def build_performance_evaluator(engine_config: OrderedDict):
    return PerformanceEvaluator(engine_config["evaluator"])
