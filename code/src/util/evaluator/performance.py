from collections import OrderedDict
from .metric import build_metric


class PerformanceEvaluator:
    """
    Class to compute performance metric
    """

    def __init__(self, config: OrderedDict):
        self.config = config
        self.metric = build_metric(config["metric"])
        self.n_eval = self.config["n_eval"]
        self.root_dir = self.config["root_dir"]

    def time2evaluate(
        self, current_epoch: int, current_iter: int, check_by: str = "epoch"
    ) -> bool:
        if check_by == "epoch":
            if current_epoch % self.config["epoch_freq"] == 0:
                return True
            else:
                return False
        elif check_by == "iter":
            if current_iter % self.config["iter_freq"] == 0:
                return True
            else:
                return False
        else:
            raise NotImplementedError

    def calculate_metrics(self, output):
        return self.metric(output)
