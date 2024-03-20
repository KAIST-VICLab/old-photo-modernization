import logging
from collections import OrderedDict


class BaseLogger:
    """
    Class to save the state of logger
    """

    def __init__(self, config: OrderedDict, is_distributed: bool = False):
        self.config = config
        self.is_distributed = is_distributed

    def time2log(self, current_epoch: int, current_iter: int) -> bool:
        if current_iter % self.config["iter_freq"] == 0:
            return True
        else:
            return False

    @staticmethod
    def log_train_iter(
        logger: logging.Logger,
        current_epoch: int,
        n_epoch: int,
        epoch_iter: int,
        n_iters_per_epoch: int,
        losses: dict,
        optim_time: float,
        dataload_time: float,
        utility_time: float,
    ):
        message = (
            f"[Epoch: {current_epoch}/{n_epoch}, "
            f"epoch iter: {epoch_iter}/{n_iters_per_epoch}, "
            f"optim time: {optim_time:.3f}, dataload time: {dataload_time:.3f}, "
            f"utility time: {utility_time:.3f}] "
        )
        # TODO: synchronize between process
        for loss_name, loss_value in losses.items():
            message += f"{loss_name}: {loss_value:.6f} "

        logger.info(message)  # print the message

    @staticmethod
    def log_eval_iter(
        logger: logging.Logger,
        current_iter: int,
        total_iter: int,
        current_metrics: dict,
        current_losses: dict,
        iter_time: float,
    ):
        message = (
            f"Eval - Iter:({current_iter}/{total_iter}) | Time: {iter_time:.4f} | "
        )
        # TODO: syncrhonize between process
        current_metrics_str = ""
        for metric_name, metric_val in current_metrics.items():
            current_metrics_str += f"{metric_name}: {metric_val:.4f} "
        # TODO: synchronize between process
        current_losses_str = ""
        for loss_name, loss_val in current_losses.items():
            current_losses_str += f"{loss_name}: {loss_val:.4f} "
        message = message + current_metrics_str + "| " + current_losses_str

        logger.info(message)

    @staticmethod
    def log_training_eval(
        logger: logging.Logger,
        global_iter: int,
        global_eval_metrics: dict,
        global_eval_losses: dict,
        time: float,
    ):
        message = f"Training - Eval finished - Global Iter: {global_iter} | Time: {time:.4f} | "
        # TODO: synchronize between process
        current_metrics_str = ""
        for metric_name, metric_val in global_eval_metrics.items():
            current_metrics_str += f"{metric_name}: {metric_val:.4f} "
        # TODO: synchronize
        current_losses_str = ""
        for loss_name, loss_val in global_eval_losses.items():
            current_losses_str += f"{loss_name}: {loss_val:.4f} "
        message = message + current_metrics_str + "| " + current_losses_str

        logger.info(message)
