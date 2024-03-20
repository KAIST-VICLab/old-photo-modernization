import warnings
import logging
import time
import torch

from collections import OrderedDict
from ..util.checkpointer import Checkpointer
from ..util.evaluator import PerformanceEvaluator
from ..util.visualizer.base import BaseVisualizer
from ..util.logger import BaseLogger
from ..util.writer import ImageWriter
from ..data.dataloader import BaseDataLoader
from ..model import BaseModel

warnings.simplefilter("ignore", UserWarning)

log = logging.getLogger(__name__)


class DistributedTrainer:
    def __init__(
        self,
        model: BaseModel,
        train_loader: BaseDataLoader,
        eval_loader: BaseDataLoader,
        logger: BaseLogger,
        visualizer: BaseVisualizer,
        performance_evaluator: PerformanceEvaluator,
        checkpointer: Checkpointer,
        image_writer: ImageWriter,
        training_config: OrderedDict,
        engine_config: OrderedDict,
        global_rank: int,
    ):
        self.training_config = training_config
        self.engine_config = engine_config

        self.logger = logger
        self.visualizer = visualizer
        self.performance_evaluator = performance_evaluator
        self.checkpointer = checkpointer
        self.image_writer = image_writer

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model

        self.debug_mode = self.engine_config["debug_mode"]
        log.info(f"Engine debug mode: {self.debug_mode}")

        log.info(
            "The number of training images = {}".format(len(self.train_loader.dataset))
        )

        log.info("Model initialization")
        # Load everything from checkpoint
        continue_config = self.training_config["continue"]
        if continue_config["is_continue"]:
            log.info("Training by resuming from checkpoint")
            checkpoint = {}
            if continue_config["pretrained_step"] is not None:
                step = continue_config["pretrained_step"]
                log.info(
                    f"Load engine state, summary, and model state from checkpoint step: {step}"
                )
                checkpoint = self.checkpointer.load(step)
            elif continue_config["pretrained_path"] is not None:
                path = continue_config["pretrained_path"]
                log.info(
                    f"Load engine state, summary, and model state from checkpoint path: {path}"
                )
                checkpoint = self.checkpointer.direct_load(path)
            else:
                raise NotImplementedError

            # Load everything
            self.summary = checkpoint["summary"]
            engine_state = checkpoint["engine"]
            self.total_iters = engine_state["total_iters"]
            if engine_state["epoch_iter"] == len(self.train_loader):
                # start as the next epoch
                self.start_epoch = engine_state["current_epoch"] + 1
            else:
                self.start_epoch = engine_state["current_epoch"]
            self.model.setup(train_loader=train_loader, eval_loader=eval_loader)
        else:
            log.info("Training from scratch")
            self.summary = {}
            self.total_iters = 0
            self.start_epoch = self.training_config["start_epoch"]  # base-1
            self.model.setup(train_loader=train_loader, eval_loader=eval_loader)

        self.n_epochs = self.training_config["n_epochs"]
        self.n_epochs_decay = self.training_config["n_epochs_decay"]
        log.info(
            f"Trainer is initialized with start epoch: {self.start_epoch}, "
            f"n_epoch: {self.n_epochs + self.n_epochs_decay}, "
            f"total_iters: {self.total_iters}, "
            f"summary: {self.summary}"
        )

        # distributed
        self.global_rank = global_rank

    def train(self):
        optim_time = 0.1

        log.info("Initial evaluation before training")
        self.eval()

        log.info("Training - Train start")
        self.model.train()
        for current_epoch in range(
            self.start_epoch, self.n_epochs + self.n_epochs_decay + 1
        ):
            # Initialization
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            self.train_loader.set_epoch(current_epoch)
            log.info(f"Train - current epoch: {current_epoch}")
            for i, data in enumerate(self.train_loader):
                iter_start_time = time.time()

                self.total_iters += 1
                epoch_iter += 1
                # batch_size = data[self.training_config["input_key"]].size(0)

                if self.logger.time2log(current_epoch, self.total_iters):
                    data_time = iter_start_time - iter_data_time

                if current_epoch == self.start_epoch and i == 0:
                    self.model.data_dependent_initialize(data)
                    self.model.plot_model_graph(
                        data,
                        self.visualizer,
                        input_key=self.training_config["input_key"],
                    )

                optimize_start_time = time.time()

                self.model.feed_data(data)
                self.model.optimize_parameters(
                    current_epoch, epoch_iter, self.total_iters
                )

                torch.cuda.synchronize()

                # it's better to use batch processing time instead of per item time
                optim_time = (
                    time.time() - optimize_start_time
                ) * 0.005 + 0.995 * optim_time

                utility_start_time = time.time()

                if self.visualizer.time2visualize(current_epoch, self.total_iters):
                    # Knowledge: This visualization takes around 0.3 second
                    log.info("visualize intermediate training results")
                    self.model.compute_train_visuals()
                    self.visualizer.add_image_dict(
                        tag="visuals/training/train",
                        img_tensor_dict=self.model.get_train_visuals(),
                        step=f"e{current_epoch}_ei{epoch_iter}_ti{self.total_iters}",
                    )
                    if self.debug_mode:
                        model_dict = self.model.get_models()
                        for model_name, model in model_dict.items():
                            for n, p in model.named_parameters():
                                if (
                                    p is not None
                                    and p.requires_grad
                                    and "bias" not in n
                                    and p.grad is not None
                                ):
                                    tag = n.replace(".", "/")
                                    tag = model_name + "/" + tag
                                    self.visualizer.add_histogram(
                                        tag,
                                        p.data,
                                        f"e{current_epoch}_ei{epoch_iter}_ti{self.total_iters}",
                                    )
                                    self.visualizer.add_histogram(
                                        tag + "/grad",
                                        p.grad.data,
                                        f"e{current_epoch}_ei{epoch_iter}_ti{self.total_iters}",
                                    )

                if self.global_rank == 0 and self.checkpointer.time2checkpoint(
                    current_epoch, self.total_iters, check_by="iter"
                ):  # save by iter
                    # Knowledge: Checkpointing takes around 1.7 second
                    log.info(
                        "saving the latest model (epoch %d, total_iters %d)"
                        % (current_epoch, self.total_iters)
                    )
                    saved_state = self.model.dump()
                    step = f"e{current_epoch}_ei{epoch_iter}_ti{self.total_iters}"
                    engine_state = {
                        "current_epoch": current_epoch,
                        "epoch_iter": epoch_iter,
                        "total_iters": self.total_iters,
                    }
                    self.summary["losses/train"] = self.model.get_current_losses()
                    self.checkpointer.save(
                        step,
                        saved_state["model"],
                        saved_state["model_internal_state"],
                        engine_state,
                        self.summary,
                    )

                if self.logger.time2log(current_epoch, self.total_iters):
                    # Knowledge: this visualization takes < 0.1 second
                    losses = self.model.get_current_losses()
                    self.visualizer.add_scalars(
                        "losses/training/train",
                        losses,
                        step=f"e{current_epoch}_ei{epoch_iter}_ti{self.total_iters}",
                    )

                    utility_time = time.time() - utility_start_time
                    BaseLogger.log_train_iter(
                        log,
                        current_epoch,
                        self.n_epochs + self.n_epochs_decay,
                        epoch_iter,
                        len(self.train_loader),
                        losses,
                        optim_time,
                        data_time,
                        utility_time,
                    )

                if self.performance_evaluator.time2evaluate(
                    current_epoch, self.total_iters, check_by="iter"
                ):
                    log.info(f"Training - Evaluation - Iter: {self.total_iters}")
                    self.eval()

                iter_data_time = time.time()

            if self.performance_evaluator.time2evaluate(
                current_epoch, self.total_iters, check_by="epoch"
            ):
                log.info(f"Training - Evaluation - Epoch: {current_epoch}")
                self.eval()

            # Knowledge: Updating learning rate is better after optimizing
            self.model.update_learning_rate(current_epoch)

            if self.global_rank == 0 and self.checkpointer.time2checkpoint(
                current_epoch, self.total_iters, check_by="epoch"
            ):
                log.info(
                    f"saving the model at the end of epoch: {current_epoch}"
                    f", total iters: {self.total_iters}"
                )
                saved_state = self.model.dump()
                step = f"e{current_epoch}_ei{epoch_iter}_ti{self.total_iters}"
                engine_state = {
                    "current_epoch": current_epoch,
                    "epoch_iter": epoch_iter,
                    "total_iters": self.total_iters,
                }
                self.summary["losses/train"] = self.model.get_current_losses()

                self.checkpointer.save(
                    step,
                    saved_state["model"],
                    saved_state["model_internal_state"],
                    engine_state,
                    self.summary,
                )
                self.checkpointer.save(
                    "latest",
                    saved_state["model"],
                    saved_state["model_internal_state"],
                    engine_state,
                    self.summary,
                )

            # some loggings
            log.info(
                "Training - Train - epoch end %d / %d \t Time taken: %d sec"
                % (
                    current_epoch,
                    self.training_config["n_epochs"]
                    + self.training_config["n_epochs_decay"],
                    time.time() - epoch_start_time,
                )
            )

    @torch.no_grad()
    def eval(self):
        log.info("Training - Eval start")
        global_eval_metrics = OrderedDict()
        global_eval_losses = OrderedDict()
        self.model.eval()

        eval_start_time = time.time()
        total_iters = min(len(self.eval_loader), self.performance_evaluator.n_eval)
        for i, data in enumerate(self.eval_loader):
            if i >= self.performance_evaluator.n_eval:
                break

            iter_start_time = time.time()
            self.model.feed_data(data)
            self.model.forward_eval()

            torch.cuda.synchronize()

            if i == 0:
                log.info("visualize evaluation results")
                self.visualizer.add_image_dict(
                    tag="visuals/training/eval",
                    img_tensor_dict=self.model.get_eval_visuals(),
                    step=f"eval:e0_ei0_ti{self.total_iters}",
                )

            current_losses = self.model.get_current_losses()
            current_output = self.model.get_current_output()
            current_metrics = self.performance_evaluator.calculate_metrics(
                current_output
            )
            if self.image_writer.is_output_visual():
                # naming is different compared to the evaluator (evaluation phase)
                self.image_writer.write(
                    self.model.get_eval_visuals(),
                    step=f"val_ti{self.total_iters}",
                    filename=f"{i:06d}",
                )

            for metric_name, metric_val in current_metrics.items():
                if metric_name not in global_eval_metrics:
                    global_eval_metrics[metric_name] = metric_val
                else:
                    global_eval_metrics[metric_name] += metric_val

            for loss_name, loss_val in current_losses.items():
                if loss_name not in global_eval_losses:
                    global_eval_losses[loss_name] = loss_val
                else:
                    global_eval_losses[loss_name] += loss_val

            iter_end_time = time.time()
            if i % 10 == 0:
                BaseLogger.log_eval_iter(
                    log,
                    i,
                    total_iters,
                    current_metrics,
                    current_losses,
                    iter_end_time - iter_start_time,
                )

        for metric_name, _ in global_eval_metrics.items():
            global_eval_metrics[metric_name] /= total_iters
        for loss_name, _ in global_eval_losses.items():
            global_eval_losses[loss_name] /= total_iters

        eval_end_time = time.time()
        BaseLogger.log_training_eval(
            log,
            self.total_iters,
            global_eval_metrics,
            global_eval_losses,
            eval_end_time - eval_start_time,
        )
        self.visualizer.add_scalars(
            tag="losses/training/eval",
            scalar_dict=global_eval_losses,
            step=f"e0_ei0_ti{self.total_iters}",
        )
        self.visualizer.add_scalars(
            tag="metric/training/eval",
            scalar_dict=global_eval_metrics,
            step=f"e0_ei0_ti{self.total_iters}",
        )
        self.summary["metric/eval"] = global_eval_metrics
        self.summary["losses/eval"] = global_eval_losses

        self.model.train()
