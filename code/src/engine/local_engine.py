import logging

from src.data.dataloader import build_dataloader
from src.engine.util import (
    set_random_seed,
    set_cudnn_benchmark,
    init_device,
    set_cudnn_deterministic,
)
from src.model import build_model

from src.util.logger import build_logger
from src.util.visualizer import build_visualizer
from src.util.checkpointer import build_checkpointer
from src.util.evaluator import build_performance_evaluator
from src.util.writer import build_image_writer

from .trainer import SingleGPUTrainer
from .evaluator import SingleGPUEvaluator
import random

from .base import BaseEngine
from collections import OrderedDict

log = logging.getLogger(__name__)


class LocalMachineSingleGPU(BaseEngine):  # Single Process == Single GPU
    def __init__(self, global_config: OrderedDict):
        """
        :param config: global config
        """
        super().__init__(global_config)

        log.info(
            "Initializing visualizer, performance evaluator, checkpointer, and logger"
        )
        self.visualizer = build_visualizer(self.engine_config)
        self.performance_evaluator = build_performance_evaluator(self.engine_config)
        self.image_writer = build_image_writer(self.engine_config)
        self.checkpointer = build_checkpointer(self.engine_config)
        self.logger = build_logger(self.engine_config)

        seed = self.engine_config.get("manual_seed", None)
        if seed is None:
            seed = random.randint(1, 10000)
        set_random_seed(seed)
        log.info(f"Setting random seed in engine with seed: {seed}")

        # additional GPU backend settings
        set_cudnn_benchmark(self.engine_config["cudnn_benchmark"])
        set_cudnn_deterministic(self.engine_config["cudnn_deterministic"])
        log.info(
            f"Setting cudnn benchmark: {self.engine_config['cudnn_benchmark']}, "
            f"deterministic: {self.engine_config['cudnn_deterministic']}"
        )

        self.gpu_ids = [self.engine_config["gpu_id"]]
        self.device = init_device(self.gpu_ids)

        log.info("Creating train and eval loader")
        self.train_loader = build_dataloader(
            self.datasets_config["train"]["dataloader"],
            self.datasets_config["train"]["dataset"],
            seed,
        )
        self.eval_loader = build_dataloader(
            self.datasets_config["eval"]["dataloader"],
            self.datasets_config["eval"]["dataset"],
            seed,
        )

        self.phase = self.engine_config["phase"]
        log.info(f"Phase: {self.phase}")
        log.info("Creating the model")
        self.model = build_model(
            self.model_config,
            self.phase,
            self.device,
            self.gpu_ids,
            self.engine_config["verbose"],
            is_distributed=False,
        )

        if self.phase == "training":
            log.info("Creating trainer instance")
            self.trainer = SingleGPUTrainer(
                model=self.model,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
                logger=self.logger,
                visualizer=self.visualizer,
                performance_evaluator=self.performance_evaluator,
                checkpointer=self.checkpointer,
                image_writer=self.image_writer,
                training_config=self.model_config["training"],
                engine_config=self.engine_config,
            )
        else:  # evaluation
            log.info("Creating evaluator instance")
            self.evaluator = SingleGPUEvaluator(
                model=self.model,
                eval_loader=self.eval_loader,
                logger=self.logger,
                visualizer=self.visualizer,
                performance_evaluator=self.performance_evaluator,
                checkpointer=self.checkpointer,
                image_writer=self.image_writer,
                eval_config=self.model_config["evaluation"],
                engine_config=self.engine_config,
            )

    def run(self):
        if self.phase == "training":
            self.trainer.train()
        elif self.phase == "evaluation":
            self.evaluator.eval()
        else:
            raise NotImplementedError
