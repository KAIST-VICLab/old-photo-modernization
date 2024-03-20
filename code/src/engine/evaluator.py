from collections import OrderedDict
from ..data.dataloader import BaseDataLoader
from ..util.checkpointer import Checkpointer
from ..util.logger import BaseLogger
from ..util.visualizer.base import BaseVisualizer
from ..util.evaluator.performance import PerformanceEvaluator
from ..util.writer import ImageWriter
from ..model.base import BaseModel

from pathlib import Path
import logging

log = logging.getLogger(__name__)


class SingleGPUEvaluator:
    def __init__(
        self,
        model: BaseModel,
        eval_loader: BaseDataLoader,
        logger: BaseLogger,
        visualizer: BaseVisualizer,
        performance_evaluator: PerformanceEvaluator,
        checkpointer: Checkpointer,
        image_writer: ImageWriter,
        eval_config: OrderedDict,
        engine_config: OrderedDict,
    ):
        self.eval_config = eval_config
        self.engine_config = engine_config

        self.logger = logger
        self.visualizer = visualizer
        self.performance_evaluator = performance_evaluator
        self.checkpointer = checkpointer
        self.image_writer = image_writer

        self.eval_loader = eval_loader
        self.model = model

        self.is_eval = self.engine_config["phase"] == "evaluation"

        # Load everything from checkpoint
        checkpoint = {}
        if self.eval_config["pretrained_step"] is not None:
            log.info(
                f"Load checkpoint from step: {self.eval_config['pretrained_step']}"
            )
            step = self.eval_config["pretrained_step"]
            checkpoint = self.checkpointer.load(step)
        elif self.eval_config["pretrained_path"] is not None:
            log.info(
                f"Load checkpoint from path: {self.eval_config['pretrained_path']}"
            )
            path = self.eval_config["pretrained_path"]
            checkpoint = self.checkpointer.direct_load(path)
        else:
            raise NotImplementedError

        log.info(
            "The number of evaluation images = {}".format(len(self.eval_loader.dataset))
        )

        # Load everything for evaluation: total_iters + model
        engine_state = checkpoint["engine"]
        self.total_iters = engine_state["total_iters"]

        log.info(
            f"Model - initialization and load from checkpoint - latest total iters: {self.total_iters}"
        )
        self.model.setup()
        self.model.load(checkpoint)

        log.info("Evaluation start")

    def eval(self):
        # global_metric = {}
        self.model.eval()
        for i, data in enumerate(self.eval_loader):
            if i >= self.eval_config["n_test"]:
                break

            self.model.feed_data(data)
            self.model.forward_test()

            visuals = self.model.get_eval_visuals()
            img_paths = self.model.get_image_paths()
            filename_details = self.model.get_filename_details()

            filenames = []
            if filename_details is None:
                for path in img_paths:
                    filenames.append(f"{Path(path).stem}")
            else:
                for path, detail in zip(img_paths, filename_details):
                    filenames.append(f"{Path(path).stem}_{detail}")
            self.image_writer.write_batch(
                visuals, step=f"test_ti{self.total_iters}", filenames=filenames
            )

            if i % 10 == 0:
                log.info("processing (%04d)-th batch... %s" % (i, img_paths[0]))

            # Testing: No need to output the metric since usually the ground truth is not available
            # output = self.model.get_current_output()
            # current_metric = self.performance_evaluator.calculate_metrics(output)
            #
            # for metric_name, metric_val in current_metric.items():
            #     if metric_name not in global_metric:
            #         global_metric[metric_name] = metric_val
            #     else:
            #         global_metric[metric_name] += metric_val

        log.info("Evaluation finish")
