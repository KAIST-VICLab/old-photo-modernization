import torch
import logging

from abc import ABC, abstractmethod
from collections import OrderedDict

from .component.scheduler import get_scheduler
from src.data.dataloader import BaseDataLoader

log = logging.getLogger(__name__)


class BaseModel(ABC):
    # TODO someday: support multi GPU training and distributed training
    def __init__(
        self,
        model_config: OrderedDict,
        phase: str,
        device: torch.device,
        gpu_ids: list,
        verbose: bool = False,
        is_distributed: bool = False,
    ):
        self.model_config = model_config
        self.is_distributed = is_distributed

        # required variable from engine
        self.gpu_ids = gpu_ids
        self.device = device
        self.verbose = verbose
        self.is_train = phase == "training"

        self.schedulers = []
        self.optimizers = []

        self.loss_names = []
        self.model_names = []
        self.train_visual_names = []
        self.eval_visual_names = []
        self.output_names = []
        self.image_paths = []
        # TODO: include the way to handle multi GPU
        # TODO: include the way to handle distributed machine training
        self.training_state = {}
        self.metric = 0  # learning rate policy plateau

        self.is_training_train = True
        self.is_training_eval = False

    # TODO: add the way for grad_hook for debugging
    def setup(
        self,
        train_loader: BaseDataLoader = None,
        eval_loader: BaseDataLoader = None,
        checkpoint=None,
    ) -> None:
        """
        Print networks, create schedulers, parallelize
        :param checkpoint:
        """
        if self.is_train:
            self.schedulers = [
                get_scheduler(optimizer, self.model_config["training"]["lr_scheduler"])
                for optimizer in self.optimizers
            ]
        self.print_networks(self.verbose)
        self.to(self.device)

        if checkpoint is not None:
            self.load(checkpoint)

        if self.is_distributed:
            self.parallelize()

        if self.is_train:
            self.setup_optimizer()
            self.setup_loss()

    @abstractmethod
    def plot_model_graph(
        self, input_dict: dict = None, visualizer=None, input_key: str = "A"
    ) -> None:
        pass

    def to(self, device: torch.device, filter: list = None) -> None:
        for name in self.model_names:
            passed_filter = False
            if filter is None:
                passed_filter = True
            elif name in filter:
                passed_filter = True
            if isinstance(name, str) and passed_filter:
                net = getattr(self, name)
                setattr(self, name, net.to(device))

    # def serialize(self) -> None:
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             net = getattr(self, name)
    #             net_without_ddp = net
    #             setattr(self, f"{name}_without_ddp", net_without_ddp)

    def parallelize(self, filter: list = None) -> None:
        for name in self.model_names:
            passed_filter = False
            if filter is None:
                passed_filter = True
            elif name in filter:
                passed_filter = True

            if isinstance(name, str) and passed_filter:
                net = getattr(self, name)
                net = torch.nn.parallel.DistributedDataParallel(
                    net, device_ids=self.gpu_ids, find_unused_parameters=False
                )
                net_without_ddp = net.module
                setattr(self, name, net)
                setattr(self, name + "_without_ddp", net_without_ddp)

    @abstractmethod
    def setup_optimizer(self) -> None:
        pass

    @abstractmethod
    def setup_loss(self) -> None:
        pass

    @abstractmethod
    def data_dependent_initialize(self, data: dict) -> None:
        pass

    @abstractmethod
    def feed_data(self, data: dict) -> None:
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps
        """
        pass

    @abstractmethod
    def forward(self) -> None:
        """
        Run forward pass
        """
        pass

    @abstractmethod
    def optimize_parameters(
        self, current_epoch, current_epoch_iter, total_iter
    ) -> None:
        # TODO: add internal state for epoch, iter, or anything so
        # we can implement multi stage training
        """
        Calculate loss, gradients, and update network weights;
        called in every training iteration
        :param total_iter:
        :param current_epoch:
        :param current_epoch_iter:
        """
        pass

    def train(self) -> None:
        """
        Make train mode
        """
        self.is_training_train = True
        self.is_training_eval = False
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self) -> None:
        """
        Make eval mode
        """
        self.is_training_train = False
        self.is_training_eval = True
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    @abstractmethod
    def _forward_eval(self) -> None:
        pass

    def forward_eval(self) -> None:
        """
        Forward function in evaluation time (validation, etc)
        """
        with torch.no_grad():
            self._forward_eval()
            self.compute_eval_visuals()
            self.compute_output()

    def forward_test(self) -> None:
        """
        Forward function in evaluation time - testing
        :return:
        """
        with torch.no_grad():
            self.forward()
            self.compute_eval_visuals()

    # @abstractmethod
    # def compute_eval_losses(self) -> None:
    #     """
    #     Prpare evaluation losses for output later
    #     :return:
    #     """
    #     pass

    @abstractmethod
    def compute_train_visuals(self) -> None:
        """
        Prepare visuals for training visualization
        """
        pass

    @abstractmethod
    def compute_eval_visuals(self) -> None:
        """
        Prepare visuals for evaluation visualization
        :return:
        """
        pass

    @abstractmethod
    def compute_output(self) -> None:
        """
        Prepare output for performance measurement
        :return:
        """
        pass

    # External: visuals, saving & Loading network, std output

    def get_models(self) -> OrderedDict:
        model_dict = OrderedDict()
        for name in self.model_names:
            if isinstance(name, str):
                model_dict[name] = getattr(self, name)
        return model_dict

    def get_image_paths(self) -> list:
        """
        Return image paths for the current data
        """
        return self.image_paths

    def get_filename_details(self) -> list:
        """
        Return additional filename details for the current data
        """
        return []

    def get_train_visuals(self) -> OrderedDict:
        # TODO: change into property using decorator
        visual_dict = OrderedDict()
        for name in self.train_visual_names:
            if isinstance(name, str) and hasattr(self, name):
                visual_dict[name] = getattr(self, name)
        return visual_dict

    def get_eval_visuals(self) -> OrderedDict:
        visual_dict = OrderedDict()
        for name in self.eval_visual_names:
            if isinstance(name, str) and hasattr(self, name):
                visual_dict[name] = getattr(self, name)
        return visual_dict

    def get_current_output(self) -> OrderedDict:
        output_dict = OrderedDict()
        for name in self.output_names:
            if isinstance(name, str) and hasattr(self, name):
                output_dict[name] = getattr(self, name)
        return output_dict

    def get_current_losses(self) -> OrderedDict:
        losses_dict = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses_dict[name] = float(getattr(self, name))
        return losses_dict

    @abstractmethod
    def dump(self) -> dict:
        """
        Wrapper for dumping model state:
            - dumping every model's state
            - dumping optimizer
            - dumping other important states
        """
        # TODO: handling parallel saving
        return {}

    def dump_model(self) -> dict:
        model_state = {}
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if self.is_distributed and hasattr(net, "module"):
                    model_state[name] = net.module.state_dict()
                else:
                    model_state[name] = net.state_dict()
        return model_state

    @abstractmethod
    def dump_additional_internal_state(self) -> dict:
        return {}

    def dump_internal_state(self) -> dict:
        state = {"schedulers": [], "optimizers": []}
        for s in self.schedulers:
            state["schedulers"].append(s.state_dict())
        for o in self.optimizers:
            state["optimizers"].append(o.state_dict())
        additional_states = self.dump_additional_internal_state()
        state["additional_states"] = additional_states
        return state

    @abstractmethod
    def reset_internal_state(self):
        """
        Perform necessary operations to reset internal state, such as:
        - Updating learning rate
        :return:
        """
        pass

    @abstractmethod
    def load(self, checkpoint) -> None:
        """
        Wrapper for loading checkpoint
        """
        # TODO: handle data parallel
        pass

    def load_model(
        self, model_state: dict, strict: bool = True, filter: list = None
    ) -> None:
        for name in self.model_names:
            passed_filter = False
            if filter is None:
                passed_filter = True
            elif name in filter:
                passed_filter = True
            if isinstance(name, str) and passed_filter:
                model = getattr(self, name)
                current_model_state_dict = model_state[name]
                model.load_state_dict(current_model_state_dict, strict=strict)

    @abstractmethod
    def load_additional_internal_states(self, internal_state: dict) -> None:
        pass

    def load_internal_state(self, internal_state: dict) -> None:
        loaded_schedulers_state_dict = internal_state["schedulers"]
        loaded_optimizers_state_dict = internal_state["optimizers"]
        if self.is_train:
            assert len(loaded_schedulers_state_dict) == len(self.schedulers)
            assert len(loaded_optimizers_state_dict) == len(self.optimizers)

            for i, o in enumerate(loaded_optimizers_state_dict):
                self.optimizers[i].load_state_dict(o)
            for i, s in enumerate(loaded_schedulers_state_dict):
                self.schedulers[i].load_state_dict(s)

        self.load_additional_internal_states(internal_state)

    def print_networks(self, verbose: bool, filter: list = None) -> None:
        log.info(f"{'-'*5} Networks initialized {'-'*5}")
        for name in self.model_names:
            passed_filter = False
            if filter is None:
                passed_filter = True
            elif name in filter:
                passed_filter = True
            if isinstance(name, str) and passed_filter:
                net = getattr(self, name)
                num_params = sum(map(lambda x: x.numel(), net.parameters()))
                if verbose:
                    log.info(net)
                log.info(
                    "[Network %s] Total number of parameters: %.3f M"
                    % (name, num_params / 1e6)
                )
        log.info(f"{'='*30}")

    # Additional logics: learning rate + requires grad setter
    def _set_lr(self, lr_groups_l):
        """
        Set learning rate for warmup
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def _get_init_lr(self):
        """
        Get the initial lr for warming up depending on the optimizer
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        # Common
        for scheduler in self.schedulers:
            if self.model_config["training"]["lr_scheduler"]["policy"] == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        # Warming-up learning rate TODO: later
        if cur_iter < warmup_iter:
            # get initial learning rate for each group
            init_lr_g_l = self._get_init_lr()
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            self._set_lr(warm_up_lr_l)

        lr = self.optimizers[0].param_groups[0]["lr"]
        log.info("Update learning rate - current learning rate = {}".format(lr))

    def set_requires_grad(self, nets, requires_grad: bool = False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
