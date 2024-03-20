import torch
import torch.nn.functional as F
import logging
import numpy as np

from ..base import BaseModel
from src.util.visualizer.palette import COCOSTUFF_PALETTE
from torch import nn
from src.model.component.loss import ContextualLoss_forward, WeightedAverage, GANLoss

from src.model.component.network.gan.discriminator import NLayerDiscriminator

from src.model.component.network.style_transfer.mrsf.network import (
    DirectMRSFNetwork,
    MergingRefinementNetwork,
)
from src.data.dataloader import BaseDataLoader
from src.model.component.init import init_net


log = logging.getLogger(__name__)


class DirectMRSFModel(BaseModel):
    def __init__(
        self, model_config, phase, device, gpu_ids, verbose=False, is_distributed=False
    ):
        super(DirectMRSFModel, self).__init__(
            model_config, phase, device, gpu_ids, verbose, is_distributed
        )

        if phase == "evaluation":
            self.multistage_conf = self.model_config["evaluation"]["multistage"]
            self.stage = self.multistage_conf["stage"]
        else:
            self.multistage_conf = self.model_config["training"]["multistage"]
            self.stage = self.multistage_conf["stage"]

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        if self.is_train:
            self.train_visual_names = [
                "content",
                "style_1",
                "style_2",
                "GT",
                "content_annotation",
                "style_1_annotation",
                "style_2_annotation",
                "out",
                "out_1",
                "out_2",
                "GT_m1",
                "GT_m2",
                "ST_m1",
                "ST_m2",
            ]

        if phase == "evaluation":
            self.eval_visual_names = [
                "content",
                "style_1",
                "style_2",
                "content_annotation",
                "out",
                "out_1",
                "out_2",
                "att_1",
                "att_2",
            ]
        else:  # training phase
            self.eval_visual_names = [
                "content",
                "style_1",
                "style_2",
                "content_annotation",
                "out",
                "out_1",
                "out_2",
                "att_1",
                "att_2",
            ]

        self.output_names = []

        model_params = self.model_config["params"]
        self.model_params = model_params

        # style transfer backbone
        encoder_path = self.model_params["st_backbone"]["encoder_path"]
        decoder_path = self.model_params["st_backbone"]["decoder_path"]
        if self.stage == 1:
            self.model_names = ["netS"]

            self.netS = DirectMRSFNetwork(
                encoder_path, decoder_path, device=self.device
            )
        elif self.stage == 2:
            self.model_names = ["netS", "netMR"]

            self.netS = DirectMRSFNetwork(
                encoder_path, decoder_path, device=self.device
            )

            pretrained_path = self.multistage_conf["pretrained_path"]
            checkpoint = torch.load(pretrained_path)
            model_state_dict = checkpoint["model"]
            self.load_model(model_state_dict, strict=True, filter=["netS"])
            self.set_requires_grad(self.netS, False)

            # simple sanity check
            assert torch.allclose(
                self.netS.local_global_fusion_block.l1_std_fusion[0]
                .weight.mean()
                .to(self.device),
                model_state_dict["netS"][
                    "local_global_fusion_block.l1_std_fusion.0.weight"
                ]
                .mean()
                .to(self.device),
            )
            assert torch.allclose(
                self.netS.local_global_fusion_block.l1_std_fusion[0]
                .weight.sum()
                .to(self.device),
                model_state_dict["netS"][
                    "local_global_fusion_block.l1_std_fusion.0.weight"
                ]
                .sum()
                .to(self.device),
            )

            netMR_params = self.model_params["netMR"]
            self.netMR = MergingRefinementNetwork(**netMR_params, device=self.device)
            init_net(
                self.netMR,
                self.model_params["init_type"],
                self.model_params["init_gain"],
            )

            netD_params = self.model_params["netD"]
            self.netD = NLayerDiscriminator(
                **netD_params, norm_layer=nn.InstanceNorm2d
            ).to(self.device)
        else:
            raise NotImplementedError

        self.set_requires_grad([self.netS.encoder, self.netS.decoder], False)

        # label_count
        self.label_count = self.model_params["label_count"]

        self.coco_palette = torch.tensor(
            COCOSTUFF_PALETTE, dtype=torch.int64, device=self.device
        )
        # TODO: decay based on paper

    def setup(
        self,
        train_loader: BaseDataLoader = None,
        eval_loader: BaseDataLoader = None,
        checkpoint=None,
    ) -> None:
        self.print_networks(True)
        self.to(self.device)

        if checkpoint is not None:
            self.load(checkpoint)

        if self.is_distributed:
            if self.stage == 1:
                self.parallelize(filter=["netS"])
            elif self.stage == 2:
                self.parallelize(filter=["netMR"])
            else:
                raise NotImplementedError

        if self.is_train:
            self.setup_optimizer()
            self.setup_loss()

    def setup_loss(self):
        self.loss_config = self.model_config["training"]["loss"]

        self.criterionMSEMean = torch.nn.MSELoss().to(self.device)
        if self.stage == 1:
            self.loss_names = [
                "loss_S",
                "loss_img_single",
                "loss_content_percep_single",
                "loss_style_contextual_single",
            ]
            self.criterionL1 = torch.nn.L1Loss(reduction="none").to(self.device)
            self.criterionContextualForward = ContextualLoss_forward().to(self.device)

            self.loss_weight = self.loss_config["stage_1"]
        elif self.stage == 2:
            self.loss_names = [
                "loss_MR",
                "loss_img_merging_refinement",
                "loss_percep_merging_refinement",
                "loss_smoothness_merging_refinement",
                "loss_GAN",
                "loss_D",
                "loss_D_real",
                "loss_D_fake",
            ]
            self.criterionL1Mean = torch.nn.L1Loss().to(self.device)
            self.weighted_layer = WeightedAverage()
            self.criterionGAN = GANLoss("lsgan").to(self.device)

            self.loss_weight = self.loss_config["stage_2"]

        else:
            raise NotImplementedError

    def setup_optimizer(self):
        optim_config = self.model_config["training"]["optimizer"]
        if self.stage == 1:
            netS_optim_params = optim_config["netS"]["params"]
            self.optimizerS = torch.optim.Adam(
                self.netS.parameters(),
                lr=netS_optim_params["lr"],
                betas=(netS_optim_params["beta1"], netS_optim_params["beta2"]),
            )
            self.optimizers.append(self.optimizerS)
        elif self.stage == 2:
            netMR_optim_params = optim_config["netMR"]["params"]
            netD_optim_params = optim_config["netD"]["params"]
            self.optimizerMR = torch.optim.Adam(
                self.netMR.parameters(),
                lr=netMR_optim_params["lr"],
                betas=(netMR_optim_params["beta1"], netMR_optim_params["beta2"]),
            )
            self.optimizerD = torch.optim.Adam(
                self.netD.parameters(),
                lr=netD_optim_params["lr"],
                betas=(netD_optim_params["beta1"], netD_optim_params["beta2"]),
            )
            self.optimizers.append(self.optimizerMR)
            self.optimizers.append(self.optimizerD)
        else:
            raise NotImplementedError

    def data_dependent_initialize(self, data):
        log.info("Data dependent initialization")
        super().data_dependent_initialize(data)
        pass

    def plot_model_graph(
        self, input_dict: dict = None, visualizer=None, input_key: str = "A"
    ) -> None:
        log.info("Plot model graph")
        pass

    def feed_data(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.content = input["content"].to(self.device)
        for i in range(len(input["style_img_list"])):
            setattr(
                self,
                "style_{}".format(i + 1),
                input["style_img_list"][i].to(self.device),
            )
        self.style_img_list = [
            style_img.to(self.device) for style_img in input["style_img_list"]
        ]

        self.image_paths = input["img_path"]

        # One-hot semantic map
        bs, _, h, w = self.content.size()
        nc = self.label_count
        if self.is_train:
            self.GT = input["GT"].to(self.device)

            self.GT_m1 = input["content_mask_list"][0].to(self.device)
            self.GT_m2 = input["content_mask_list"][1].to(self.device)

            self.gt_mask_list = input["content_mask_list"]
            self.gt_mask_list = [
                gt_mask.to(self.device) for gt_mask in self.gt_mask_list
            ]

            self.ST_m1 = input["style_mask_list"][0].to(self.device)
            self.ST_m2 = input["style_mask_list"][1].to(self.device)

            self.style_mask_list = input["style_mask_list"]
            self.style_mask_list = [
                st_mask.to(self.device) for st_mask in self.style_mask_list
            ]

            self.content_annotation = input["content_annotation"].to(
                self.device
            )  # BS, H, W
            self.style_annotation_list = [
                style_annotation.to(self.device)
                for style_annotation in input["style_annotation_list"]
            ]  # len, BS, H, W

            self.content_annotation = torch.unsqueeze(self.content_annotation, 1)
            self.style_annotation_list = [
                torch.unsqueeze(style_annotation, 1)
                for style_annotation in self.style_annotation_list
            ]

            self.style_1_annotation = self.style_annotation_list[0]
            self.style_2_annotation = self.style_annotation_list[1]

            # Preprocess semantic map

            self.identity_content_img = input["identity_content_img"].to(self.device)

            self.identity_content_annotation = input["identity_content_annotation"].to(
                self.device
            )
            self.identity_content_annotation = torch.unsqueeze(
                self.identity_content_annotation, 1
            )
            identity_content_semantics = torch.zeros(
                (bs, nc, h, w), device=self.device, dtype=torch.float
            )
            self.identity_content_semantics = torch.scatter(
                identity_content_semantics,
                dim=1,
                index=self.identity_content_annotation,
                value=1.0,
            )

            # DEBUGGING
            self.rotated_GT_img = input["rotated_GT"].to(self.device)
            self.rotated_GT_annotation = input["rotated_GT_annotation"].to(self.device)
            self.rotated_GT_annotation = torch.unsqueeze(self.rotated_GT_annotation, 1)
            rotated_GT_semantics = torch.zeros(
                (bs, nc, h, w), device=self.device, dtype=torch.float
            )
            self.rotated_GT_semantics = torch.scatter(
                rotated_GT_semantics, dim=1, index=self.rotated_GT_annotation, value=1.0
            )

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.is_train:
            (
                self.out_list,
                intermediate_feats_dict,
                similarity_map_list,
            ) = self.netS.transfer(self.content, self.style_img_list)
            for i in range(len(self.out_list)):
                setattr(self, "out_{}".format(i + 1), self.out_list[i])
            self.out_1_feats, _ = self.netS.encoder.forward_with_intermediate(
                self.out_1
            )
            self.GT_feats, _ = self.netS.encoder.forward_with_intermediate(self.GT)

            if self.stage == 2:
                # Cannot do backward to the same block two times, that's why detach is important
                self.out = self.netMR(
                    intermediate_feats_dict, similarity_map_list
                )  # prevent netR gradient to flow to netS
                self.out_feats, _ = self.netS.encoder.forward_with_intermediate(
                    self.out
                )

        else:
            (
                self.out_list,
                intermediate_feats_dict,
                similarity_map_list,
            ) = self.netS.transfer(self.content, self.style_img_list)
            for i in range(len(self.out_list)):
                setattr(self, "out_{}".format(i + 1), self.out_list[i])

            if self.stage == 2:
                self.out, attention_weight = self.netMR(
                    intermediate_feats_dict,
                    similarity_map_list,
                    return_attention_weight=True,
                )
                for i in range(len(attention_weight)):
                    setattr(self, "att_{}".format(i + 1), attention_weight[i])

    def forward_test(self) -> None:
        with torch.no_grad():
            self.forward()
            self.compute_eval_visuals()

    def optimize_parameters(self, current_epoch, current_epoch_iter, total_iter):
        if self.stage == 1:  # Single stylization subnet training
            self.forward()
            self.optimizerS.zero_grad()
            self.loss_S = self.compute_S_loss()
            self.loss_S.backward()
            self.optimizerS.step()
        elif self.stage == 2:  # Merging refinement subnet training
            self.forward()

            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizerD.zero_grad()
            self.loss_D = self.compute_D_loss()
            self.loss_D.backward()
            self.optimizerD.step()

            # update G
            self.set_requires_grad(self.netD, False)
            # update R only
            self.optimizerMR.zero_grad()
            self.loss_MR = self.compute_MR_loss()
            self.loss_MR.backward()
            self.optimizerMR.step()
        else:
            raise NotImplementedError

    def _forward_eval(self) -> None:
        if self.stage == 1:
            self.forward()
            self.loss_S = self.compute_S_loss().item()
        elif self.stage == 2:
            self.forward()
            self.loss_D = self.compute_D_loss().item()
            self.loss_MR = self.compute_MR_loss().item()
        else:
            raise NotImplementedError

    def compute_S_loss(self):
        eps = self.loss_weight["eps"]

        # Style transfer - single
        # Style loss
        loss_img_all_1 = self.criterionL1(self.out_1, self.GT)

        loss_img_region_1 = (loss_img_all_1 * self.gt_mask_list[0]).sum() / (
            self.gt_mask_list[0].sum() + eps
        )

        self.loss_img_single = loss_img_region_1
        self.loss_img_single = (
            self.loss_weight["img_single"] * self.loss_img_single
        )  # Mean

        # Content perceptual loss
        loss_content_percep_all_1 = self.criterionMSEMean(
            self.out_1_feats[4], self.GT_feats[4].detach()
        )

        self.loss_content_percep_single = loss_content_percep_all_1
        self.loss_content_percep_single = (
            self.loss_weight["content_percep_single"] * self.loss_content_percep_single
        )  # mean

        # Contextual style loss
        self.loss_style_contextual_single = torch.zeros(1, device=self.device)
        for level in [3, 4]:
            loss_contextual_all_1 = self.criterionContextualForward(
                self.out_1_feats[level], self.GT_feats[level].detach()
            )

            loss_contextual_all = loss_contextual_all_1
            self.loss_style_contextual_single = (
                self.loss_style_contextual_single + loss_contextual_all
            )
        self.loss_style_contextual_single = (
            self.loss_weight["style_contextual_single"]
            * self.loss_style_contextual_single
        )

        self.loss_S = (
            self.loss_img_single
            + self.loss_content_percep_single
            + self.loss_style_contextual_single
        )

        return self.loss_S

    def compute_MR_loss(self):
        # Merging refinement loss
        self.loss_img_merging_refinement = self.loss_weight[
            "img_merging_refinement"
        ] * self.criterionL1Mean(self.out, self.GT)
        self.loss_percep_merging_refinement = self.loss_weight[
            "percep_merging_refinement"
        ] * self.criterionMSEMean(self.out_feats[4], self.GT_feats[4].detach())

        # Local smoothness
        scale_factor = 1
        patch_size = 3
        alpha = 10
        weighted_out = self.weighted_layer(
            self.out, patch_size=patch_size, alpha=alpha, scale_factor=scale_factor
        )
        loss_smoothness = self.criterionMSEMean(
            F.interpolate(self.out, scale_factor=scale_factor), weighted_out
        )
        self.loss_smoothness_merging_refinement = (
            self.loss_weight["smoothness_merging_refinement"] * loss_smoothness
        )

        self.loss_GAN = (
            self.loss_weight["adversarial"]
            * self.criterionGAN(self.netD(self.out), True).mean()
        )
        self.loss_MR = (
            self.loss_img_merging_refinement
            + self.loss_percep_merging_refinement
            + self.loss_smoothness_merging_refinement
            + self.loss_GAN
        )

        return self.loss_MR

    def compute_D_loss(self):
        # Fake
        fake = self.out.detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        # Real
        pred_real = self.netD(self.GT)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        return self.loss_D

    def compute_train_visuals(self):
        super().compute_train_visuals()
        for name in self.train_visual_names:
            if isinstance(name, str) and hasattr(self, name):
                if name in [
                    "content_annotation",
                    "style_1_annotation",
                    "style_2_annotation",
                    "rotated_GT_annotation",
                    "identity_content_annotation",
                ]:
                    image_tensor = getattr(self, name)

                    image_tensor = torch.permute(
                        image_tensor, (0, 2, 3, 1)
                    )  # NCHW -> NHWC
                    image_rgb_tensor = torch.zeros_like(image_tensor)
                    for i in range(len(self.coco_palette)):
                        class_label = torch.tensor(i).long()
                        rgb_value = self.coco_palette[i]

                        image_rgb_tensor = torch.where(
                            image_tensor == class_label, rgb_value, image_rgb_tensor
                        )
                    image_tensor = image_rgb_tensor
                    image_tensor = torch.permute(
                        image_tensor, (0, 3, 1, 2)
                    )  # NHWC -> NCHW
                    image_tensor = image_tensor / 255.0
                    setattr(self, name, image_tensor)
                else:
                    image_tensor = getattr(self, name)
                    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
                    if name in ["att_1", "att_2"]:
                        image_tensor = image_tensor.repeat(1, 3, 1, 1)

                    setattr(self, name, image_tensor)

                    # image_tensor = getattr(self, name)
                    # image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
                    # image_tensor = torch.permute(image_tensor, (0, 2, 3, 1)) # NCHW -> NHWC
                    # image_tensor = image_tensor * 255.0
                    # setattr(self, name, image_tensor)

    def compute_eval_visuals(self):
        super().compute_eval_visuals()
        for name in self.eval_visual_names:
            if isinstance(name, str) and hasattr(self, name):
                if name in [
                    "content_annotation",
                    "style_1_annotation",
                    "style_2_annotation",
                    "rotated_GT_annotation",
                    "identity_content_annotation",
                ]:
                    image_tensor = getattr(self, name)

                    image_tensor = torch.permute(
                        image_tensor, (0, 2, 3, 1)
                    )  # NCHW -> NHWC
                    image_rgb_tensor = torch.zeros_like(image_tensor)
                    for i in range(len(self.coco_palette)):
                        class_label = torch.tensor(i).long()
                        rgb_value = self.coco_palette[i]

                        image_rgb_tensor = torch.where(
                            image_tensor == class_label, rgb_value, image_rgb_tensor
                        )
                    image_tensor = image_rgb_tensor
                    image_tensor = torch.permute(
                        image_tensor, (0, 3, 1, 2)
                    )  # NHWC -> NCHW
                    image_tensor = image_tensor / 255.0
                    setattr(self, name, image_tensor)
                else:
                    image_tensor = getattr(self, name)
                    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

                    if name in ["att_1", "att_2"]:
                        image_tensor = image_tensor.repeat(1, 3, 1, 1)

                    setattr(self, name, image_tensor)

    def compute_output(self):
        super().compute_output()

        for name in self.output_names:
            if isinstance(name, str) and hasattr(self, name):
                image_tensor = getattr(self, name)
                if type(image_tensor) == torch.Tensor:
                    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
                    image_tensor = torch.permute(
                        image_tensor, (0, 2, 3, 1)
                    )  # NCHW -> NHWC
                    image_tensor = image_tensor * 255.0
                    image_numpy = image_tensor.cpu().float().numpy().astype(np.uint8)
                    setattr(self, name, image_numpy)

    def dump(self):
        state = {}
        model_state = super().dump_model()
        model_internal_state = super().dump_internal_state()
        state["model"] = model_state
        state["model_internal_state"] = model_internal_state
        return state

    def dump_additional_internal_state(self):
        additional_states = {}
        return additional_states

    def reset_internal_state(self):
        reset_internal_state_params = self.model_config["training"]["continue"][
            "reset_model_internal_state"
        ]["params"]
        scheduler_step = reset_internal_state_params["scheduler_step"]
        for current_epoch in range(scheduler_step):
            self.update_learning_rate(current_epoch)
        raise NotImplementedError

    def load(self, checkpoint):
        model_state = checkpoint["model"]
        model_internal_state = checkpoint.get("model_internal_state", {})
        super().load_model(model_state, strict=False)
        if self.is_train:
            is_reset_internal_state = self.model_config["training"]["continue"][
                "reset_model_internal_state"
            ]["is_reset"]
            if is_reset_internal_state:
                log.info("Reset internal state")
                self.reset_internal_state()
            else:
                log.info("Load internal state")
                super().load_internal_state(model_internal_state)

    def load_additional_internal_states(self, internal_state: dict) -> None:
        pass

    def get_filename_details(self) -> list:
        return None
