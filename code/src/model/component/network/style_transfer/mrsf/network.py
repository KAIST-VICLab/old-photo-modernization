from src.model.component.init import init_net
from src.model.component.network.common.unet import UnetGenerator
from src.model.component.network.style_transfer.mrsf.modules.backbone import (
    Encoder,
    Decoder,
)


from .modules.core import (
    AlignmentBlock,
    CorrelationBlock,
    FeatureRegionWiseMerging,
    LocalGlobalFusionBlock,
    SimpleSingleTransformModule,
    StyleExtractorV2,
    calc_mean_std,
)
from torch import nn

import torch
import torch.nn.functional as F


class DirectMRSFNetwork(nn.Module):
    def __init__(self, encoder_path, decoder_path, device):
        super(DirectMRSFNetwork, self).__init__()
        self.device = device

        self.encoder = Encoder()
        self.decoder = Decoder()

        # Photo stlyization network
        # Encoder decoder
        encoder_state_dict = torch.load(encoder_path)
        decoder_state_dict = torch.load(decoder_path)

        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)

        self.label_count = 184

        self.style_extractor = StyleExtractorV2()
        self.correlation_block = CorrelationBlock()
        self.alignment_block = AlignmentBlock()
        self.transform_module = SimpleSingleTransformModule()
        self.local_global_fusion_block = LocalGlobalFusionBlock()

        init_net(self.style_extractor)
        init_net(self.correlation_block)
        init_net(self.alignment_block)
        init_net(self.transform_module)
        init_net(self.local_global_fusion_block)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {"encoder": {}, "decoder": {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            feats["encoder"][level] = x

        feats["decoder"][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            feats["decoder"][level - 1] = x
        return feats, skips

    def extract_global_style_code(self, multi_level_features):
        global_mean_l1, global_std_l1 = calc_mean_std(
            multi_level_features["decoder"][1]
        )
        global_mean_l2, global_std_l2 = calc_mean_std(
            multi_level_features["decoder"][2]
        )
        global_mean_l3, global_std_l3 = calc_mean_std(
            multi_level_features["decoder"][3]
        )
        global_mean_l4, global_std_l4 = calc_mean_std(
            multi_level_features["decoder"][4]
        )

        global_mean_code = [
            global_mean_l1,
            global_mean_l2,
            global_mean_l3,
            global_mean_l4,
        ]
        global_std_code = [global_std_l1, global_std_l2, global_std_l3, global_std_l4]

        return global_mean_code, global_std_code

    def extract_style_code(self, style_img):
        style_feats, _ = self.get_all_feature(style_img)
        multi_level_feats = [
            style_feats["decoder"][1],
            style_feats["decoder"][2],
            style_feats["decoder"][3],
            style_feats["decoder"][4],
        ]  # 64, 128, 256, 512

        (
            local_mean_code,
            local_std_code,
            global_mean_code,
            global_std_code,
        ) = self.style_extractor(multi_level_feats)

        return (
            style_feats,
            local_mean_code,
            local_std_code,
            global_mean_code,
            global_std_code,
        )

    def transfer(self, content, style_list):
        # Style extraction
        style_feats_list = []
        local_mean_code_list = []  # N_STYLES, N_LEVEL (1, 2, 3, 4)
        local_std_code_list = []  # N_STYLES, N_LEVEL (1, 2, 3, 4)
        global_mean_code_list = []  # N_STYLES, N_LEVEL (1, 2, 3, 4)
        global_std_code_list = []  # N_STYLES, N_LEVEL (1, 2, 3, 4)
        for style in style_list:
            (
                style_feats,
                local_mean_code,
                local_std_code,
                global_mean_code,
                global_std_code,
            ) = self.extract_style_code(
                style
            )  # not deterministic

            style_feats_list.append(style_feats)
            local_mean_code_list.append(local_mean_code)
            local_std_code_list.append(local_std_code)
            global_mean_code_list.append(global_mean_code)
            global_std_code_list.append(global_std_code)

        # Feature extraction
        content_feat, content_skips = content, {}
        content_feats = {"encoder": {}}
        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            content_feats["encoder"][level] = content_feat

        # Align local and global code
        corr_mat_list = []
        similarity_map_list = []
        for style_feats in style_feats_list:
            corr_mat, sim_map = self.correlation_block(content_feats, style_feats)
            corr_mat_list.append(corr_mat)
            similarity_map_list.append(sim_map)

        for i, corr_mat in enumerate(corr_mat_list):
            # No need to align global code
            local_mean_code_list[i] = self.alignment_block(
                corr_mat, local_mean_code_list[i]
            )
            local_std_code_list[i] = self.alignment_block(
                corr_mat, local_std_code_list[i]
            )

        # Global local feature fusion
        transformation_levels = [1, 2]
        mean_fused_codes = {}
        std_fused_codes = {}
        for style_idx in range(len(style_feats_list)):  # Fix wrong len
            # fuse per style
            (
                current_style_mean_fused_codes,
                current_styl_std_fused_codes,
            ) = self.local_global_fusion_block(
                local_mean_code_list[style_idx],
                local_std_code_list[style_idx],
                global_mean_code_list[style_idx],
                global_std_code_list[style_idx],
                transformation_levels,
            )
            mean_fused_codes[style_idx] = current_style_mean_fused_codes
            std_fused_codes[style_idx] = current_styl_std_fused_codes

        out_list = [content_feat] * len(style_list)
        # Transformation and fusion for each level with multi_level_fusion
        intermediate_feats_dict = {1: []}  # for fusion
        for level in [4, 3, 2, 1]:
            # Then fuse it
            for style_idx in range(len(style_feats_list)):
                if level in transformation_levels:
                    # Change this one whether to use global code, local code, or fusion
                    current_mean_code = mean_fused_codes[style_idx][
                        level
                    ]  # if global-> level-1, if fused level
                    current_std_code = std_fused_codes[style_idx][level]
                    out_list[style_idx] = self.transform_module(
                        out_list[style_idx], current_mean_code, current_std_code
                    )
                if level == 1:
                    intermediate_feats_dict[level].append(out_list[style_idx])
                out_list[style_idx] = self.decode(
                    out_list[style_idx], content_skips, level
                )

        # Merging using feature wise merging
        _, _, target_H, target_W = out_list[0].size()
        similarity_map_list = [
            F.interpolate(sim_map, size=(target_H, target_W), mode="nearest")
            for sim_map in similarity_map_list
        ]

        return out_list, intermediate_feats_dict, similarity_map_list


class MergingRefinementNetwork(nn.Module):
    def __init__(
        self,
        device,
        input_nc: int = 3,
        output_nc: int = 3,
        num_downs: int = 7,
        ngf: int = 64,
        use_dropout: bool = False,
    ):
        super(MergingRefinementNetwork, self).__init__()
        self.device = device

        self.refinement_net = UnetGenerator(
            input_nc,
            output_nc,
            num_downs,
            ngf,
            nn.InstanceNorm2d,
            use_dropout=use_dropout,
        )  # full model

        self.feat_region_wise_merging = FeatureRegionWiseMerging()

        init_net(self.refinement_net)  # remove if concatenated version
        init_net(self.feat_region_wise_merging)

    def forward(
        self,
        intermediate_feats_dict,
        similarity_map_list,
        return_attention_weight=False,
    ):
        merging_out, attention_weight = self.feat_region_wise_merging(
            intermediate_feats_dict, similarity_map_list
        )
        # full model
        out = self.refinement_net(merging_out)

        if return_attention_weight:
            return out, attention_weight
        else:
            return out
