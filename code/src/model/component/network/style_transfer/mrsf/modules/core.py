import torch
import torch.nn.functional as F

from torch import nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class ResBlockV2(torch.nn.Module):
    def __init__(self, dim):
        super(ResBlockV2, self).__init__()

        model = [
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim, dim, 3, 1, 1),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class StyleExtractorV2(torch.nn.Module):
    # Local: Pass Check, Global: ~ (Likely Pass Check)
    def __init__(self, eps=1.0e-5):
        super(StyleExtractorV2, self).__init__()

        self.epsilon = eps
        # nhiddens = 128
        # n_seg_con = 128

        # Features from decoder
        self.pre_feat_l1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )
        self.pre_feat_l2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )
        self.pre_feat_l3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )
        self.pre_feat_l4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )

        # Gate conv for shared block since we want to mask the features for each level
        # depending on the condition from semantic segmentation

        self.mean_weight_l1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 1, 1, 0),
        )
        self.std_weight_l1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 0), nn.LeakyReLU(0.1), nn.Conv2d(32, 64, 1, 1, 0)
        )
        self.mean_weight_l2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 1, 1, 0),
        )
        self.std_weight_l2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0), nn.LeakyReLU(0.1), nn.Conv2d(64, 128, 1, 1, 0)
        )
        self.mean_weight_l3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 1, 1, 0),
        )
        self.std_weight_l3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 1, 1, 0),
        )
        self.mean_weight_l4 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 1, 1, 0),
        )
        self.std_weight_l4 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 1, 1, 0),
        )

    def _compute_local_mean_std(self, feat, k_size):
        BS, C, H, W = feat.size()
        unfolded_feat = F.unfold(feat, kernel_size=k_size, stride=1)
        unfolded_feat = unfolded_feat.reshape(BS, C, k_size * k_size, -1)
        local_var = unfolded_feat.var(dim=2)
        local_var = local_var.reshape(
            BS, C, H - ((k_size // 2) * 2), W - ((k_size // 2) * 2)
        )
        local_mean = F.avg_pool2d(feat, kernel_size=k_size, stride=1)
        local_var = local_var + self.epsilon
        local_std = local_var.sqrt()
        return local_mean, local_std

    def forward(self, multi_level_feats):
        # Preprocessing
        ori_feat_l1, ori_feat_l2, ori_feat_l3, ori_feat_l4 = multi_level_feats

        # Feature extraction
        feat_l1 = self.pre_feat_l1(ori_feat_l1)
        feat_l2 = self.pre_feat_l2(ori_feat_l2)
        feat_l3 = self.pre_feat_l3(ori_feat_l3)
        feat_l4 = self.pre_feat_l4(ori_feat_l4)

        # Local mean and std computation
        k_size = 3
        padding = k_size // 2
        padding = (padding, padding, padding, padding)

        feat_l1 = F.pad(feat_l1, pad=padding, mode="reflect")
        feat_l2 = F.pad(feat_l2, pad=padding, mode="reflect")
        feat_l3 = F.pad(feat_l3, pad=padding, mode="reflect")
        feat_l4 = F.pad(feat_l4, pad=padding, mode="reflect")

        local_mean_l1, local_std_l1 = self._compute_local_mean_std(
            feat_l1, k_size=k_size
        )
        local_mean_l2, local_std_l2 = self._compute_local_mean_std(
            feat_l2, k_size=k_size
        )
        local_mean_l3, local_std_l3 = self._compute_local_mean_std(
            feat_l3, k_size=k_size
        )
        local_mean_l4, local_std_l4 = self._compute_local_mean_std(
            feat_l4, k_size=k_size
        )

        # Mean and std post processing
        local_mean_l1 = self.mean_weight_l1(local_mean_l1)
        local_std_l1 = self.std_weight_l1(local_std_l1)
        local_mean_l2 = self.mean_weight_l2(local_mean_l2)
        local_std_l2 = self.std_weight_l2(local_std_l2)
        local_mean_l3 = self.mean_weight_l3(local_mean_l3)
        local_std_l3 = self.std_weight_l3(local_std_l3)
        local_mean_l4 = self.mean_weight_l4(local_mean_l4)
        local_std_l4 = self.std_weight_l4(local_std_l4)

        global_mean_l1, global_std_l1 = self._compute_global_mean_std(ori_feat_l1)
        global_mean_l2, global_std_l2 = self._compute_global_mean_std(ori_feat_l2)
        global_mean_l3, global_std_l3 = self._compute_global_mean_std(ori_feat_l3)
        global_mean_l4, global_std_l4 = self._compute_global_mean_std(ori_feat_l4)

        # Repeat for global code
        l1_spatial_size = local_mean_l1.size()[2:]
        l2_spatial_size = local_mean_l2.size()[2:]
        l3_spatial_size = local_mean_l3.size()[2:]
        l4_spatial_size = local_mean_l4.size()[2:]

        global_mean_l1 = global_mean_l1.repeat(1, 1, *l1_spatial_size)
        global_std_l1 = global_std_l1.repeat(1, 1, *l1_spatial_size)
        global_mean_l2 = global_mean_l2.repeat(1, 1, *l2_spatial_size)
        global_std_l2 = global_std_l2.repeat(1, 1, *l2_spatial_size)
        global_mean_l3 = global_mean_l3.repeat(1, 1, *l3_spatial_size)
        global_std_l3 = global_std_l3.repeat(1, 1, *l3_spatial_size)
        global_mean_l4 = global_mean_l4.repeat(1, 1, *l4_spatial_size)
        global_std_l4 = global_std_l4.repeat(1, 1, *l4_spatial_size)

        local_mean_code = [local_mean_l1, local_mean_l2, local_mean_l3, local_mean_l4]
        local_std_code = [local_std_l1, local_std_l2, local_std_l3, local_std_l4]
        global_mean_code = [
            global_mean_l1,
            global_mean_l2,
            global_mean_l3,
            global_mean_l4,
        ]
        global_std_code = [global_std_l1, global_std_l2, global_std_l3, global_std_l4]

        return local_mean_code, local_std_code, global_mean_code, global_std_code

    def forward_single(self, feat, level):
        ori_feat = feat

        pre_feat_layer = getattr(self, "pre_feat_l{}".format(level))
        feat = pre_feat_layer(feat)

        k_size = 3
        padding = k_size // 2
        padding = (padding, padding, padding, padding)

        feat = F.pad(feat, pad=padding, mode="reflect")

        local_mean, local_std = self._compute_local_mean_std(feat, k_size=k_size)

        local_mean_layer = getattr(self, "mean_weight_l{}".format(level))
        local_std_layer = getattr(self, "std_weight_l{}".format(level))

        local_mean_code = local_mean_layer(local_mean)
        local_std_code = local_std_layer(local_std)

        global_mean_code, global_std_code = self._compute_global_mean_std(ori_feat)
        return local_mean_code, local_std_code, global_mean_code, global_std_code

    def _compute_global_mean_std(self, feat):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + self.epsilon
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


class SimpleSingleTransformModule(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super(SimpleSingleTransformModule, self).__init__()
        self.epsilon = eps

    def _compute_global_mean_std(self, feat):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + self.epsilon
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, content_feat, style_mean_code, style_std_code):
        content_mean, content_std = self._compute_global_mean_std(content_feat)
        out = (content_feat - content_mean) / (
            content_std + self.epsilon
        )  # BUG: every level you do the transformation which makes compounding results
        # We can create combination of parallel and sequential transformation
        out = out * style_std_code + style_mean_code
        return out


class LocalGlobalFusionBlock(nn.Module):
    def __init__(self):
        super(LocalGlobalFusionBlock, self).__init__()

        self.l1_mean_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.l1_std_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.l2_mean_fusion = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, 1, 1),
        )
        self.l2_std_fusion = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, 1, 1),
        )
        self.l3_mean_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1),
        )
        self.l3_std_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1),
        )
        self.l4_mean_fusion = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, 1, 1),
        )
        self.l4_std_fusion = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, 1, 1),
        )

    def forward(
        self,
        local_mean_code_list,
        local_std_code_list,
        global_mean_code_list,
        global_std_code_list,
        transformation_levels,
    ):
        mean_fused_codes = {}
        std_fused_codes = {}
        for level in transformation_levels:
            mean_fusion_layer = getattr(self, "l{}_mean_fusion".format(level))
            std_fusion_layer = getattr(self, "l{}_std_fusion".format(level))

            current_global_mean_code = global_mean_code_list[level - 1]
            current_global_std_code = global_std_code_list[level - 1]
            current_local_mean_code = local_mean_code_list[level - 1]
            current_local_std_code = local_std_code_list[level - 1]

            current_mean_code = torch.cat(
                [current_global_mean_code, current_local_mean_code], dim=1
            )
            current_std_code = torch.cat(
                [current_global_std_code, current_local_std_code], dim=1
            )

            fused_mean_code = mean_fusion_layer(current_mean_code)
            fused_std_code = std_fusion_layer(current_std_code)

            mean_fused_codes[level] = fused_mean_code
            std_fused_codes[level] = fused_std_code

        return mean_fused_codes, std_fused_codes


class FeatureRegionWiseMerging(nn.Module):
    def __init__(self):
        super(FeatureRegionWiseMerging, self).__init__()

        self.spatial_attn = nn.Conv2d(2, 1, 3, 1, 1)  # Full Model
        # remove for baseline

        self.output_block = nn.Sequential(  # full Model
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),  # remove this
            nn.Conv2d(64, 3, 3, 1, 1),  # remove this
        )

    def forward(self, intermediate_feats_list, similarity_map_list):
        # Preprocessing
        # Block gradient to flow for alternating optimization
        similarity_map_list = [sim_map.detach() for sim_map in similarity_map_list]
        for level, feats in intermediate_feats_list.items():
            new_feats = [feat.detach() for feat in feats]
            intermediate_feats_list[level] = new_feats

        # similarity_map_list: N_STYLES, 2, 256, 256
        # intermediate_feats_list: N_LEVELS, N_STYLES, ...
        # full model

        spatial_attention_weight = [
            self.spatial_attn(sim_map) for sim_map in similarity_map_list
        ]
        spatial_attention_weight = torch.stack(spatial_attention_weight)

        feats = torch.stack(intermediate_feats_list[1])

        # Merge
        # full model
        multi_style_feats = feats * F.softmax(spatial_attention_weight, dim=0)
        multi_style_feats = torch.sum(multi_style_feats, dim=0)
        merging_out = self.output_block(multi_style_feats)  # for feature

        return merging_out, F.softmax(spatial_attention_weight / 0.5, dim=0)


class CorrelationBlock(torch.nn.Module):
    def __init__(self):
        super(CorrelationBlock, self).__init__()

        self.cond_net_channel = 128
        self.feature_channel = 64
        self.n_out = 256

        self.layer_l1 = nn.Sequential(
            nn.Conv2d(64, self.feature_channel, 3, 1, 1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.feature_channel, self.feature_channel, 3, 1, 1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.LeakyReLU(0.1),
        )
        self.layer_l2 = nn.Sequential(
            nn.Conv2d(128, self.feature_channel, 3, 1, 1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.feature_channel, self.feature_channel, 3, 1, 1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.LeakyReLU(0.1),
        )
        self.layer_l3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, self.feature_channel, 3, 1, 1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.LeakyReLU(0.1),
        )
        self.layer_l4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, self.feature_channel, 3, 1, 1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.LeakyReLU(0.1),
        )

        self.shared_space_block = nn.Sequential(
            nn.Conv2d(self.n_out, self.n_out, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.n_out, self.n_out, 3, 1, 1),
            nn.InstanceNorm2d(self.n_out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.n_out, self.n_out, 3, 1, 1),
            nn.InstanceNorm2d(self.n_out),
            nn.LeakyReLU(),  # using resblock here will makes the network fail to allign correctly
        )

        # Cross attention
        self.in_channels = self.n_out
        self.inter_channels = 256

        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, 1, 1, 0)

    def forward(self, multi_level_content_feats, multi_level_style_feats):
        _, _, target_con_H, target_con_W = multi_level_content_feats["encoder"][
            4
        ].size()
        cont_feats_4 = multi_level_content_feats["encoder"][4]
        cont_feats_3 = F.interpolate(
            multi_level_content_feats["encoder"][3],
            size=(target_con_H, target_con_W),
            mode="nearest",
        )
        cont_feats_2 = F.interpolate(
            multi_level_content_feats["encoder"][2],
            size=(target_con_H, target_con_W),
            mode="nearest",
        )
        cont_feats_1 = F.interpolate(
            multi_level_content_feats["encoder"][1],
            size=(target_con_H, target_con_W),
            mode="nearest",
        )

        cont_feats_4 = self.layer_l4(cont_feats_4)
        cont_feats_3 = self.layer_l3(cont_feats_3)
        cont_feats_2 = self.layer_l2(cont_feats_2)
        cont_feats_1 = self.layer_l1(cont_feats_1)
        cont_feats = torch.cat(
            [cont_feats_4, cont_feats_3, cont_feats_2, cont_feats_1], dim=1
        )

        _, _, target_sty_H, target_sty_W = multi_level_style_feats["encoder"][4].size()
        styl_feats_4 = multi_level_style_feats["encoder"][4]
        styl_feats_3 = F.interpolate(
            multi_level_style_feats["encoder"][3],
            size=(target_sty_H, target_sty_W),
            mode="nearest",
        )
        styl_feats_2 = F.interpolate(
            multi_level_style_feats["encoder"][2],
            size=(target_sty_H, target_sty_W),
            mode="nearest",
        )
        styl_feats_1 = F.interpolate(
            multi_level_style_feats["encoder"][1],
            size=(target_sty_H, target_sty_W),
            mode="nearest",
        )

        styl_feats_4 = self.layer_l4(styl_feats_4)
        styl_feats_3 = self.layer_l3(styl_feats_3)
        styl_feats_2 = self.layer_l2(styl_feats_2)
        styl_feats_1 = self.layer_l1(styl_feats_1)
        styl_feats = torch.cat(
            [styl_feats_4, styl_feats_3, styl_feats_2, styl_feats_1], dim=1
        )

        cont_feats = self.shared_space_block(cont_feats)
        styl_feats = self.shared_space_block(styl_feats)

        BS, _, target_H, target_W = cont_feats.size()

        theta = self.theta(cont_feats).view(BS, self.inter_channels, -1)  # N C HW
        theta_permute = theta.permute(0, 2, 1)  # N HW C
        phi = self.phi(styl_feats).view(BS, self.inter_channels, -1)  # N C HW

        f = torch.matmul(theta_permute, phi)  # N HW HW
        corr_mat = F.softmax(f, dim=-1)  # N HW HW -> N 1024 1024

        # Max and Avg as similarity map (look CBAM) -> then use this to help the selection of multiple images
        max_similarity_map = torch.max(f, dim=-1, keepdim=True)[0]
        mean_similarity_map = torch.mean(f, dim=-1, keepdim=True)

        max_similarity_map = max_similarity_map.view(BS, 1, target_H, target_W)
        mean_similarity_map = mean_similarity_map.view(BS, 1, target_H, target_W)

        similarity_map = torch.cat([max_similarity_map, mean_similarity_map], dim=1)
        return corr_mat, similarity_map


class AlignmentBlock(nn.Module):
    def __init__(self):
        super(AlignmentBlock, self).__init__()
        self.post_l1 = nn.Sequential(
            ResBlockV2(64),
            ResBlockV2(64),
            ResBlockV2(64),
        )
        self.post_l2 = nn.Sequential(
            ResBlockV2(128),
            ResBlockV2(128),
        )
        self.post_l3 = nn.Sequential(ResBlockV2(256))
        self.post_l4 = nn.Identity()

    def forward(self, correlation_matrix, multi_level_code):
        BS, _, target_H, target_W = multi_level_code[3].size()

        code_l1 = F.interpolate(
            multi_level_code[0], size=(target_H, target_W), mode="nearest"
        )
        code_l2 = F.interpolate(
            multi_level_code[1], size=(target_H, target_W), mode="nearest"
        )
        code_l3 = F.interpolate(
            multi_level_code[2], size=(target_H, target_W), mode="nearest"
        )
        code_l4 = F.interpolate(
            multi_level_code[3], size=(target_H, target_W), mode="nearest"
        )

        code_l1 = code_l1.view(BS, 64, -1).permute(0, 2, 1)  # B HW C
        code_l2 = code_l2.view(BS, 128, -1).permute(0, 2, 1)  # B HW C
        code_l3 = code_l3.view(BS, 256, -1).permute(0, 2, 1)  # B HW C
        code_l4 = code_l4.view(BS, 512, -1).permute(0, 2, 1)  # B HW C

        aligned_code_l1 = (
            torch.matmul(correlation_matrix, code_l1).permute(0, 2, 1).contiguous()
        )  # B C HW
        aligned_code_l2 = (
            torch.matmul(correlation_matrix, code_l2).permute(0, 2, 1).contiguous()
        )  # B C HW
        aligned_code_l3 = (
            torch.matmul(correlation_matrix, code_l3).permute(0, 2, 1).contiguous()
        )  # B C HW
        aligned_code_l4 = (
            torch.matmul(correlation_matrix, code_l4).permute(0, 2, 1).contiguous()
        )  # B C HW

        aligned_code_l1 = aligned_code_l1.view(BS, 64, target_H, target_W)
        aligned_code_l2 = aligned_code_l2.view(BS, 128, target_H, target_W)
        aligned_code_l3 = aligned_code_l3.view(BS, 256, target_H, target_W)
        aligned_code_l4 = aligned_code_l4.view(BS, 512, target_H, target_W)

        aligned_code_l1 = F.interpolate(
            aligned_code_l1, size=multi_level_code[0].size()[2:], mode="nearest"
        )
        aligned_code_l2 = F.interpolate(
            aligned_code_l2, size=multi_level_code[1].size()[2:], mode="nearest"
        )
        aligned_code_l3 = F.interpolate(
            aligned_code_l3, size=multi_level_code[2].size()[2:], mode="nearest"
        )
        aligned_code_l4 = F.interpolate(
            aligned_code_l4, size=multi_level_code[3].size()[2:], mode="nearest"
        )

        aligned_code_l1 = self.post_l1(aligned_code_l1)
        aligned_code_l2 = self.post_l2(aligned_code_l2)
        aligned_code_l3 = self.post_l3(aligned_code_l3)
        aligned_code_l4 = self.post_l4(aligned_code_l4)

        return [aligned_code_l1, aligned_code_l2, aligned_code_l3, aligned_code_l4]


if __name__ == "__main__":
    con_tensor = torch.randn((1, 64, 128, 128), requires_grad=True, device="cuda")
    sty_tensor = torch.randn((1, 64, 128, 128), requires_grad=True, device="cuda")

    con_tensor = torch.randn((1, 64, 128, 128), requires_grad=True, device="cuda")
    sty_tensor = torch.randn((1, 64, 128, 128), requires_grad=True, device="cuda")
    con_mask = torch.randn((1, 1, 128, 128), requires_grad=True, device="cuda")
    sty_mask = torch.randn((1, 1, 128, 128), requires_grad=True, device="cuda")
