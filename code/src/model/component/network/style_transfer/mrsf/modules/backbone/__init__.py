from collections import OrderedDict
from torch import nn

import torch
import torch.nn.functional as F


class PoolingLayer(nn.Module):
    def __init__(self, out_channels):
        super(PoolingLayer, self).__init__()
        self.kernel = self._generate_gaussian_kernel(out_channels=out_channels)

    def _generate_gaussian_kernel(self, out_channels=3):
        kernel = torch.Tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ]
        )

        kernel /= 256
        kernel = kernel.repeat(out_channels, 1, 1, 1)
        return kernel

    def _apply_gaussian(self, img):
        if self.kernel.device != img.device:
            self.kernel = self.kernel.to(img.device)
        _, c, _, _ = img.size()
        img = F.pad(img, (2, 2, 2, 2), mode="reflect")
        out = F.conv2d(img, self.kernel, groups=c)  # see conv2d pytorch documentatiion
        return out

    def forward(self, x):
        gaussian_img = self._apply_gaussian(x)
        LF_pooled_feat = F.interpolate(
            gaussian_img, scale_factor=0.5, mode="bilinear", align_corners=True
        )
        LF_feat = F.interpolate(
            LF_pooled_feat, scale_factor=2, mode="bilinear", align_corners=True
        )
        HF_feat = x - LF_feat
        return LF_pooled_feat, HF_feat


class UnpoolingLayer(nn.Module):
    def __init__(self):
        super(UnpoolingLayer, self).__init__()

    def forward(self, LF, HF_skip):
        LF_unpooled_feat = F.interpolate(LF, scale_factor=2)
        feat = LF_unpooled_feat + HF_skip
        return feat


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = PoolingLayer(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = PoolingLayer(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = PoolingLayer(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    def forward(self, x):
        skips = {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
        return x, skips

    def forward_with_intermediate(self, x):
        feats, skips = {}, {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            feats[level] = x
        return feats, skips

    def encode(self, x, skips, level):
        assert level in {1, 2, 3, 4}
        if level == 1:
            out = self.conv0(x)
            out = self.relu(self.conv1_1(self.pad(out)))
            return out

        elif level == 2:
            out = self.relu(self.conv1_2(self.pad(x)))

            LF_pooled_feat, HF = self.pool1(out)
            skips["pool1"] = HF
            out = self.relu(self.conv2_1(self.pad(LF_pooled_feat)))
            return out

        elif level == 3:
            out = self.relu(self.conv2_2(self.pad(x)))

            LF_pooled_feat, HF = self.pool2(out)
            skips["pool2"] = HF
            out = self.relu(self.conv3_1(self.pad(LF_pooled_feat)))
            return out

        else:
            out = self.relu(self.conv3_2(self.pad(x)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_4(self.pad(out)))

            LF_pooled_feat, HF = self.pool3(out)
            skips["pool3"] = HF
            out = self.relu(self.conv4_1(self.pad(LF_pooled_feat)))
            return out

    def load_vgg_weights(self, vgg_path):
        new_key_mapping = {
            "0": "conv0",
            "2": "conv1_1",
            "5": "conv1_2",
            "9": "conv2_1",
            "12": "conv2_2",
            "16": "conv3_1",
            "19": "conv3_2",
            "22": "conv3_3",
            "25": "conv3_4",
            "29": "conv4_1",
        }

        new_state_dict = OrderedDict()
        state_dict = torch.load(vgg_path)

        for key, value in state_dict.items():
            layer_num, weight_type = key.split(".")
            if layer_num in new_key_mapping:
                new_key = ".".join([new_key_mapping[layer_num], weight_type])
                new_state_dict[new_key] = value

        self.load_state_dict(new_state_dict, strict=False)  # WavePool cannot be loaded


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        self.unpool3 = UnpoolingLayer()
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.unpool2 = UnpoolingLayer()
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.unpool1 = UnpoolingLayer()
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)

    def forward(self, x, skips):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x

    def forward_with_modulated_masks(self, x, skips, modulated_masks, apply_mask_at=1):
        for level in [4, 3, 2, 1]:
            modulated_mask = modulated_masks[level]
            if level == 1:
                x = x
            else:
                x = x * modulated_mask  # mask feature
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            HF = skips["pool3"]
            out = self.unpool3(out, HF)
            out = self.relu(self.conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            HF = skips["pool2"]
            out = self.unpool2(out, HF)
            return self.relu(self.conv2_2(self.pad(out)))
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            HF = skips["pool1"]
            out = self.unpool1(out, HF)
            return self.relu(self.conv1_2(self.pad(out)))
        else:
            out = self.relu(self.conv1_1(self.pad(x)))
            out = self.conv0(out)
            return out

    def set_requires_grad(self, level, requires_grad=False):
        dict_named_params_per_level = {
            1: ["conv0", "conv1_1"],
            2: ["conv1_2", "conv2_1"],
            3: ["conv2_2", "conv3_1"],
            4: ["conv3_2", "conv3_3", "conv3_4", "conv4_1"],
        }
        for name, param in self.named_parameters():
            if name.split(".")[0] in dict_named_params_per_level[level]:
                param.requires_grad = requires_grad
