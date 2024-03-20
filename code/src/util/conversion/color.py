import torch
from kornia.color import lab


def kornia_lab2rgb(img_lab: torch.Tensor) -> torch.Tensor:
    # img_lab: N C H W
    # with normalized range in [-1, 1]
    # output: unnormalized RGB image tensor
    img_l = img_lab[:, [0], :, :]
    img_ab = img_lab[:, 1:, :, :]
    img_l = (img_l + 1.0) * 50.0
    img_ab = img_ab * 128
    img_rgb = torch.cat([img_l, img_ab], dim=1)
    img_rgb = lab.lab_to_rgb(img_rgb)
    return img_rgb


def kornia_rgb2lab(img_rgb: torch.Tensor) -> torch.Tensor:
    # img_rgb: C H W
    # with unnormalized range [0, 1]
    # output: normalized LAB image tensor
    img_lab = lab.rgb_to_lab(img_rgb)
    img_l = img_lab[[0], :, :] / 50.0 - 1.0  # range: [-1, 1]
    img_ab = img_lab[1:, :, :] / 128  # range: [-1, 1]
    img_lab = torch.cat([img_l, img_ab], dim=0)
    return img_lab
