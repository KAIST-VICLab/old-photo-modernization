import torch
from torch import nn
import sys
import torch.nn.functional as F
import logging

from ..network.common.vgg import VGG19PerceptualNetwork

log = logging.getLogger(__name__)


class HuberLoss(nn.Module):
    def __init__(self, delta=0.01):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def __call__(self, input, target):
        mask = torch.zeros_like(input)
        mann = torch.abs(input - target)
        eucl = 0.5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl * mask + self.delta * (mann - .5 * self.delta) * (1 - mask)
        loss = eucl * mask / self.delta + (mann - 0.5 * self.delta) * (1 - mask)
        return torch.sum(loss, dim=1, keepdim=False).mean()


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE loss for better classification result
    Source: https://github.com/birdortyedi/vcnet-blind-image-inpainting
    """

    def __init__(self, epsilon=1e-2):
        super(WeightedBCELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, out, target, weights=None):
        out = out.clamp(self.epsilon, 1 - self.epsilon)
        if (
            weights is not None
        ):  # weights = [1-unknown_pixel_ratio, unknown_pixel_ratio]
            assert len(weights) == 2
            loss = weights[1] * (target * torch.log(out)) + weights[0] * (
                (1 - target) * torch.log(1 - out)
            )
        else:
            loss = target * torch.log(out) + (1 - target) * torch.log(1 - out)
        return torch.neg(torch.mean(loss))


class EdgeAwareSmoothnessLoss(nn.Module):
    """
    Edge aware smoothness loss from optical flow estimation literature
    Source:
        - https://github.com/jianfenglihg/UnOpticalFlow
    """

    def __init__(self):
        super(EdgeAwareSmoothnessLoss, self).__init__()

    def gradients(self, img):
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        return dx, dy

    def forward(self, out, content):
        img_grad_x, img_grad_y = self.gradients(content)
        w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
        w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

        dx, dy = self.gradients(out)
        dx2, _ = self.gradients(dx)
        _, dy2 = self.gradients(dy)
        error = (w_x[:, :, :, 1:] * torch.abs(dx2)).mean((1, 2, 3)) + (
            w_y[:, :, 1:, :] * torch.abs(dy2)
        ).mean((1, 2, 3))
        return error / 2.0


def feature_normalize(feature_in):
    feature_in_norm = (
        torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    )
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm


class ContextualLoss_forward(nn.Module):
    """
    Forward contextual loss
    input is Al, Bl, channel = 1, range ~ [0, 255]
    Source: https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization
    """

    def __init__(self):
        super(ContextualLoss_forward, self).__init__()
        pass

    def forward(self, X_features, Y_features, h=0.1, feature_centering=True):
        """
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        """
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        # feature_size = X_features.shape[2]

        # to normalized feature vectors
        if feature_centering:
            X_features = X_features - Y_features.view(
                batch_size, feature_depth, -1
            ).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            Y_features = Y_features - Y_features.view(
                batch_size, feature_depth, -1
            ).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features).view(
            batch_size, feature_depth, -1
        )  # batch_size * feature_depth * feature_size^2
        Y_features = feature_normalize(Y_features).view(
            batch_size, feature_depth, -1
        )  # batch_size * feature_depth * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(
            0, 2, 1
        )  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(
            X_features_permute, Y_features
        )  # batch_size * feature_size^2 * feature_size^2
        d = d.clamp(min=0)  # min distance is 0, max distance is infinite

        # normalized distance: dij_bar
        # can be zero in our case since we use synthetic reference
        d_norm = d / (
            torch.min(d, dim=-1, keepdim=True)[0] + 1e-5
        )  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)
        if torch.any(torch.isnan(A_ij)):
            log.debug(torch.min(d), torch.max(d))
            log.debug(torch.min(d_norm), torch.max(d_norm))

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)
        loss = -torch.log(CX)
        return loss.mean()


class GlobalStyleLoss(nn.Module):
    def __init__(self, device, epsilon=1e-5):
        super(GlobalStyleLoss, self).__init__()
        self.epsilon = epsilon
        self.device = device
        self.mse_criterion = nn.MSELoss()

    def _calc_mean_std(self, feat):
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + self.epsilon
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, out_feats, style_feats, levels=[1, 2]):
        # use 1-2 since we only change that in style transformation
        loss = torch.zeros(1, device=self.device)
        for level in levels:
            out_mean, out_std = self._calc_mean_std(out_feats[level])
            style_mean, style_std = self._calc_mean_std(style_feats[level])
            loss += self.mse_criterion(out_mean, style_mean) + self.mse_criterion(
                out_std, style_std
            )
        return loss


class LocalStyleLoss(nn.Module):
    def __init__(self, device, epsilon=1e-5):
        super(LocalStyleLoss, self).__init__()
        self.epsilon = epsilon
        self.device = device
        self.mse_criterion = nn.MSELoss()

    def _compute_weighted_mean(self, masked_feat, mask):
        N, C, _, _ = masked_feat.size()
        mean = torch.sum(masked_feat, dim=(2, 3)) / (torch.sum(mask) + self.epsilon)
        mean = mean.view(N, C, 1, 1)
        return mean

    def _compute_weighted_std(self, masked_feat, mask):
        N, C, _, _ = masked_feat.size()
        _mean = self._compute_weighted_mean(masked_feat, mask)
        std = torch.sqrt(
            (
                torch.sum(torch.pow(masked_feat - _mean, 2) * mask, dim=(2, 3))
                / (torch.sum(mask) + self.epsilon)
            )
            + self.epsilon
        )
        std = std.view(N, C, 1, 1)
        return std

    def _calc_masked_mean_std(self, feat, mask):
        masked_feat = feat * mask

        mean = self._compute_weighted_mean(masked_feat, mask)
        std = self._compute_weighted_std(masked_feat, mask)

        return mean, std

    def forward(self, out_feats, style_feats, out_mask, style_mask, levels=[1, 2]):
        loss = torch.zeros(1, device=self.device)
        for level in levels:
            _, _, tar_H, tar_W = out_feats[level].size()
            resized_out_mask = F.interpolate(
                out_mask, size=(tar_H, tar_W), mode="nearest"
            )
            resized_style_mask = F.interpolate(
                style_mask, size=(tar_H, tar_W), mode="nearest"
            )

            out_mean, out_std = self._calc_masked_mean_std(
                out_feats[level], resized_out_mask
            )
            style_mean, style_std = self._calc_masked_mean_std(
                style_feats[level], resized_style_mask
            )
            loss += self.mse_criterion(out_mean, style_mean) + self.mse_criterion(
                out_std, style_std
            )
        return loss


def find_local_patch(x, patch_size):
    N, C, H, W = x.shape
    x_unfold = F.unfold(
        x,
        kernel_size=(patch_size, patch_size),
        padding=(patch_size // 2, patch_size // 2),
        stride=(1, 1),
    )

    return x_unfold.view(N, x_unfold.shape[1], H, W)


class WeightedAverage(nn.Module):
    """
    Weighted average loss to make the image smoother by using simple distance of RGB image # noqa:E501
    Source: https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization # noqa: E501
    """

    def __init__(
        self,
    ):
        super(WeightedAverage, self).__init__()

    def forward(self, x_rgb, patch_size=3, alpha=1, scale_factor=1):
        # alpha=0: less smooth; alpha=inf: smoother
        x_rgb = F.interpolate(x_rgb, scale_factor=scale_factor)
        r = x_rgb[:, 0:1, :, :]
        g = x_rgb[:, 1:2, :, :]
        b = x_rgb[:, 2:3, :, :]
        l_channel = 0.2126 * r + 0.7152 * g + 0.0722 * b

        local_r = find_local_patch(r, patch_size)
        local_g = find_local_patch(g, patch_size)
        local_b = find_local_patch(b, patch_size)
        local_l = 0.2126 * local_r + 0.7152 * local_g + 0.0722 * local_b
        local_difference_l = (local_l - l_channel) ** 2

        correlation = nn.functional.softmax(-1 * local_difference_l / alpha, dim=1)

        return torch.cat(
            (
                torch.sum(correlation * local_r, dim=1, keepdim=True),
                torch.sum(correlation * local_g, dim=1, keepdim=True),
                torch.sum(correlation * local_b, dim=1, keepdim=True),
            ),
            1,
        )


class NonlocalWeightedAverage(nn.Module):
    def __init__(
        self,
    ):
        """
        Nonlocal weighted average loss to make the image smoother
            by using simple distance of RGB image
        Source: https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization
        """
        super(NonlocalWeightedAverage, self).__init__()

    def forward(self, x_rgb, feature, patch_size=3, alpha=0.1, scale_factor=1):
        # alpha=0: less smooth; alpha=inf: smoother
        # input feature is normalized feature
        x_rgb = F.interpolate(x_rgb, scale_factor=scale_factor)
        batch_size, channel, height, width = x_rgb.shape
        feature = F.interpolate(feature, size=(height, width))
        batch_size = x_rgb.shape[0]
        x_rgb = x_rgb[:, :, :, :].view(batch_size, 3, -1)
        x_rgb = x_rgb.permute(0, 2, 1)

        local_feature = find_local_patch(feature, patch_size)
        local_feature = local_feature.view(batch_size, local_feature.shape[1], -1)

        correlation_matrix = torch.matmul(local_feature.permute(0, 2, 1), local_feature)
        correlation_matrix = nn.functional.softmax(correlation_matrix / alpha, dim=-1)

        weighted_rgb = torch.matmul(correlation_matrix, x_rgb)
        weighted_rgb = weighted_rgb.permute(0, 2, 1).contiguous()
        weighted_rgb = weighted_rgb.view(batch_size, 3, height, width)
        return weighted_rgb


class LocalStyleCodeLoss(nn.Module):
    """
    The loss encourage the network to produce
        accurate prediction of local style code (mean and std)
        by using the help of gorund truthg image and loss
    Idea by ours
    Result: not that good when I last used it
    """

    def __init__(self, device, epsilon=1e-5):
        super(LocalStyleCodeLoss, self).__init__()
        self.epsilon = epsilon
        self.device = device
        self.mse_criterion = nn.MSELoss()

    def _compute_weighted_mean(self, masked_feat, mask):
        N, C, _, _ = masked_feat.size()
        mean = torch.sum(masked_feat, dim=(2, 3)) / (torch.sum(mask) + self.epsilon)
        mean = mean.view(N, C, 1, 1)
        return mean

    def _compute_weighted_std(self, masked_feat, mask):
        N, C, _, _ = masked_feat.size()
        _mean = self._compute_weighted_mean(masked_feat, mask)
        std = torch.sqrt(
            (
                torch.sum(torch.pow(masked_feat - _mean, 2) * mask, dim=(2, 3))
                / (torch.sum(mask) + self.epsilon)
            )
            + self.epsilon
        )
        std = std.view(N, C, 1, 1)
        return std

    def _calc_masked_mean_std(self, feat, mask):
        masked_feat = feat * mask

        mean = self._compute_weighted_mean(masked_feat, mask)
        std = self._compute_weighted_std(masked_feat, mask)

        return mean, std

    def forward(self, out_feats, style_feats, out_mask, style_mask, levels=[1, 2]):
        loss = torch.zeros(1, device=self.device)
        for level in levels:
            _, _, tar_H, tar_W = out_feats[level].size()
            resized_out_mask = F.interpolate(
                out_mask, size=(tar_H, tar_W), mode="nearest"
            )
            resized_style_mask = F.interpolate(
                style_mask, size=(tar_H, tar_W), mode="nearest"
            )

            out_mean, out_std = self._calc_masked_mean_std(
                out_feats[level], resized_out_mask
            )
            style_mean, style_std = self._calc_masked_mean_std(
                style_feats[level], resized_style_mask
            )
            loss += self.mse_criterion(out_mean, style_mean) + self.mse_criterion(
                out_std, style_std
            )
        return loss


class GlobalStyleCodeLoss(nn.Module):
    """
    The loss encourage the global style code between
        the prediction and ground truth to be as similar as possible
    Source: Our idea
    """

    def __init__(self, device, epsilon=1e-5):
        super(GlobalStyleCodeLoss, self).__init__()
        self.epsilon = epsilon
        self.device = device
        self.mse_criterion = nn.MSELoss()

    def _calc_mean_std(self, feat):
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + self.epsilon
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, out_feats, style_feats, levels=[1, 2]):
        # use 1-2 since we only change that in style transformation
        loss = torch.zeros(1, device=self.device)
        for level in levels:
            out_mean, out_std = self._calc_mean_std(out_feats[level])
            style_mean, style_std = self._calc_mean_std(style_feats[level])
            loss += self.mse_criterion(out_mean, style_mean) + self.mse_criterion(
                out_std, style_std
            )
        return loss


class TVLoss(nn.Module):
    """
    Total variation loss to make the output image looks smoother
    Source: this loss is common
    """

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, tensor):
        w_variance = torch.mean(torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:]))
        h_variance = torch.mean(torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]))
        loss = w_variance + h_variance
        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective.
                                It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer(
            "real_label", torch.tensor(target_real_label, requires_grad=False)
        )
        self.register_buffer(
            "fake_label", torch.tensor(target_fake_label, requires_grad=False)
        )
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp", "nonsaturating", "hinge"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real) -> torch.Tensor:
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - tpyically the prediction from a discriminator
            target_is_real (bool) - if the ground truth label is for real images
                                    or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size
            of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def compute_loss(
        self, prediction: torch.Tensor, target_is_real, for_discriminator=True
    ):
        """
        Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - typically the prediction output from a discriminator
            target_is_real (bool) - if the ground truth label is for real images
                                    or fake images
        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == "nonsaturating":
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        elif self.gan_mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(
                        prediction - 1,
                        self.get_target_tensor(prediction, target_is_real=False),
                    )
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(
                        -prediction - 1,
                        self.get_target_tensor(prediction, target_is_real=False),
                    )
                    loss = -torch.mean(minval)
            else:
                loss = -torch.mean(prediction)

        return loss

    def __call__(
        self, prediction, target_is_real: bool, for_discriminator: bool = True
    ):
        if isinstance(prediction, list):
            loss = 0
            for pred in prediction:
                if isinstance(pred, list):
                    pred = pred[-1]
                loss_tensor = self.compute_loss(pred, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(prediction)
        else:
            return self.compute_loss(prediction, target_is_real, for_discriminator)


def cal_gradient_penalty(
    netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0
):
    """
    Calculate the gradient penalty loss, used in WGAN-GP paper
        https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device(...)
                                        if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not
                                        [real | fake | mixed].
        constant (float)            -- the constant used in formula
                                        ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if (
            type == "real"
        ):  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = (
                alpha.expand(
                    real_data.shape[0], real_data.nelement() // real_data.shape[0]
                )
                .contiguous()
                .view(*real_data.shape)
            )
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError("{} not implemented".format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolatesv,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (
            ((gradients + 1e-16).norm(2, dim=1) - constant) ** 2
        ).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device: torch.device):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = VGG19PerceptualNetwork().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        pred_feats, gt_feats = self.vgg(pred), self.vgg(gt)
        loss = 0
        for i in range(len(pred_feats)):
            loss += self.weights[i] * self.criterion(
                pred_feats[i], gt_feats[i].detach()
            )
        return loss


class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
