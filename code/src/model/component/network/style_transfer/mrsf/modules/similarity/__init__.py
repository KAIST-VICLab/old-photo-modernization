import torch

from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()

        model = []
        model += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=True),
            nn.ReLU(inplace=True),
        ]
        model += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, 1, 0, bias=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out + x


class PostCorrelationBlock(nn.Module):
    def __init__(self, in_dim: int, inter_dim: int):
        super(PostCorrelationBlock, self).__init__()

        model = []
        model += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_dim, inter_dim, 3, 1, 0, bias=True),
        ]
        model += [ResidualBlock(inter_dim)]
        model += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(inter_dim, in_dim, 3, 1, 0, bias=True),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out


class SimilarityMapBlock(nn.Module):
    # cross attention
    def __init__(self, in_channels: int, inter_channels: int):
        super(SimilarityMapBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.theta = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.phi = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.post_corr_block = PostCorrelationBlock(1, 32)

    def forward(self, query_feat: torch.Tensor, key_feat: torch.Tensor):
        batch_size, _, feat_height, feat_width = query_feat.size()

        # get spatial correlation
        theta_x = self.theta(query_feat).view(
            batch_size, self.inter_channels, -1
        )  # N x C x HW
        phi_x = self.phi(key_feat).view(
            batch_size, self.inter_channels, -1
        )  # N x C x HW
        theta_x = theta_x.permute(0, 2, 1)  # N x HW x C

        f = torch.matmul(theta_x, phi_x)  # N X HW X HW

        similarity_map = torch.unsqueeze(f, dim=1)  # N X 1 X HW X HW
        similarity_map = torch.max(similarity_map, dim=-1, keepdim=True)[
            0
        ]  # N X 1 X HW X 1
        similarity_map = similarity_map.view(
            batch_size, 1, feat_height, feat_width
        )  # N X 1 X H X W

        similarity_map = self.post_corr_block(similarity_map)

        # f_div_C = F.softmax((f / self.temperature), dim=-1)
        # N x HW x HW == correlation matrix to warp key to query
        return similarity_map


if __name__ == "__main__":
    sim_block = SimilarityMapBlock(512, 512)
    content_feat = torch.randn((1, 512, 64, 64))
    style_feat = torch.randn((1, 512, 64, 64))
    sim_map = sim_block(content_feat, style_feat)
    print(sim_map)
    print(sim_map.size())
