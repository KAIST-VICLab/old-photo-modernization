from torch import nn


class ResBlockV2(nn.Module):
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


class BasicResidualBlock(nn.Module):
    """
    Define a basic residual block from ResNet
    Source: https://github.com/taesungp/contrastive-unpaired-translation
    """

    def __init__(self, dim, padding_type, norm_layer, act_type, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(BasicResidualBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, act_type, use_dropout, use_bias
        )

    def build_conv_block(
        self, dim, padding_type, norm_layer, act_type, use_dropout, use_bias
    ):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer,
            and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        if act_type == "relu":
            conv_block += [nn.ReLU(True)]
        elif act_type == "lrelu":
            conv_block += [nn.LeakyReLU(inplace=True)]
        elif act_type == "prelu":
            conv_block += [nn.PReLU()]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
