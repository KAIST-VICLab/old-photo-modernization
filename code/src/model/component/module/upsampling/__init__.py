from torch import nn

from ..filter import get_filter
from ..padding import get_pad_layer
import numpy as np
import torch.nn.functional as F


class BasicUpsample(nn.Module):
    def __init__(self, channels, pad_type="repl", filt_size=4, stride=2):
        super(BasicUpsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(
            self.pad(inp),
            self.filt,
            stride=self.stride,
            padding=1 + self.pad_size,
            groups=inp.shape[1],
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]
