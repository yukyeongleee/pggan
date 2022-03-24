import math

import torch
import torch.nn as nn

from numpy import prod

def getLayerNormalizationFactor(x):
    """
    Get per-layer normalization constant from He’s initializer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])

    return math.sqrt(2.0 / fan_in)

class ConstrainedLayer(nn.Module):
    """
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 init_bias_to_zero=True):
        """
        equalized (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        init_bias_to_zero (bool): if true, bias will be initialized to zero
        """

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if init_bias_to_zero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):

        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedConv2d(ConstrainedLayer):

    def __init__(self,
                 in_channels,
                 out_channels, 
                 kernel_size,
                 padding=0,
                 bias=True,
                 **kwargs):
        """
        A nn.Conv2d module with specific constraints
            - Shape of nn.Conv2d.weight: (out_channels, in_channels, kernel_size[0], kernel_size[1])
        """

        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(in_channels, out_channels,
                                            kernel_size, padding=padding,
                                            bias=bias),
                                  **kwargs)


class EqualizedLinear(ConstrainedLayer):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 **kwargs):
        """
        A nn.Linear module with specific constraints
            - Shape of nn.Linear.weight: (out_features, in_features)
        """

        ConstrainedLayer.__init__(self,
                                  nn.Linear(in_features, out_features,
                                  bias=bias), **kwargs)