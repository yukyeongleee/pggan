import torch
import torch.nn as nn

from lib.utils import set_norm_layer, set_activate_layer, AdaIN, upscale2d
from lib.layers import PixelwiseVectorNorm, EqualizedConv2d, EqualizedLinear

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode="bilinear"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

#------------------------------------------------------------------------------------------
# ConvBlock
#   1. Upsample / Conv(padding)
#       - padding options : 'zeros'(default), 'reflect', 'replicate' or 'circular'
#       - if you choose upsample option, you have to set stride==1
#   2. Norm
#       - Norm options : 'bn', 'in', 'none'
#   3. activation
#       - activation options : 'relu', 'tanh', 'sig', 'none'
#------------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size=3, stride=2, padding=1, \
        norm_type='bn', activation_type='relu', transpose=False):
        super(ConvBlock, self).__init__()

        if transpose:
            self.up = Interpolate(scale_factor=stride)
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=padding)
        else:
            self.up = transpose
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=padding)
        
        self.norm = set_norm_layer(norm_type, output_dim) # bn, in, none
        self.activation = set_activate_layer(activation_type) # relu, lrelu, tanh, sig, none

    def forward(self, x):
        if self.up:
            x = self.up(x)

        x = self.conv(x)

        if self.norm:
            x = self.norm(x)

        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=1, norm='in', activation='lrelu'):
        super(ResBlock, self).__init__()

        self.norm1 = set_norm_layer(norm, out_c)
        self.norm2 = set_norm_layer(norm, out_c)
        self.activ = set_activate_layer(activation)
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.resize = Interpolate(scale_factor=scale_factor)

    def forward(self, feat):
        feat1 = self.norm1(feat)
        feat1 = self.activ(feat1)
        feat1 = self.conv1(feat1)
        feat1 = self.resize(feat1)
        feat1 = self.norm2(feat1)
        feat1 = self.activ(feat1)
        feat1 = self.conv2(feat1)

        feat2 = self.conv1x1(feat)
        feat2 = self.resize(feat2)

        return feat1 + feat2


class AdaINResBlock(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=1, activation='lrelu', style_dim=512):
        super(AdaINResBlock, self).__init__()

        self.AdaIN1 = AdaIN(style_dim, in_c)
        self.AdaIN2 = AdaIN(style_dim, out_c)
        self.activ = set_activate_layer(activation)
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.resize = Interpolate(scale_factor=scale_factor)

    def forward(self, feat, v_sid):
        feat1 = self.AdaIN1(feat, v_sid)
        feat1 = self.activ(feat1)
        feat1 = self.conv1(feat1)
        feat1 = self.resize(feat1)
        feat1 = self.AdaIN2(feat1, v_sid)
        feat1 = self.activ(feat1)
        feat1 = self.conv2(feat1)

        # skip connction
        feat2 = self.conv1x1(feat) # chnnel dim
        feat2 = self.resize(feat2) # size 

        return feat1 + feat2

class ProgressiveGeneratorBlock(nn.Module):

    def __init__(self, prev_depth, new_depth, equalizedlR=True, initBiasToZero=True, apply_norm=True, is_first=False):
        super(ProgressiveGeneratorBlock, self).__init__()
        
        self.conv_list = []
        if not is_first:
            self.conv_list.append(EqualizedConv2d(prev_depth, 
                                                  new_depth,
                                                  3,
                                                  padding=1,
                                                  equalized=equalizedlR,
                                                  initBiasToZero=initBiasToZero)))
        self.conv_list.append(EqualizedConv2d(new_depth, 
                                              new_depth,
                                              3,
                                              padding=1,
                                              equalized=equalizedlR,
                                              initBiasToZero=initBiasToZero)))

        self.activ = set_activate_layer('lrelu')

        self.apply_norm = apply_norm
        if self.apply_norm:
            self.norm = PixelwiseVectorNorm()
        
        self.is_first = is_first

    def forward(self, x):
        
        if not is_first:
            x = upscale2d(x)

        for conv in self.conv_list:
            x = self.activ(conv(x))
            if self.apply_norm:
                x = norm(x)

        return x
            
class toRGBBlock(nn.Module):

    def __init__(self, new_depth, output_dim=3, equalizedlR=True, initBiasToZero=True)
        self.toRGB = EqualizedConv2d(new_depth,
                                     output_dim,
                                     1,
                                     equalized=self.equalizedlR,
                                     initBiasToZero=self.initBiasToZero))
    
    def forward(self, x, is_last=False):

        y = self.toRGB(x)
        if not is_last:
            y = upscale2d(y)

        return y

        
