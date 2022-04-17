from asyncio import FastChildWatcher
from re import X
import torch
import torch.nn as nn

from lib.utils import downscale2d, set_norm_layer, set_activate_layer, AdaIN, upscale2d, num_flat_features
from lib.layers import EqualizedConv2d, EqualizedLinear

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

    def __init__(self, prev_depth, new_depth, equalized_lr=True, init_bias_to_zero=True, norm=None, is_first=False):
        super(ProgressiveGeneratorBlock, self).__init__()
        
        
        self.block = []
        if not is_first:
            self.block.append(EqualizedConv2d(prev_depth, 
                                                  new_depth,
                                                  3,
                                                  padding=1,
                                                  equalized=equalized_lr,
                                                  init_bias_to_zero=init_bias_to_zero))
            self.block.append(nn.LeakyReLU(0.2))
            if norm: 
                self.block.append(norm)

        self.block.append(EqualizedConv2d(new_depth, 
                                              new_depth,
                                              3,
                                              padding=1,
                                              equalized=equalized_lr,
                                              init_bias_to_zero=init_bias_to_zero))
        self.block.append(nn.LeakyReLU(0.2))
        if norm: 
            self.block.append(norm)

        self.block = nn.Sequential(*self.block)

        self.is_first = is_first

    def forward(self, x):
        
        if not self.is_first:
            x = upscale2d(x)
        x = self.block(x)

        return x
            
class toRGBBlock(nn.Module):

    def __init__(self, new_depth, output_dim=3, equalized_lr=True, init_bias_to_zero=True):
        super(toRGBBlock, self).__init__()
        
        self.toRGB = EqualizedConv2d(new_depth,
                                     output_dim,
                                     1,
                                     equalized=equalized_lr,
                                     init_bias_to_zero=init_bias_to_zero)
    
    def forward(self, x, apply_upscale=False):

        x = self.toRGB(x)
        if apply_upscale:
            x = upscale2d(x)

        return x


class ProgressiveDiscriminatorBlock(nn.Module):

    def __init__(self, new_depth, prev_depth, equalized_lr=True, init_bias_to_zero=True):
        super(ProgressiveDiscriminatorBlock, self).__init__()

        self.block = []
        self.block.append(EqualizedConv2d(new_depth,
                                              new_depth,
                                              3,
                                              padding=1,
                                              equalized=equalized_lr,
                                              init_bias_to_zero=init_bias_to_zero))
        self.block.append(nn.LeakyReLU(0.2))
        self.block.append(EqualizedConv2d(new_depth,
                                              prev_depth,
                                              3,
                                              padding=1,
                                              equalized=equalized_lr,
                                              init_bias_to_zero=init_bias_to_zero))
        self.block.append(nn.LeakyReLU(0.2))
        self.block.append(nn.AvgPool2d((2, 2)))

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        
        x = self.block(x)
        
        return x


def concatenate_stddev_channel(x, subgroup_size=4):
    r"""
    Add a minibatch standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) Get the mean, over all pixels and all channels of thsi ValueError
        3) expand the layer and cocatenate it with the input
    Args:
        - x (tensor): previous layer
        - subGroupSize (int): size of the mini-batches on which the standard deviation
        should be computed
    """
    size = x.size() # (N, Ch, W, H)
    subgroup_size = min(size[0], subgroup_size)
    if size[0] % subgroup_size != 0:
        subgroup_size = size[0]
    subgroup_num = int(size[0] / subgroup_size)
    if subgroup_size > 1:
        y = x.view(-1, subgroup_size, size[1], size[2], size[3]) 
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-8)
        y = y.view(subgroup_num, -1)
        y = torch.mean(y, 1).view(subgroup_num, 1)
        y = y.expand(subgroup_num, size[2]*size[3]).view((subgroup_num, 1, 1, size[2], size[3]))
        y = y.expand(subgroup_num, subgroup_size, -1, -1, -1) 
        y = y.contiguous().view((-1, 1, size[2], size[3]))
    else:
        y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    return torch.cat([x, y], dim=1) # (N, Ch+1, W, H)

        
class LastProgressiveDiscriminatorBlock(nn.Module):

    def __init__(self, depth, equalized_lr=True, init_bias_to_zero=True, apply_minibatch_norm=False):
        super(LastProgressiveDiscriminatorBlock, self).__init__()
        
        self.conv_list = []
        entry_dim = depth + 1 if apply_minibatch_norm else depth
        self.conv = EqualizedConv2d(entry_dim, 
                                    depth,
                                    3,
                                    padding=1,
                                    equalized=equalized_lr,
                                    init_bias_to_zero=init_bias_to_zero)
        self.linear = EqualizedLinear(depth * 16, 
                                        depth,
                                        equalized=equalized_lr,
                                        init_bias_to_zero=init_bias_to_zero)

        self.activ = nn.LeakyReLU(0.2)

        self.apply_minibatch_norm = apply_minibatch_norm

    def forward(self, x, subgroup_size=4):
        
        if self.apply_minibatch_norm:
            x = concatenate_stddev_channel(x, subgroup_size=subgroup_size)

        x = self.activ(self.conv(x))
        
        x = x.view(-1, num_flat_features(x))
        x = self.activ(self.linear(x))

        return x


class fromRGBBlock(nn.Module):

    #@# new_depth, input_dim=3 --> input_dim=3, new_depth=None
    def __init__(self, input_dim=3, new_depth=None, equalized_lr=True, init_bias_to_zero=True):
        super(fromRGBBlock, self).__init__()
        
        self.fromRGB = EqualizedConv2d(input_dim,
                                     new_depth,
                                     1,
                                     equalized=equalized_lr,
                                     init_bias_to_zero=init_bias_to_zero)
       
        self.activ = nn.LeakyReLU(0.2)
    
    def forward(self, x, apply_downscale=False):

        if apply_downscale:
            x = downscale2d(x)

        x = self.activ(self.fromRGB(x))

        return x
