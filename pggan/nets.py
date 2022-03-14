import submodel.arcface as arcface
import torch
from torch import nn
import torch.nn.functional as F
from lib.utils import AdaIN
from lib.blocks import ProgressiveGeneratorBlock, toRGBBlock
from lib.layers import EqualizedLinear, PixelwiseVectorNorm

class Generator(nn.Module):
    
    def __init__(self,
                 latent_dim,
                 first_depth,
                 initBiasToZero=True,
                 LReLU_slope=0.2,
                 apply_norm=True,
                 generationActivation=None,
                 output_dim=3,
                 equalizedlR=True):
        r"""
        Build a generator for a progressive GAN model
        Args:
            - dimLatent (int): dimension of the latent vector
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - normalization (bool): normalize the input latent vector
            - generationActivation (function): activation function of the last
                                               layer (RGB layer). If None, then
                                               the identity is used
            - dimOutput (int): dimension of the output image. 3 -> RGB, 1 ->
                               grey levels
            - equalizedlR (bool): set to true to initiualize the layers with
                                  N(0,1) and apply He's constant at runtime
        """
        super(Generator, self).__init__()

        self.equalizedlR = equalizedlR
        self.initBiasToZero = initBiasToZero

        # Initalize the blocks
        self.block_depths = [first_depth]
        self.blocks = nn.ModuleList()
        self.toRGB_blocks = nn.ModuleList()

        # Initialize the block 0
        self.init_format_layer(latent_dim)
        self.output_dim = output_dim
        self.first_block = ProgressiveGeneratorBlock(first_depth, first_depth, 
                                                 equalized=self.equalizedlR,
                                                 initBiasToZero=self.initBiasToZero, 
                                                 apply_norm=True, is_first=True)

        self.toRGB_blocks.append(toRGBBlock(first_depth, 
                                            output_dim=self.output_dim, 
                                            equalizedlR=self.equalizedlR, 
                                            initBiasToZero=self.True))

        # self.block_0.append(EqualizedConv2d(depth_0, depth_0, 3,
        #                                         equalized=equalizedlR,
        #                                         initBiasToZero=initBiasToZero,
        #                                         padding=1))

        # self.toRGB_layers.append(EqualizedConv2d(depth_0, self.output_dim, 1,
        #                                         equalized=equalizedlR,
        #                                         initBiasToZero=initBiasToZero))

        # Initalize the upscaling parameters
        # alpha : when a new scale is added to the network, the previous
        # layer is smoothly merged with the output in the first stages of
        # the training
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(LReLu_slope)

        # normalization
        self.norm_layer = None
        if apply_norm:
            self.norm_layer = PixelwiseVectorNorm()

        # Last layer activation function
        self.generationActivation = generationActivation
        self.first_depth = first_depth


    def init_format_layer(self, latent_dim):
        r"""
        The format layer represents the first weights applied to the latent
        vector. It converts a 1xdimLatent input into a 4 x 4 xscalesDepth[0]
        layer.
        """

        self.latent_dim = latent_dim
        self.latent_format_layer = EqualizedLinear(self.latent_dim,
                                                    16 * self.scalesDepth[0],
                                                    equalized=self.equalizedlR,
                                                    initBiasToZero=self.initBiasToZero)

    def get_output_size(self):
        r"""
        Get the size of the generated image.
        """
        side = 4 * (2**(len(self.toRGBLayers) - 1))
        return (side, side)

    def add_block(self, new_depth):
        r"""
        Add a new scale to the model. Increasing the output resolution by
        a factor 2
        Args:
            - depthNewScale (int): depth of each conv layer of the new scale
        """
        prev_depth = self.block_depths[-1]

        self.block_depths.append(new_depth)
        self.blocks.append(ProgressiveGeneratorBlock(prev_depth, new_depth, 
                                                     equalized=self.equalizedlR,
                                                     initBiasToZero=self.initBiasToZero, 
                                                     apply_norm=True))
        self.toRGB_blocks.append(toRGBBlock(new_depth, 
                                            output_dim=self.output_dim, 
                                            equalizedlR=self.equalizedlR, 
                                            initBiasToZero=self.True, 
                                            is_last=False))

        # self.blocks.append(nn.ModuleList())

        # self.scaleLayers[-1].append(EqualizedConv2d(depthLastScale,
        #                                             depthNewScale,
        #                                             3,
        #                                             padding=1,
        #                                             equalized=self.equalizedlR,
        #                                             initBiasToZero=self.initBiasToZero))
        # self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale, depthNewScale,
        #                                             3, padding=1,
        #                                             equalized=self.equalizedlR,
        #                                             initBiasToZero=self.initBiasToZero))

        # self.toRGBLayers.append(EqualizedConv2d(depthNewScale,
        #                                         self.dimOutput,
        #                                         1,
        #                                         equalized=self.equalizedlR,
        #                                         initBiasToZero=self.initBiasToZero))

    def set_new_alpha(self, alpha):
        r"""
        Update the value of the merging factor alpha
        Args:
            - alpha (float): merging factor, must be in [0, 1]
        """
        
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def forward(self, x):

        ## Normalize the input ?
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = x.view(-1, num_flat_features(x)) ### CHECK
        # format layer
        x = self.leakyRelu(self.latent_format_layer(x))
        x = x.view(x.size()[0], -1, 4, 4)

        x = self.norm_layer(x)

        # Firts Block (no upsampling)
        x = self.first_block(x)
        # for convLayer in self.groupScale0:
        #     x = self.leakyRelu(convLayer(x))
        #     if self.normalizationLayer is not None:
        #         x = self.normalizationLayer(x)

        # To RGB 
        # If there are 2 blocks and blending is required (alpha > 0)
        if self.alpha > 0 and len(self.blocks) == 1:
            x_prev = self.toRGB_blocks[-2](x)

        # Upper scales
        for i, block in enumerate(self.blocks, 0):
            x = block(x)

            # To RGB
            # If there are more than 2 blocks blending is required (alpha > 0)
            if self.alpha > 0 and i == (len(self.blocks) - 2):
                x_prev = self.toRGB_blocks[-2](x)

            # x = Upscale2d(x)
            # for convLayer in layerGroup:
            #     x = self.leakyRelu(convLayer(x))
            #     if self.normalizationLayer is not None:
            #         x = self.normalizationLayer(x)

            # if self.alpha > 0 and scale == (len(self.scaleLayers) - 2):
            #     y = self.toRGBLayers[-2](x)
            #     y = Upscale2d(y)

        # To RGB (no alpha parameter for now)
        x = self.toRGB_blocks[-1](x)

        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            x = self.alpha * y + (1.0-self.alpha) * x

        if self.generationActivation is not None:
            x = self.generationActivation(x)

        return x


class Generator_Adain_Upsample(nn.Module):
    def __init__(self, style_dim=512, n_blocks=9, scale_factor=1, activation='relu'):
        assert (n_blocks >= 0)
        super(Generator_Adain_Upsample, self).__init__()

        self.first_layer = ConvBlock(3, 64, kernel_size=7, stride=1, padding=3)
        self.down1 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.down2 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.down3 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                AdaINResBlock(512, 512, scale_factor=scale_factor, activation=activation, style_dim=style_dim)]
        self.BottleNeck = nn.Sequential(*BN)
        
        self.up3 = ConvBlock(512, 256, kernel_size=3, stride=2, padding=1, transpose=True)
        self.up2 = ConvBlock(256, 128, kernel_size=3, stride=2, padding=1, transpose=True)
        self.up1 = ConvBlock(128, 64, kernel_size=3, stride=2, padding=1, transpose=True)
        self.last_layer = ConvBlock(64, 3, kernel_size=7, stride=1, padding=3, activation_type="tanh")

        # face recognition model: arcface
        self.arcface = arcface.Backbone(50, 0.6, 'ir_se').eval()
        self.arcface.load_state_dict(torch.load('ptnn/arcface.pth', map_location="cuda"), strict=False)
        for param in self.arcface.parameters():
            param.requires_grad = False

    def get_id(self, I):
        return self.arcface(F.interpolate(I[:, :, 16:240, 16:240], [112, 112], mode='bilinear', align_corners=True))

    def forward(self, I_source, I_target):
        id_source = self.get_id(I_source)
        x = I_target

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        x = self.down3(skip3)

        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, id_source)

        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.last_layer(x)
        return x
        