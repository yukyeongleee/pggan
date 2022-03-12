import submodel.arcface as arcface
import torch
from torch import nn
import torch.nn.functional as F
from lib.utils import AdaIN
from lib.blocks import ConvBlock, AdaINResBlock



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
        