import torch
import torch.nn as nn
import torch.nn.functional as F
from submodel import arcface
from lib.utils import weight_init
from lib.blocks import ConvBlock


class AADLayer(nn.Module):
    def __init__(self, cin, c_attr, c_id=512):
        super(AADLayer, self).__init__()
        self.c_attr = c_attr
        self.c_id = c_id
        self.cin = cin

        self.conv1 = nn.Conv2d(c_attr, cin, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(c_attr, cin, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(c_id, cin)
        self.fc2 = nn.Linear(c_id, cin)
        self.norm = nn.InstanceNorm2d(cin, affine=False)

        self.conv_h = nn.Conv2d(cin, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, h_in, z_attr, z_id):
        # h_in cxnxn
        # zid 256x1x1
        # zattr cxnxn
        h = self.norm(h_in)
        gamma_attr = self.conv1(z_attr)
        beta_attr = self.conv2(z_attr)

        gamma_id = self.fc1(z_id)
        beta_id = self.fc2(z_id)
        A = gamma_attr * h + beta_attr
        gamma_id = gamma_id.reshape(h.shape[0], self.cin, 1, 1).expand_as(h)
        beta_id = beta_id.reshape(h.shape[0], self.cin, 1, 1).expand_as(h)
        I = gamma_id * h + beta_id

        M = torch.sigmoid(self.conv_h(h))

        out = (torch.ones_like(M).to(M.device) - M) * A + M * I
        return out


class AAD_ResBlk(nn.Module):
    def __init__(self, cin, cout, c_attr, c_id=512):
        super(AAD_ResBlk, self).__init__()
        self.cin = cin
        self.cout = cout

        self.AAD1 = AADLayer(cin, c_attr, c_id)
        self.conv1 = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.AAD2 = AADLayer(cin, c_attr, c_id)
        self.conv2 = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        if cin != cout:
            self.AAD3 = AADLayer(cin, c_attr, c_id)
            self.conv3 = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu3 = nn.ReLU(inplace=True)

    def forward(self, h, z_attr, z_id):
        x = self.AAD1(h, z_attr, z_id)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.AAD2(x, z_attr, z_id)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.cin != self.cout:
            h = self.AAD3(h, z_attr, z_id)
            h = self.relu3(h)
            h = self.conv3(h)
        x = x + h
        
        return x


class MLAttrEncoder(nn.Module):
    def __init__(self):
        super(MLAttrEncoder, self).__init__()
        norm = "bn"
        activation = "lrelu"
        
        self.conv1 = ConvBlock(3, 32, 3, 2, 1, norm_type=norm, activation_type=activation)
        self.conv2 = ConvBlock(32, 64, 3, 2, 1, norm_type=norm, activation_type=activation)
        self.conv3 = ConvBlock(64, 128, 3, 2, 1, norm_type=norm, activation_type=activation)
        self.conv4 = ConvBlock(128, 256, 3, 2, 1, norm_type=norm, activation_type=activation)
        self.conv5 = ConvBlock(256, 512, 3, 2, 1, norm_type=norm, activation_type=activation)
        self.conv6 = ConvBlock(512, 1024, 3, 2, 1, norm_type=norm, activation_type=activation)
        self.conv7 = ConvBlock(1024, 1024, 3, 2, 1, norm_type=norm, activation_type=activation)

        self.deconv1 = ConvBlock(1024, 1024, 3, 2, 1, norm_type=norm, activation_type=activation, transpose=True)
        self.deconv2 = ConvBlock(2048, 512, 3, 2, 1, norm_type=norm, activation_type=activation, transpose=True)
        self.deconv3 = ConvBlock(1024, 256, 3, 2, 1, norm_type=norm, activation_type=activation, transpose=True)
        self.deconv4 = ConvBlock(512, 128, 3, 2, 1, norm_type=norm, activation_type=activation, transpose=True)
        self.deconv5 = ConvBlock(256, 64, 3, 2, 1, norm_type=norm, activation_type=activation, transpose=True)
        self.deconv6 = ConvBlock(128, 32, 3, 2, 1, norm_type=norm, activation_type=activation, transpose=True)

        self.apply(weight_init)

    def forward(self, Xt):
        feat1 = self.conv1(Xt)
        # 32x128x128
        feat2 = self.conv2(feat1)
        # 64x64x64
        feat3 = self.conv3(feat2)
        # 128x32x32
        feat4 = self.conv4(feat3)
        # 256x16xx16
        feat5 = self.conv5(feat4)
        # 512x8x8
        feat6 = self.conv6(feat5)
        # 1024x4x4
        z_attr1 = self.conv7(feat6)
        # 1024x2x2

        z_attr2 = self.deconv1(z_attr1)
        z_attr3 = self.deconv2(torch.cat((z_attr2, feat6), dim=1))
        z_attr4 = self.deconv3(torch.cat((z_attr3, feat5), dim=1))
        z_attr5 = self.deconv4(torch.cat((z_attr4, feat4), dim=1))
        z_attr6 = self.deconv5(torch.cat((z_attr5, feat3), dim=1))
        z_attr7 = self.deconv6(torch.cat((z_attr6, feat2), dim=1))
        z_attr8 = F.interpolate(torch.cat((z_attr7, feat1), dim=1), scale_factor=2, mode='bilinear', align_corners=True)
        return z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8


class AADGenerator(nn.Module):
    def __init__(self, c_id=512):
        super(AADGenerator, self).__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.AADBlk1 = AAD_ResBlk(1024, 1024, 1024, c_id)
        self.AADBlk2 = AAD_ResBlk(1024, 1024, 1024, c_id)
        self.AADBlk3 = AAD_ResBlk(1024, 1024, 512, c_id)
        self.AADBlk4 = AAD_ResBlk(1024, 512, 256, c_id)
        self.AADBlk5 = AAD_ResBlk(512, 256, 128, c_id)
        self.AADBlk6 = AAD_ResBlk(256, 128, 64, c_id)
        self.AADBlk7 = AAD_ResBlk(128, 64, 32, c_id)
        self.AADBlk8 = AAD_ResBlk(64, 3, 64, c_id)

        self.apply(weight_init)

    def forward(self, z_attr, z_id):
        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))
        m2 = F.interpolate(self.AADBlk1(m, z_attr[0], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m3 = F.interpolate(self.AADBlk2(m2, z_attr[1], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m4 = F.interpolate(self.AADBlk3(m3, z_attr[2], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m5 = F.interpolate(self.AADBlk4(m4, z_attr[3], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m6 = F.interpolate(self.AADBlk5(m5, z_attr[4], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m7 = F.interpolate(self.AADBlk6(m6, z_attr[5], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m8 = F.interpolate(self.AADBlk7(m7, z_attr[6], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        y = self.AADBlk8(m8, z_attr[7], z_id)
        return torch.tanh(y)


class AEI_Net(nn.Module):
    def __init__(self, c_id=512):
        super(AEI_Net, self).__init__()
        self.encoder = MLAttrEncoder()
        self.generator = AADGenerator(c_id)

        # face recognition model: arcface
        self.arcface = arcface.Backbone(50, 0.6, 'ir_se').eval()
        self.arcface.load_state_dict(torch.load('ptnn/arcface.pth', map_location="cuda"), strict=False)
        for param in self.arcface.parameters():
            param.requires_grad = False

    def forward(self, I_s, I_t):
        id = self.get_id(I_s)
        attr = self.get_attr(I_t)
        Y = self.generator(attr, id)
        return Y, id, attr

    def get_attr(self, I):
        return self.encoder(I)

    def get_id(self, I):
        return self.arcface(F.interpolate(I[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
