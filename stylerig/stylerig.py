import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import weight_init
from torchvision import transforms
from submodel import arcface
from submodel.deep3dmm import ParametricFaceModel, ReconNet
from submodel.e4e import Encoder4Editing, get_keys
import dnnlib, legacy

class DFR(nn.Module):
    def __init__(self):
        super(DFR, self).__init__()

        # face recognition model: arcface

        MLPs = [nn.Linear(16*512,512), nn.LeakyReLU(0.2)]
        
        for _ in range(2):
            MLPs.append(nn.Linear(512,512))
            MLPs.append(nn.LeakyReLU(0.2))

        MLPs.append(nn.Linear(512,257))
        MLPs.append(nn.LeakyReLU(0.2))
        
        self.MLPs = nn.Sequential(*MLPs)

        self.apply(weight_init)
        
    def forward(self, w):
        p = self.MLPs(w.view(-1, 16*512))
        return p

class RIGNET(nn.Module):
    def __init__(self):
        super(RIGNET, self).__init__()

        # rignet
        self.E = nn.ModuleDict()
        for i in range(16):
            layers = [nn.Linear(512,32), nn.LeakyReLU(0.2)]
            for _ in range(2):
                layers.append(nn.Linear(32,32))
                layers.append(nn.LeakyReLU(0.2))
            self.E[f'MLP_{i}'] = nn.Sequential(*layers)

        self.D = nn.ModuleDict()
        for i in range(16):
            # layers = [nn.Linear(64+3+32,512), nn.LeakyReLU(0.2)]
            layers = [nn.Linear(3+32,512), nn.LeakyReLU(0.2)]
            for _ in range(2):
                layers.append(nn.Linear(512,512))
                layers.append(nn.LeakyReLU(0.2))
            self.D[f'MLP_{i}'] = nn.Sequential(*layers)

        self.E.apply(weight_init)
        self.D.apply(weight_init)
        
        # face recognition model: arcface
        self.arcface = arcface.Backbone(50, 0.6, 'ir_se')
        self.arcface.load_state_dict(torch.load('ptnn/arcface.pth', map_location="cuda"), strict=False)
        self.arcface.eval()
        for param in self.arcface.parameters():
            param.requires_grad = False

        # normalization
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                
        # 3D face reconstruction model
        self.net_recon = ReconNet()
        state_dict = torch.load("ptnn/deep3d.pth", map_location="cuda")
        self.net_recon.load_state_dict(state_dict['net_recon'])
        self.net_recon.eval()
        for param in self.net_recon.parameters():
            param.requires_grad = False
        self.facemodel = ParametricFaceModel(is_train=False)

        # # renderer
        # self.renderer = MeshRenderer(
        #     rasterize_fov=2 * np.arctan(112 / 1015) * 180 / np.pi, znear=5, zfar=15, rasterize_size=int(2 * 112)
        # )
        # for param in self.renderer.parameters():
        #     param.requires_grad = False
        
        # stylegan
        # with dnnlib.util.open_url("ptnn/newrot512_3000.pkl") as f:
        with dnnlib.util.open_url("ptnn/kface_combine_512_3600.pkl") as f:
            net_Dict = legacy.load_network_pkl(f)
            self.StyleGAN_G = net_Dict['G_ema']
        self.StyleGAN_G.cuda().eval()
        for param in self.StyleGAN_G.parameters():
            param.requires_grad = False

        # face pooling
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256)).eval()

        self.e4e = Encoder4Editing(50, 'ir_se', size=512)
        ckpt = torch.load("ptnn/iteration_100000.pt", map_location='cuda')
        self.e4e.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        self.e4e.eval()
        for param in self.e4e.parameters():
            param.requires_grad = False
        self.latent_avg = self.StyleGAN_G.mapping.w_avg.unsqueeze(0).unsqueeze(0).repeat([1,16,1])

    def encoder(self, w):

        out = []
        for i in range(16):
            out.append(self.E[f'MLP_{i}'](w[:, i, :]).unsqueeze(1))

        return torch.cat(out, dim=1)

    def decoder(self, l_w, c_angle):
        out = []
        for i in range(16):
            input = torch.cat([l_w[:, i, :], c_angle], dim=1)
            out.append(self.D[f'MLP_{i}'](input).unsqueeze(1))

        return torch.cat(out, dim=1)


    def get_id(self, I):
        return self.arcface(F.interpolate(I[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))

    def get_render(self, coeff):
        pred_vertex, pred_tex, pred_color, pred_lm = self.facemodel.compute_for_render(coeff)
        pred_mask, _, pred_face = self.renderer(pred_vertex, self.facemodel.face_buf, feat=pred_color)
        return pred_mask, pred_face, pred_lm
        
    def get_3dlm_from_w(self, w):
        I = self.get_image_from_w(w)
        coeff = self.get_coeff3d(I/2+0.5)
        pred_vertex, pred_tex, pred_color, pred_lm = self.facemodel.compute_for_render(coeff)
        return pred_lm

    def get_coeff3d(self, I):
        coeffs = self.net_recon(I[:, :, 16:240, 16:240])
        return coeffs

    def get_image_from_w(self, w):
        image = self.face_pool(self.StyleGAN_G.synthesis(w, noise_mode='const'))
        return image
        
    def get_mix_3dlm(self, I_source, I_target):
        with torch.no_grad():
            c_source = self.get_coeff3d(I_source)
            c_target = self.get_coeff3d(I_target)
            c_source[:, 224: 227] = c_target[:, 224: 227]
            pred_vertex, pred_tex, pred_color, pred_lm = self.facemodel.compute_for_render(c_source)
        return pred_lm

    def get_w_with_inversion(self, image):
        w = self.e4e(image) + self.latent_avg
        return w

    def get_recon_image(self, image):
        w = self.get_w_with_inversion(image)
        recon_image = self.get_image_from_w(w)
        return recon_image, w

    def get_angle(self, image):
        coeff = self.get_coeff3d(image)
        return coeff[:, 224: 227]

    def forward(self, I_source, target_angle):

        w_source = self.get_w_with_inversion(I_source)
        l_w = self.encoder(w_source)
        d = self.decoder(l_w, target_angle)
        w_reenact = w_source + d
        lm3d_reenact = self.get_3dlm_from_w(w_reenact)

        return w_reenact, lm3d_reenact
