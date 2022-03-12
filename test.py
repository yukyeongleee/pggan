import torch
import wandb
import os
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from lib import utils
import cv2
import glob
import torchvision

sys.path.append("./")
sys.path.append("./submodel/")
sys.path.append("./submodel/stylegan2")
from stylerig.stylerig import RIGNET

RandomGenerator = np.random.RandomState(42)
G = RIGNET().cuda().train()
ckpt_path = f'/home/compu/abc/training_result/d10/ckpt/G_40000.pt'
ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cuda'))
G.load_state_dict(ckpt_dict['model'], strict=False)

transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

img_paths = sorted(glob.glob("/home/compu/abc/samples/k-celeb/*.*g"))

with torch.no_grad():
    for img_path in img_paths:
        img_name = os.path.split(img_path)[1][:-4]
        print(f"processing >>> {img_name}")
        img = Image.open(img_path)
        img = transforms(img).unsqueeze(0).cuda()
        angle = G.get_angle(img)
        recon_source, w_source = G.get_recon_image(img)

        image_list = [img, recon_source]
        for yaw in [0, 15, 30, 45, 60, 75, 90]: 
            target_angle = torch.tensor([[angle[0][0],yaw/90,angle[0][2]]]).cuda()
            w_reenact, lm3d_reenact = G(img, target_angle)
            I_reenact = G.get_image_from_w(w_reenact)
            image_list.append(I_reenact)

        images = torch.cat(image_list, dim=0)
        sample_image = torchvision.utils.make_grid(images.detach().cpu(), nrow=images.shape[0]).numpy().transpose([1,2,0]) * 127.5 + 127.5
        cv2.imwrite(f'samples/result_{img_name}.jpg', sample_image[:,:,::-1])
