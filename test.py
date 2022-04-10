import torch
import wandb
import os
import sys

from lib.model_loader import CreateModel

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from lib.utils import make_grid_image
import cv2
import glob
import torchvision

sys.path.append("./")
sys.path.append("./submodel/")
sys.path.append("./submodel/stylegan2")

from pggan.nets import Generator
from pggan.options import TestOptions

def LoadModel(args):
    G = Generator(args.latent_dim,
                        args.depths[0],
                        args.init_bias_to_zero,
                        args.LReLU_slope,
                        args.apply_pixel_norm,
                        args.generator_last_activation,
                        args.output_dim,
                        args.equalized_lr)

    for depth in args.depths[1:]:
        G.add_block(depth)

    # Load Checkpoint
    ckpt_dict = torch.load(args.ckpt_path, map_location=torch.device('cuda'))

    # Check whether the given checkpoint matches with model parameters
    A = set(ckpt_dict['model'].keys())
    B = set(G.state_dict().keys())
    assert A == B

    G.load_state_dict(ckpt_dict['model'])

    return G.to(args.gpu)

def test(args): 
    # RandomGenerator = np.random.RandomState(42)

    G = LoadModel(args).eval()

    with torch.no_grad():

        latent_input = torch.randn(args.n_samples, args.latent_dim).to(args.gpu)
        generated_imgs = G(latent_input).cpu().detach().numpy()

        generated_imgs = np.transpose(generated_imgs, axes=[0, 2, 3, 1])
        generated_imgs = (generated_imgs * 0.5 + 0.5) * 255
        # print(generated_imgs.shape) # [10, 3, 128, 128]
        
        """
        Issue: dark image
        """
        for i in range(generated_imgs.shape[0]):
            # sample_image = make_grid_image(generated_imgs).transpose([1,2,0]) * 255
            # cv2.imwrite(f'{args.save_path}/result.jpg', sample_image[:,:,::-1])
            cv2.imwrite(f'{args.save_path}/result_{i:2d}.jpg', generated_imgs[i])


if __name__ == "__main__":

    args = TestOptions().parse()

    os.makedirs(args.save_path, exist_ok=True)

    test(args)
