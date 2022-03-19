import torch
from torch.utils.data import DataLoader

from lib import checkpoint, utils
from lib.model import ModelInterface
from lib.dataset import UnsupervisedDataset
from pggan.nets import Generator, Discriminator
from pggan.loss import WGANGPLoss


class ProgressiveGAN(ModelInterface):
    def __init__(self, args, gpu):
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        super().__init__(args, gpu)
        
    def initialize_models(self):
        self.initialize_generator()
        self.initialize_discriminator()

    def initialize_generator(self):
        self.G = Generator(self.args.latent_dim,
                            self.args.depths[0],
                            self.args.initBiasToZero,
                            self.args.LReLU_slope,
                            self.args.apply_pixel_norm,
                            self.args.generationActivation,
                            self.args.output_dim,
                            self.args.equalizedlR)

        # Add scales if necessary
        for depth in self.args.depths[1:]:
            self.G.add_block(depth)

        # If new scales are added, give the generator a blending layer
        if self.args.depths[1:]:
            self.G.set_new_alpha(self.args.alpha)

        self.G.cuda(self.gpu).train()

    def initialize_discriminator(self):
        self.D = Discriminator(self.args.depths[0],
                                self.args.initBiasToZero,
                                self.args.LReLU_slope,
                                self.args.decision_layer_size,
                                self.args.apply_minibatch_norm,
                                self.args.input_dim, # input_dim output_dim
                                self.args.equalizedlR)

        # Add scales if necessary
        for depth in self.args.depths[1:]:
            self.D.add_block(depth)

        # If new scales are added, give the generator a blending layer
        if self.args.depths[1:]:
            self.D.set_new_alpha(self.args.alpha)

        self.D.cuda(self.gpu).train()

    # Override
    def load_next_batch(self):
        """
        Load next batch of source image, target image, and boolean values that denote 
        if source and target are identical.
        """
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            batch = next(self.train_iterator)
        batch = batch.to(self.gpu)
        return batch

    # Override
    def set_dataset(self):
        """
        Initialize dataset using the dataset paths specified in the command line arguments.
        CelebA: 202,599 face images of the size 178×218 from 10,177 celebrities
        """
        dataset = UnsupervisedDataset(self.args.dataset_root, self.args.isMaster)
        N = len(dataset)
        N_train = round(N * 0.7)
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(dataset, [N_train, N - N_train])

    # Override
    def set_validation(self):
        """
        Predefine test images only if args.valid_dataset_root is specified.
        These images are anchored for checking the improvement of the model.
        """
        if False: # self.args.valid_dataset_root: # assets/valid
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.args.batch_per_gpu, num_workers=8, drop_last=True)
            batch = next(iter(self.valid_dataloader))
            self.valid_batch = batch.to(self.gpu)

    def set_loss_collector(self):
        self._loss_collector = WGANGPLoss(self.args)

    def train_step(self):
        """
        Corresponds to optimizeParameters from pytorch_GAN_zoo/models/base_GAN.py
        """
        
        img_real = self.load_next_batch()
        # print("img_real", img_real.shape) # (8, 3, 256, 256)
        n_samples = len(img_real)


        ###########
        # Train D #
        ###########

        img_real.requires_grad_()
        pred_real = self.D(img_real)
        
        latent_input = torch.randn(n_samples, self.args.latent_dim).to(self.gpu)
        img_fake = self.G(latent_input).detach()
        pred_fake = self.D(img_fake)

        D_dict = {
            "img_real": img_real,
            "img_fake": img_fake,
            "pred_real": pred_real,
            "pred_fake": pred_fake,
        }

        loss_D = self.loss_collector.get_loss_D(D_dict)
        utils.update_net(self.opt_D, loss_D)

        ###########
        # Train G #
        ###########

        latent_input = torch.randn(n_samples, self.args.latent_dim).to(self.gpu)
        img_fake = self.G(latent_input)
        pred_fake, _ = self.D(img_fake, True)

        G_dict = {
            "pred_fake": pred_fake,
        }

        loss_G = self.loss_collector.get_loss_G(G_dict)
        utils.update_net(self.opt_G, loss_G)

        return pred_fake

    def validation(self, step):
        with torch.no_grad():
            Y = self.G(self.valid_source, self.valid_target)
        utils.save_image(self.args, step, "valid_imgs", [self.valid_source, self.valid_target, Y])

    def save_image(self, result, step):
        utils.save_image(self.args, step, "imgs", result)
        
    @property
    def loss_collector(self):
        return self._loss_collector