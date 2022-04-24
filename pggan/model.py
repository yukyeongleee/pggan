import torch
from torch.utils.data import DataLoader

from lib import checkpoint, utils
from lib.model import ModelInterface
from lib.dataset import UnsupervisedDataset
from pggan.nets import Generator, Discriminator
from pggan.loss import WGANGPLoss


class ProgressiveGAN(ModelInterface):
    def __init__(self, args, gpu):
        self.scale_index = 0
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        super().__init__(args, gpu)
        
    def initialize_models(self):
        self.initialize_generator()
        self.initialize_discriminator()

        self.G.train()
        self.D.train()

    def initialize_generator(self):
        self.G = Generator(self.args.latent_dim,
                            self.args.depths[0],
                            self.args.init_bias_to_zero,
                            self.args.LReLU_slope,
                            self.args.apply_pixel_norm,
                            self.args.generator_last_activation,
                            self.args.output_dim,
                            self.args.equalized_lr)


        self.G.cuda(self.gpu)

    def initialize_discriminator(self):
        self.D = Discriminator(self.args.depths[0],
                                self.args.init_bias_to_zero,
                                self.args.LReLU_slope,
                                self.args.decision_layer_size,
                                self.args.apply_minibatch_norm,
                                self.args.input_dim, # input_dim output_dim
                                self.args.equalized_lr)

        self.D.cuda(self.gpu)

    # Override
    def save_checkpoint(self, global_step):
        """
        Save model and optimizer parameters.
        """
        ckpt_dict = {
            "args": self.args.__dict__,
            "global_step": global_step,
            "alpha_G": self.G.alpha,
            "alpha_D": self.D.alpha,
            "alpha_index": self.alpha_index,
            "alpha_jump_value": self.alpha_jump_value,
            "next_alpha_jump_step": self.next_alpha_jump_step,
            "scale_index": self.scale_index,
            "next_scale_jump_step": self.next_scale_jump_step,
        }

        checkpoint.save_checkpoint(self.G, self.opt_G, name='G', ckpt_dict=ckpt_dict)
        checkpoint.save_checkpoint(self.D, self.opt_D, name='D', ckpt_dict=ckpt_dict)
       
    # Override
    def load_checkpoint(self):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """

        G_ckpt_dict, D_ckpt_dict = \
        checkpoint.load_checkpoint(self.args, name='G'), \
        checkpoint.load_checkpoint(self.args, name='D')

        self.args.update(G_ckpt_dict["args"])
        self.global_step = G_ckpt_dict["global_step"]
        self.G.alpha = G_ckpt_dict["alpha_G"]
        self.D.alpha = G_ckpt_dict["alpha_D"]
        self.alpha_index = G_ckpt_dict["alpha_index"]
        self.alpha_jump_value = G_ckpt_dict["alpha_jump_value"]
        self.next_alpha_jump_step = G_ckpt_dict["next_alpha_jump_step"]
        self.scale_index = G_ckpt_dict["scale_index"]
        self.next_scale_jump_step = G_ckpt_dict["next_scale_jump_step"]

        for index in range(self.scale_index):
            self.G.add_block(self.args.depths[index])
            self.D.add_block(self.args.depths[index])
            self.G.cuda()
            self.D.cuda()

        self.reset_solver()
        
        self.G.load_state_dict(G_ckpt_dict['model'], strict=False)
        self.opt_G.load_state_dict(G_ckpt_dict['optimizer'])

        self.D.load_state_dict(D_ckpt_dict['model'], strict=False)
        self.opt_D.load_state_dict(D_ckpt_dict['optimizer'])

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
        dataset = UnsupervisedDataset(self.args.dataset_root_list, self.scale_index, self.args.isMaster)
        N = len(dataset)
        N_train = round(N * 0.7)
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(dataset, [N_train, N - N_train])

    def set_loss_collector(self):
        self._loss_collector = WGANGPLoss(self.args)
    
    def reset_solver(self): 
        """
        Reset data loaders corresponding to the output image size,
        and reset optimizers as the number of learnable parameters are changed.
        This method is required only when the Generator and the Discriminator add a new block.
        """
        self.set_dataset()
        self.set_data_iterator()
        self.set_optimizers()

    def reset_alpha(self, global_step):
        """
        Initialize alpha-related-variables
        This method is required only when the Generator and the Discriminator add a new block.
        """
        self.G.alpha = 0
        self.D.alpha = 0
        self.alpha_index = 0

        self.next_alpha_jump_step = global_step + self.args.alpha_jump_start[self.scale_index]
        self.alpha_jump_value = 1/self.args.alpha_jump_Ntimes[self.scale_index]

        if self.args.isMaster:
            print(f"alpha and alpha_index are initialized to 0")
            print(f"next_alpha_jump_step is set to {self.next_alpha_jump_step}")
            print(f"alpha_jump_value is set to {self.alpha_jump_value}")
        
    def change_scale(self, global_step):
        self.scale_index += 1
        self.next_scale_jump_step += self.args.max_step_at_scale[self.scale_index]
        
        # add a block to net G and net D
        self.G.add_block(self.args.depths[self.scale_index])
        self.D.add_block(self.args.depths[self.scale_index])
        self.G.cuda()
        self.D.cuda()

        self.reset_solver()
        self.reset_alpha(global_step)

        if self.args.isMaster:
            print(f"\nNOW global_step is {global_step}")
            print(f"scale_index is updated to {self.scale_index}")
            print(f"next_scale_jump_step is {self.next_scale_jump_step}")

    def change_alpha(self, global_step): 

        self.alpha_index += 1
        self.G.alpha += round(self.alpha_jump_value, 4)
        self.D.alpha += round(self.alpha_jump_value, 4)
        
        # check if alpha_index is reached to alpha_jump_Ntimes
        if self.alpha_index == self.args.alpha_jump_Ntimes[self.scale_index]:
            self.next_alpha_jump_step = 0
        else: 
            self.next_alpha_jump_step = global_step + self.args.alpha_jump_interval[self.scale_index]
        
        if self.args.isMaster:
            print(f"\nNOW global_step is {global_step}")
            print(f"alpha_index is updated to {self.alpha_index}")
            print(f"next_alpha_jump_step is {self.next_alpha_jump_step}")
            print(f"alpha is now {self.G.alpha}")

    def check_jump(self, global_step):

        # scale 이 바뀔 때
        if self.next_scale_jump_step == global_step:
            self.change_scale(global_step)

        # alpha 가 바뀔 때 (Linear mode)
        if self.next_alpha_jump_step == global_step:
            self.change_alpha(global_step)

    def train_step(self):
        """
        Corresponds to optimizeParameters from pytorch_GAN_zoo/models/base_GAN.py
        """
        
        img_real = self.load_next_batch()
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

        return [img_real, img_fake]

    def save_image(self, images, step):
        utils.save_image(self.args, step, "imgs", images)
        
    @property
    def loss_collector(self):
        return self._loss_collector
