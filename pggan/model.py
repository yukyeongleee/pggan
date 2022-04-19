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
    def load_checkpoint(self, step=-1):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """
        # No checkpoint
        if step == -1:
            return 0

        scale_jump_step = 0
        for scale_index, max_step_at_scale in enumerate(self.args.max_step_at_scale):

            scale_jump_step += max_step_at_scale

            if step >= scale_jump_step:             
                self.G.add_block(self.args.depths[scale_index + 1]) 
                self.D.add_block(self.args.depths[scale_index + 1]) 
                self.G.cuda()
                self.D.cuda()

            else:
                self.scale_index = scale_index
                self.scale_jump_step = scale_jump_step

                self.G.alpha = 0
                self.D.alpha = 0
                self.alpha_index = 0
                self.alpha_jump_step = scale_jump_step - max_step_at_scale + self.args.alpha_jump_start[scale_index]

                # Check whether the given checkpoint matches with model parameters
                ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/G_{step}.pt'
                ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cuda'))
                A = set(ckpt_dict['model'].keys())
                B = set(self.G.state_dict().keys())
                assert A == B

                # No blending at Scale 0
                if scale_index > 0:
                    step_at_scale = step - self.alpha_jump_step 
                    
                    if step_at_scale >= 0:       
                        self.alpha_index = min(step_at_scale // self.args.alpha_jump_interval[scale_index] + 1,  self.args.alpha_jump_Ntimes[scale_index])
                        self.alpha_jump_step += self.alpha_index * self.args.alpha_jump_interval[scale_index]
                        alpha = self.alpha_index / self.args.alpha_jump_Ntimes[scale_index]
                        self.G.alpha = alpha
                        self.D.alpha = alpha

                break

        checkpoint.load_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.load_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

        # Resize real images from dataset
        self.set_dataset()
        self.set_data_iterator()

        return step

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
        dataset = UnsupervisedDataset(self.args.dataset_root, self.scale_index, self.args.isMaster)
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

        loss_D = self.loss_collector.get_loss_D(D_dict, self.D)
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

    def validation(self, step):
        with torch.no_grad():
            n_valid_samples = 5
            latent_input = torch.randn(n_valid_samples, self.args.latent_dim).to(self.gpu)
            Y = self.G(latent_input)
        utils.save_image(self.args, step, "valid_imgs", Y)

    def save_image(self, result, step):
        utils.save_image(self.args, step, "imgs", result)
        
    @property
    def loss_collector(self):
        return self._loss_collector
