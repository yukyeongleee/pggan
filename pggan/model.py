import torch
from lib import checkpoint, utils
from lib.model import ModelInterface
from pggan.nets import Generator
from submodel.discriminator import StarGANv2Discriminator
from simswap.loss import SimSwapLoss


class ProgressiveGAN(ModelInterface):
    def __init__(self, args, gpu):
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        super().__init__(args, gpu)

    def initialize_models(self):
        self.G = Generator().cuda(self.gpu).train()
        self.D = StarGANv2Discriminator().cuda(self.gpu).train()

    def set_loss_collector(self):
        self._loss_collector = SimSwapLoss(self.args)

    def train_step(self):
        I_source, I_target, same_person = self.load_next_batch()

        ###########
        # Train G #
        ###########

        # Run G to swap identity from source to target image
        I_swapped = self.G(I_source, I_target)
        I_cycle = self.G(I_target, I_swapped)

        id_source = self.G.get_id(I_source)
        id_swapped = self.G.get_id(I_swapped)

        g_real = self.D(I_target)
        g_fake = self.D(I_swapped.detach())
        
        G_dict = {
            "I_source": I_source,
            "I_target": I_target, 
            "I_swapped": I_swapped,
            "I_cycle": I_cycle,

            "same_person": same_person,

            "id_source": id_source,
            "id_swapped": id_swapped,

            "g_real": g_real,
            "g_fake": g_fake
        }

        loss_G = self.loss_collector.get_loss_G(G_dict)
        utils.update_net(self.opt_G, loss_G)

        ###########
        # Train D #
        ###########

        I_target.requires_grad_()
        d_real = self.D(I_target)
        d_fake = self.D(I_swapped.detach())

        D_dict = {
            "I_target": I_target,
            "d_real": d_real,
            "d_fake": d_fake,
        }

        loss_D = self.loss_collector.get_loss_D(D_dict)
        utils.update_net(self.opt_D, loss_D)

        return [I_source, I_target, I_swapped]

    def validation(self, step):
        with torch.no_grad():
            Y = self.G(self.valid_source, self.valid_target)
        utils.save_image(self.args, step, "valid_imgs", [self.valid_source, self.valid_target, Y])

    def save_image(self, result, step):
        utils.save_image(self.args, step, "imgs", result)
        
    @property
    def loss_collector(self):
        return self._loss_collector
