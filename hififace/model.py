
import torch
from lib import checkpoint, utils
from lib.model import FaceSwapInterface
from hififace.loss import HifiFaceLoss
from hififace.nets import HififaceGenerator
from submodel.discriminator import StarGANv2Discriminator


class HifiFace(FaceSwapInterface):
    def __init__(self, args, gpu):
        self.upsample = torch.nn.Upsample(scale_factor=4).to(gpu).eval()
        super().__init__(args, gpu)

    def initialize_models(self):
        self.G = HififaceGenerator().cuda(self.gpu).train()
        self.D = StarGANv2Discriminator().cuda(self.gpu).train()

    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        self.G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True).module
        self.D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[self.gpu]).module

    def load_checkpoint(self, step=-1):
        checkpoint.load_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.load_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

    def set_optimizers(self):
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.args.lr_G, betas=(0, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.args.lr_D, betas=(0, 0.999))

    def set_loss_collector(self):
        self._loss_collector = HifiFaceLoss(self.args)

    def train_step(self):
        I_source, I_target, same_person = self.load_next_batch()

        ###########
        # train G #
        ###########
        
        I_swapped_high, I_swapped_low, z_fuse, c_fuse = self.G(I_source, I_target)
        I_swapped_low = self.upsample(I_swapped_low)
        I_cycle = self.G(I_target, I_swapped_high)[0]
        
        # Arcface 
        id_source = self.G.SAIE.get_id(I_source)
        id_swapped_high = self.G.SAIE.get_id(I_swapped_high)
        id_swapped_low = self.G.SAIE.get_id(I_swapped_low)

        # 3D landmarks
        q_swapped_high = self.G.SAIE.get_lm3d(self.G.SAIE.get_coeff3d(I_swapped_high))
        q_swapped_low = self.G.SAIE.get_lm3d(self.G.SAIE.get_coeff3d(I_swapped_low))
        q_fuse = self.G.SAIE.get_lm3d(c_fuse)

        # adversarial
        d_adv = self.D(I_swapped_high)

        G_dict = {

            "I_source": I_source,
            "I_target": I_target,
            "I_swapped_high": I_swapped_high, 
            "I_swapped_low": I_swapped_low,
            "I_cycle": I_cycle,

            "same_person": same_person,

            "id_source": id_source,
            "id_swapped_high": id_swapped_high,
            "id_swapped_low": id_swapped_low,

            "q_swapped_high": q_swapped_high,
            "q_swapped_low": q_swapped_low,
            "q_fuse": q_fuse,

            "d_adv": d_adv

        }

        loss_G = self.loss_collector.get_loss_G(G_dict)
        utils.update_net(self.opt_G, loss_G)

        ###########
        # train D #
        ###########

        I_target.requires_grad_()
        d_true = self.D(I_target)
        d_fake = self.D(I_swapped_high.detach())
        
        D_dict = {
            "d_true": d_true,
            "d_fake": d_fake,
            "I_target": I_target
        }

        loss_D = self.loss_collector.get_loss_D(D_dict)
        utils.update_net(self.opt_D, loss_D)

        return [I_source, I_target, I_swapped_high, I_cycle, I_swapped_low, self.upsample(z_fuse[:, :3, :, :])]

    def validation(self, step):
        with torch.no_grad():
            Y = self.G(self.valid_source, self.valid_target)[0]
        utils.save_image(self.args, step, "valid_imgs", [self.valid_source, self.valid_target, Y])

    def save_image(self, result, step):
        utils.save_image(self.args, step, "imgs", result)
        
    def save_checkpoint(self, step):
        checkpoint.save_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.save_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)   

    @property
    def loss_collector(self):
        return self._loss_collector
