import torch
from lib import checkpoint, utils
from lib.model import ModelInterface
from stylerig.loss import StyleRigLoss
from stylerig.stylerig import RIGNET
from submodel.discriminator import LatentCodesDiscriminator
import torch.nn.functional as F


class StyleRig(ModelInterface):
    def initialize_models(self):
        self.G = RIGNET().cuda(self.gpu).train()
        self.D = LatentCodesDiscriminator().cuda(self.gpu).train()

    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        # Data parallelism is required to use multi-GPU
        self.G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True).module
        self.D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[self.gpu]).module
        
    def load_checkpoint(self, step=-1):
        checkpoint.load_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.load_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

    def set_optimizers(self):
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.args.lr_G, betas=(0, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.args.lr_D, betas=(0, 0.999))

    def set_loss_collector(self):
        self._loss_collector = StyleRigLoss(self.args)

    def train_step(self):
        I_source, I_target, same_person = self.load_next_batch()
        
        ###########
        # train G #
        ###########

        target_angle = self.G.get_angle(I_target)
        w_reenact, lm3d_reenact = self.G(I_source, target_angle)
        I_reenact = self.G.get_image_from_w(w_reenact)
        
        lm3d_mix = self.G.get_mix_3dlm(I_source, I_target)
        
        v_source = self.G.get_id(I_source)
        v_target = self.G.get_id(I_target)
        v_reenact = self.G.get_id(I_reenact)

        d_adv = [] 
        for i in range(16):
            w_slice = w_reenact[:, i, :]
            d_adv.append(self.D(w_slice))

        G_dict = {

            "I_source": I_source,
            "I_target": I_target, 
            "I_reenact": I_reenact, 

            "v_source": v_source, 
            "v_target": v_target, 
            "v_reenact": v_reenact, 
            
            # "face3d_reenact": face3d_reenact, 
            # "face3d_mix": face3d_mix,
            "lm3d_mix": lm3d_mix,
            "lm3d_reenact": lm3d_reenact,

            "d_adv": d_adv

        }

        loss_G = self.loss_collector.get_loss_G(G_dict)
        utils.update_net(self.opt_G, loss_G)

        ###########
        # train D #
        ###########

        z = torch.from_numpy(self.RandomGenerator.randn(self.args.batch_per_gpu, 512)).to(self.gpu)
        w = self.G.StyleGAN_G.mapping(z, None, truncation_psi=0.8)
        d_real = [] 
        for i in range(16):
            w_slice = w.detach()[:, i, :]
            d_real.append(self.D(w_slice))

        d_fake = [] 
        for i in range(16):
            w_slice = w_reenact.detach()[:, i, :]
            d_fake.append(self.D(w_slice))

        D_dict = {
            "d_real": d_real,
            "d_fake": d_fake,
        }
        
        loss_D = self.loss_collector.get_loss_D(D_dict)
        utils.update_net(self.opt_D, loss_D)

        recon_source, w_source = self.G.get_recon_image(I_source)
        return [I_source, recon_source, I_target, I_reenact]

    def validation(self, step):
        return
        # with torch.no_grad():
        #     Y = self.G(self.valid_source, self.valid_target)[0]
        # utils.save_image(self.args, step, "valid_imgs", [self.valid_source, self.valid_target, Y])

    def save_image(self, result, step):
        utils.save_image(self.args, step, "imgs", result)
        
    def save_checkpoint(self, step):
        checkpoint.save_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.save_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

    @property
    def loss_collector(self):
        return self._loss_collector
        