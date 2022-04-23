import torch
from lib.loss import Loss, LossInterface

class WGANGPLoss(LossInterface):
    def get_loss_G(self, G_dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_BCE_loss(G_dict["pred_fake"], True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_G"] = round(L_G.item(), 4)

        return L_G

    def get_loss_D(self, D_dict):
        # Real 
        L_D_real = Loss.get_BCE_loss(D_dict["pred_real"], True)
        L_D_fake = Loss.get_BCE_loss(D_dict["pred_fake"], False)
        L_reg = Loss.get_r1_reg(L_D_real, D_dict["img_real"])
        L_D = L_D_real + L_D_fake + L_reg
        
        self.loss_dict["L_D_real"] = round(L_D_real.mean().item(), 4)
        self.loss_dict["L_D_fake"] = round(L_D_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D

    # def get_loss_D(self, D_dict, discriminator):
        
    #     """
    #         WGANGP 는 Discriminator 를 두 번 update 합니다. 
    #         (get_gradient_penalty 에서 한 번, L_D 를 이용해 model.D 를 업데이트 할 때 한 번)
    #     """

    #     L_D_real = Loss.get_BCE_loss(D_dict["pred_real"], True)
    #     L_D_fake = Loss.get_BCE_loss(D_dict["pred_fake"], False)
    #     L_D = L_D_real + L_D_fake
        
    #     # WGAN-GP gradient loss
    #     L_D_gp = self.get_gradient_penalty(D_dict, discriminator)
        
    #     # Drift loss (the fourth term)
    #     L_D_eps = self.get_drift_loss(D_dict)

    #     self.loss_dict["L_D_real"] = round(L_D_real.mean().item(), 4)
    #     self.loss_dict["L_D_fake"] = round(L_D_fake.mean().item(), 4)
    #     self.loss_dict["L_D_gp"] = round(L_D_gp, 4)
    #     self.loss_dict["L_D_eps"] = round(L_D_eps, 4)
    #     self.loss_dict["L_D"] = round(L_D.item() + L_D_gp + L_D_eps, 4)
    #     return L_D


    def get_gradient_penalty(self, D_dict, discriminator, backward=True):
        r"""
        Gradient penalty as described in
        "Improved Training of Wasserstein GANs"
        https://arxiv.org/pdf/1704.00028.pdf
        Args:
            - input (Tensor): batch of real data
            - fake (Tensor): batch of generated data. Must have the same size
            as the input
            - discrimator (nn.Module): discriminator network
            - weight (float): weight to apply to the penalty term
            - backward (bool): loss backpropagation
        """

        if self.args.W_gp:
                
            batchSize = D_dict["img_real"].size(0)        
            eps = torch.rand(batchSize, 1)
            eps = eps.expand(batchSize, int(D_dict["img_real"].nelement()/batchSize)).contiguous().view(D_dict["img_real"].size())
            eps = eps.to(D_dict["img_real"].get_device())
            
            interpolates = eps * D_dict["img_real"] + ((1 - eps) * D_dict["img_fake"])
            torch.autograd.Variable(interpolates, requires_grad=True)

            decisionInterpolate = discriminator(interpolates)
            decisionInterpolate = decisionInterpolate[:, 0].sum()

            gradients = torch.autograd.grad(outputs=decisionInterpolate,
                                            inputs=interpolates,
                                            create_graph=True, retain_graph=True)

            gradients = gradients[0].view(batchSize, -1)
            gradients = (gradients * gradients).sum(dim=1).sqrt()
            gradient_penalty = (((gradients - 1.0)**2)).sum() * self.args.W_gp

            if backward:
                gradient_penalty.backward(retain_graph=True)

        return gradient_penalty.item()

    def get_drift_loss(self, D_dict):
        """
        Loss for keeping D output from drifting too far away from 0
        """
        if self.args.W_drift_D:
            drift = (D_dict["pred_real"] ** 2).sum() * self.args.W_drift_D
            return drift.item()