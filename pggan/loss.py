import torch
from lib.loss import Loss, LossInterface

class WGANGPLoss(LossInterface):
    def get_loss_G(self, G_dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_BCE_loss(G_dict["pred_fake"], True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        
        self.loss_dict["L_G"] = round(L_G.item(), 4)

        return L_G


    def get_loss_D(self, D_dict, discriminator):
        # Real 
        L_D_real = Loss.get_BCE_loss(D_dict["pred_real"], True)
        L_D_fake = Loss.get_BCE_loss(D_dict["pred_fake"], False)
        L_gp = self.get_gradient_penalty(D_dict, discriminator, self.args.W_gp)
        # L_reg = Loss.get_r1_reg(L_D_real, D_dict["I_target"])
        L_D = L_D_real + L_D_fake + L_gp
        
        self.loss_dict["L_D_real"] = round(L_D_real.mean().item(), 4)
        self.loss_dict["L_D_fake"] = round(L_D_fake.mean().item(), 4)
        self.loss_dict["L_gp"] = round(L_D.item(), 4)

        return L_D


    def get_gradient_penalty(D_dict, discriminator, weight, backward=True):
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

        n_samples = len(D_dict["img_real"].size)
        
        eps = torch.rand(n_samples, 1)
        eps = eps.expand_as(n_samples, D_dict["img_real"]).contiguous()
        eps = eps.to(input.device)
        
        interpolates = eps * input + ((1 - eps) * D_dict["img_fake"])
        interpolates.requires_grad = True

        pred_interpolates = discriminator(interpolates)
        # decisionInterpolate = decisionInterpolate[:, 0].sum()

        gradients = torch.autograd.grad(outputs=pred_interpolates,
                                        inputs=interpolates,
                                        create_graph=True, retain_graph=True)

        gradients = gradients[0].view(n_samples, -1)
        gradients = (gradients * gradients).sum(dim=1).sqrt()
        gradient_penalty = (((gradients - 1.0)**2)).sum() * weight

        if backward:
            gradient_penalty.backward(retain_graph=True)

        return gradient_penalty.item()

    # def get_epsilon_loss(self, x):
    #     return 0

    # def get_logistic_gradient_penalty(self, x):
    #     return 0
