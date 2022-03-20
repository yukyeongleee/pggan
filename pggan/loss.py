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
        
        """
        comment #7
            WGANGP 는 Discriminator 를 두 번 update 합니다. 
            (get_gradient_penalty 에서 한 번, L_D 를 이용해 model.D 를 업데이트 할 때 한 번)
        """

        # Real 
        L_D_real = Loss.get_BCE_loss(D_dict["pred_real"], True)
        L_D_fake = Loss.get_BCE_loss(D_dict["pred_fake"], False)
        L_D_gp = self.get_gradient_penalty(D_dict, discriminator, self.args.W_gp)
        L_D = L_D_real + L_D_fake
        
        self.loss_dict["L_D_real"] = round(L_D_real.mean().item(), 4)
        self.loss_dict["L_D_fake"] = round(L_D_fake.mean().item(), 4)
        self.loss_dict["L_D_gp"] = round(L_D_gp, 4)
        self.loss_dict["L_D"] = round(L_D.item() + L_D_gp, 4)

        return L_D


    def get_gradient_penalty(self, D_dict, discriminator, weight, backward=True):
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
        
        """
        comment #6
            numpy array 에 사용하는 함수와 tensor 에 사용하는 함수가 혼용되어 있었습니다.
            ex) eps.expand_as vs eps.expand
            일단 아래 링크를 보고 그대로 옮겨 왔는데, 혹시 수정했던 이유가 있으면 알려주세요!
            https://github.com/facebookresearch/pytorch_GAN_zoo/blob/b75dee40918caabb4fe7ec561522717bf096a8cb/models/loss_criterions/gradient_losses.py
        """

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
        gradient_penalty = (((gradients - 1.0)**2)).sum() * weight

        if backward:
            gradient_penalty.backward(retain_graph=True)

        return gradient_penalty.item()

    # def get_epsilon_loss(self, x):
    #     return 0

    # def get_logistic_gradient_penalty(self, x):
    #     return 0
