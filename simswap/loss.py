from lib.loss import Loss, LossInterface


class SimSwapLoss(LossInterface):
    def get_loss_G(self, G_dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_BCE_loss(G_dict["g_fake"], True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        
        # Identity loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(G_dict["id_source"], G_dict["id_swapped"])
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L1_loss_with_same_person(G_dict["I_swapped"], G_dict["I_target"], G_dict["same_person"], self.args.batch_per_gpu)
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        
        # LPIPS loss
        if self.args.W_lpips:
            L_lpips = Loss.get_lpips_loss(G_dict["I_swapped"], G_dict["I_target"])
            L_G += self.args.W_lpips * L_lpips
            self.loss_dict["L_lpips"] = round(L_lpips.item(), 4)
        
        # Feature matching loss 
        if self.args.W_fm:
            L_fm = 0
            n_layers_D = 4
            num_D = 2
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            for i in range(0, n_layers_D):
                L_fm += D_weights * feat_weights * Loss.get_L1_loss(G_dict["g_fake"][i], G_dict["g_real"][i].detach())
            L_G += self.args.W_fm * L_fm
            self.loss_dict["L_fm"] = round(L_recon.item(), 4)

        self.loss_dict["L_G"] = round(L_G.item(), 4)

        return L_G

    def get_loss_D(self, D_dict):
        # Real 
        L_D_real = Loss.get_BCE_loss(D_dict["d_real"], True)
        L_D_fake = Loss.get_BCE_loss(D_dict["d_fake"], False)
        L_reg = Loss.get_r1_reg(L_D_real, D_dict["I_target"])
        L_D = L_D_real + L_D_fake + L_reg
        
        self.loss_dict["L_D_real"] = round(L_D_real.mean().item(), 4)
        self.loss_dict["L_D_fake"] = round(L_D_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        