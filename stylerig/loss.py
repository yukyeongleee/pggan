from lib.loss import Loss, LossInterface


class StyleRigLoss(LossInterface):
    def get_loss_G(self, G_dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_softplus_loss(G_dict["d_adv"], True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        
        # Id loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(G_dict["v_source"], G_dict["v_reenact"])
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # Landmark loss
        if self.args.W_lm:
            L_lm = Loss.get_L2_loss(G_dict["lm3d_reenact"], G_dict["lm3d_mix"])
            L_G += self.args.W_lm * L_lm/68
            self.loss_dict["L_lm"] = round(L_lm.item(), 4)
        
        # face3d loss
        if self.args.W_face3d:
            L_face3d = Loss.get_L2_loss(G_dict["face3d_reenact"], G_dict["face3d_mix"])
            L_G += self.args.W_face3d * L_face3d
            self.loss_dict["L_face3d"] = round(L_face3d.item(), 4)
        
        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, D_dict):
        L_real = Loss.get_softplus_loss(D_dict["d_real"], True)
        L_fake = Loss.get_softplus_loss(D_dict["d_fake"], False)
        L_D = 0.5*(L_real.mean() + L_fake.mean())
        
        self.loss_dict["L_real"] = round(L_real.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        