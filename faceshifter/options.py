from lib.options import BaseOptions


class TrainOptions(BaseOptions):
    
    def initialize(self):
        super().initialize()

        # Hyperparameters
        self.parser.add_argument('--batch_per_gpu', type=str, default=8)
        self.parser.add_argument('--max_step', type=str, default=400000)
        self.parser.add_argument('--same_prob', type=float, default=0.2)

        # Dataset
        self.parser.add_argument('--train_dataset_root_list', type=list, \
            default=[
                '/home/compu/dataset/CelebHQ',
                '/home/compu/dataset/kface_wild_1024',
                '/home/compu/dataset/ffhq16k'
            ])

        # Learning rate
        self.parser.add_argument('--lr_G', type=str, default=1e-4)
        self.parser.add_argument('--lr_D', type=str, default=4e-5)
        self.parser.add_argument('--beta1', type=float, default=0)
        self.parser.add_argument('--beta2', type=float, default=0.999)

        # Weight
        self.parser.add_argument('--W_adv', type=float, default=1)
        self.parser.add_argument('--W_id', type=float, default=5)
        self.parser.add_argument('--W_recon', type=float, default=10)
        self.parser.add_argument('--W_cycle', type=float, default=0)
        self.parser.add_argument('--W_lpips', type=float, default=0)
        self.parser.add_argument('--W_attr', type=float, default=10)