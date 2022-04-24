import numpy as np
import torch

def CreateModel(gpu, args):

    model = None
    if args.model_id == 'pggan':
        from pggan.model import ProgressiveGAN
        model = ProgressiveGAN(args, gpu)
        
    args.isMaster = gpu == 0
    model.RandomGenerator = np.random.RandomState(42)
    model.initialize_models()
    model.set_dataset()

    if args.use_mGPU:
        model.set_multi_GPU()

    model.set_data_iterator()
    model.set_validation()
    model.set_optimizers()
    
    if args.ckpt_id is not None:
        model.load_checkpoint()
    model.set_loss_collector()

    if args.isMaster:
        print(f'model {args.model_id} has created')
        
    return model, args