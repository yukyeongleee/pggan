import numpy as np
import torch

def CreateModel(gpu, args):
    args.isMaster = gpu == 0

    # Create model
    model = None
    if args.model_id == 'pggan':
        from pggan.model import ProgressiveGAN
        model = ProgressiveGAN(args, gpu)
    
    model.initialize_models()
    if args.use_mGPU:
        model.set_multi_GPU()
    
    model.set_optimizers()
    model.set_dataset()
    model.set_data_iterator()
    model.set_loss_collector()
    model.set_validation()
    model.RandomGenerator = np.random.RandomState(42)

    # Loading Checkpoints
    # Argments overlapping issue
    run_id = args.run_id
    dataset_root_list = args.dataset_root_list
    if args.ckpt_id is not None:
        model.load_checkpoint()
    args.run_id = run_id
    args.dataset_root_list = dataset_root_list
    args.isMaster = gpu == 0

    
    if args.isMaster:
        print(f'model {args.model_id} has created')
        
    return model, args