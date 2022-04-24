import torch
import wandb
import os
import sys
from lib.options import BaseOptions
from lib.model_loader import CreateModel
from lib.config import Config

sys.path.append("./")
sys.path.append("./submodel/")

def train(gpu, args): 
    # convert dictionary to class
    args = Config(args)

    # This line must be placed before the "CreateModel" method because it includes load_checkpoint.
    # If the previous checkpoint is loaded, args.ckpt_id and args.ckpt_step will be overwritten.
    load_ckpt = args.ckpt_id is not None # If True, scale- and alpha-related-variables are not required to be defined

    torch.cuda.set_device(gpu)
    model, args = CreateModel(gpu, args)

    # Initialize wandb to gather and display loss on dashboard 
    if args.isMaster and args.use_wandb:
        wandb.init(project=args.model_id, name=args.run_id)

    # set scale- and alpha-related-variables
    if not load_ckpt:
        model.alpha = 0
        model.alpha_index = 0
        model.scale_index = 0
        model.alpha_jump_value = 0
        model.next_scale_jump_step = args.max_step_at_scale[0] # 첫 번째 scale jump_step 설정
        model.next_alpha_jump_step = args.alpha_jump_start[0] # 첫 번째 alpha jump_step 설정
    
    # Training loop
    global_step = model.global_step if load_ckpt else 0
    args.max_step = min(sum(args.max_step_at_scale), args.max_step)

    while global_step < args.max_step:

        """
        Fixes #2.
        Alpha-related-lines are moved to the model.check_alpha method.
        """
        model.check_jump(global_step)
        intermediate_images = model.train_step()

        if args.isMaster:
            # Save and print loss
            if global_step % args.loss_cycle == 0:
                if args.use_wandb:
                    wandb.log(model.loss_collector.loss_dict)
                model.loss_collector.print_loss(global_step)

            # Save image
            if global_step % args.test_cycle == 0:
                model.save_image(intermediate_images, global_step)

                if args.use_validation:
                    model.validation(global_step)

            # Save checkpoint parameters 
            if global_step % args.ckpt_cycle == 0:
                model.save_checkpoint(global_step)

        global_step += 1


if __name__ == "__main__":

    """
    Fixes #1
    Argparser is relaced with Configs.
    config.py is added in ./lib directory.
    """
    # load config
    config_path = "configs.yaml"
    args = Config.from_yaml(config_path)
    
    # update configs
    args.run_id = sys.argv[1] # command line: python train.py {run_id}
    args.gpu_num = torch.cuda.device_count()
    
    # save config
    os.makedirs(f"{args.save_root}/{args.run_id}", exist_ok=True)
    args.save_yaml(config_path)

    # Set up multi-GPU training
    if args.use_mGPU:  
        torch.multiprocessing.spawn(train, nprocs=args.gpu_num, args=(args.__dict__, ))

    # Set up single GPU training
    else:
        train(gpu=0, args=args.__dict__)
