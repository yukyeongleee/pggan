import torch
import wandb
import os
import sys
from lib.options import BaseOptions
from lib.model_loader import CreateModel

sys.path.append("./")
sys.path.append("./submodel/")


def train(gpu, args): 
    torch.cuda.set_device(gpu)
    model, args, step = CreateModel(gpu, args)
    load_ckpt = step != 0 # If True, scale and alpha jump related variables need not to be defined

    # Initialize wandb to gather and display loss on dashboard 
    if args.isMaster and args.use_wandb:
        wandb.init(project=args.model_id, name=args.run_id)

    # Training loop
    global_step = step if step else 0
    args.max_step = min(sum(args.max_step_at_scale), args.max_step)

    if not load_ckpt:
        model.scale_index = 0
        model.scale_jump_step = args.max_step_at_scale[0] # 첫 번째 jump_step 설정
        model.alpha_jump_step = args.alpha_jump_start[0] # 첫 번째 jump_step 설정
    
    while global_step < args.max_step:

        # scale 이 바뀔 때
        if global_step == model.scale_jump_step:
            if model.scale_index < args.max_depths-1:
                model.scale_index += 1
                model.scale_jump_step += args.max_step_at_scale[model.scale_index]

                print(f"\nNOW global step is {global_step}")
                print(f"scale index is updated to {model.scale_index}")
                print(f"next scale jump step is {model.scale_jump_step}")

                # initialize parameters related to the alpha
                if not load_ckpt: 
                    model.G.alpha = 0
                    model.D.alpha = 0
                    model.alpha_index = 0
                    model.alpha_jump_step = global_step + args.alpha_jump_start[model.scale_index]
                
                alpha_jump_value = 1/args.alpha_jump_Ntimes[model.scale_index]

                print(f"alpha index is initialized to 0")
                print(f"next alpha jump step is set to {model.alpha_jump_step}")
                print(f"alpha jump value is set to {alpha_jump_value}")

                # add a block to net G and net D
                model.G.add_block(args.depths[model.scale_index])
                model.D.add_block(args.depths[model.scale_index])
                model.G.cuda()
                model.D.cuda()

                # dataset and data iterator
                model.set_dataset()
                model.set_data_iterator()

        # alpha 가 바뀔 때 (Linear mode)
        if global_step == model.alpha_jump_step:
            if model.scale_index > 0 and model.alpha_index < args.alpha_jump_Ntimes[model.scale_index]:
                alpha_jump_value = 1/args.alpha_jump_Ntimes[model.scale_index] # Required when the loaded ckpt is from the scale_jump_step
                model.G.alpha += alpha_jump_value
                model.D.alpha += alpha_jump_value
                model.alpha_jump_step = global_step + args.alpha_jump_interval[model.scale_index]
                model.alpha_index += 1

                print(f"\nNOW global step is {global_step}")
                print(f"alpha index is updated to {model.alpha_index}")
                print(f"next alpha jump step is {model.alpha_jump_step}")
                print(f"alpha is now {model.G.alpha}")

        result = model.train_step()

        if args.isMaster:
            # Save and print loss
            if global_step % args.loss_cycle == 0:
                if args.use_wandb:
                    wandb.log(model.loss_collector.loss_dict)
                model.loss_collector.print_loss(global_step)

            # Save image
            if global_step % args.test_cycle == 0:
                model.save_image(result, global_step)

                if args.validation:
                    model.validation(global_step) 

            # Save checkpoint parameters 
            if global_step % args.ckpt_cycle == 0:
                model.save_checkpoint(global_step)

        global_step += 1


if __name__ == "__main__":
    args = BaseOptions().parse()
    os.makedirs(args.save_root, exist_ok=True)

    # Set up multi-GPU training
    if args.use_mGPU:  
        torch.multiprocessing.spawn(train, nprocs=args.gpu_num, args=(args, ))

    # Set up single GPU training
    else:
        train(args.gpu_id, args)
