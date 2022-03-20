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

    # Initialize wandb to gather and display loss on dashboard 
    if args.isMaster and args.use_wandb:
        wandb.init(project=args.model_id, name=args.run_id)

    # Training loop
    global_step = step if step else 0
    scale_jump_step = args.max_step_at_scale[0] # 첫 번째 jump_step 설정
    alpha_jump_step = args.alpha_jump_start[0] # 첫 번째 jump_step 설정
    while global_step < args.max_step:

        """
        comment #3

        - pytorch_GAN_zoo/models/trainer/progressive_gan_trainer.py
            - line 211: 
                for scale in range(self.startScale, n_scales):
                    ...
                    
        scale 이 한 단계 높아질 때마다 레이어들이 추가되고, alpha 값이 조정됩니다. 
        코드가 복잡하게 구현되어 있는데 쉽게 풀어주겠습니다.
        """

        # scale 이 바뀔 때
        if global_step == scale_jump_step:
            if args.scale_index < args.max_depths-1:
                args.scale_index += 1
                scale_jump_step += args.max_step_at_scale[args.scale_index]

                print(f"\nNOW global step is {global_step}")
                print(f"scale index is updated to {args.scale_index}")
                print(f"next scale jump step is {scale_jump_step}")

                # initialize parameters related to the alpha
                model.G.alpha = 0
                args.alpha_index = 0
                alpha_jump_step = global_step + args.alpha_jump_start[args.scale_index]
                alpha_jump_value = 1/args.alpha_jump_Ntimes[args.scale_index]

                print(f"alpha index is initialized to 0")
                print(f"next alpha jump step is set to {alpha_jump_step}")
                print(f"alpha jump value is set to {alpha_jump_value}")

                # add a block to net G and net D
                model.G.add_block(args.depths[args.scale_index])
                model.D.add_block(args.depths[args.scale_index])
                model.G.cuda()
                model.D.cuda()

                # dataset and data iterator
                model.set_dataset()
                model.set_data_iterator()

        # alpha 가 바뀔 때
        if global_step == alpha_jump_step:
            if args.scale_index > 0 and args.alpha_index < args.alpha_jump_Ntimes[args.scale_index]:
                model.G.alpha += alpha_jump_value
                alpha_jump_step = global_step + args.alpha_jump_interval[args.scale_index]
                args.alpha_index += 1

                print(f"\nNOW global step is {global_step}")
                print(f"alpha index is updated to {args.alpha_index}")
                print(f"next alpha jump step is {alpha_jump_step}")
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

                if args.valid_dataset_root:
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
