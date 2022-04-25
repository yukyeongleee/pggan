import torch
import os

        
def load_checkpoint(args, name):    
    if args.ckpt_step is None:
        ckpt_step = 'latest'
    else:
        ckpt_step = args.ckpt_step

    ckpt_path = f'{args.save_root}/{args.ckpt_id}/ckpt/{name}_{ckpt_step}.pt'

    try:
        ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cuda'))
        return ckpt_dict
    except:
        if args.isMaster:
            print(f"Failed to load checkpoint of {name}.")
        return 0


def save_checkpoint(model, optimizer, name, ckpt_dict = {}):
    
    ckpt_dict['model'] = model.state_dict()
    ckpt_dict['optimizer'] = optimizer.state_dict()

    dir_path = f'{ckpt_dict["args"]["save_root"]}/{ckpt_dict["args"]["run_id"]}/ckpt'
    os.makedirs(dir_path, exist_ok=True)
    
    ckpt_path = dir_path + f'/{name}_{ckpt_dict["global_step"]}.pt'
    torch.save(ckpt_dict, ckpt_path)

    latest_ckpt_path = dir_path + f'/{name}_latest.pt'
    torch.save(ckpt_dict, latest_ckpt_path)
        