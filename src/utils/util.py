import importlib
import torch
from torch import distributed as dist
import os
import numpy as np

def init_config(args):
    cfg_map={
        "s": "small",
        "b": "base",
        "l": "large",
        "h": "huge",
    }
    model = args.model
    scale = cfg_map[args.model_size]
    dataset = args.dataset

    cfg = importlib.import_module(f"config.{model}_{scale}_{dataset}")
    for key, value in cfg.__dict__.items():
        if not key.startswith("__") and not hasattr(args, key):
            setattr(args, key, value)

    return args

def init_env(args):
    torch.manual_seed(args.seed)

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    
    if args.autoscale_lr:
        args.optimizer['lr'] = args.optimizer['lr'] * len(args.gpu_ids) / 8

    if args.distributed:
        init_dist()

def init_dist():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
