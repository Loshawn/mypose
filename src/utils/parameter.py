import argparse
import os
import sys

dir_root = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(dir_root)

parser = argparse.ArgumentParser()

# System
parser.add_argument("--dir_root",
                    type=str,
                    default=dir_root,
                    help="root directory of the project")

parser.add_argument("--gpu_ids", type=str, default='0', help="index of GPUs")

parser.add_argument("--distributed", action='store_true', help="index of GPUs")

parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--use_amp',
                    action='store_false',
                    help='Use mixed precision training (AMP)')
parser.add_argument('--autoscale_lr',
                    action='store_false',
                    help='Use autoscale lr')
parser.add_argument('--cudnn_benchmark',
                    action='store_false',
                    help='Use cudnn benchmark')

parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='Number of data loading workers (threads)')
parser.add_argument('--pin_memory',
                    action='store_false',
                    help='Use pin memory for faster data transfer to GPU')

parser.add_argument("--seed", type=int, default=0, help="random seed")


# Data
parser.add_argument("--dataset",
                    type=str,
                    default='wholebody',
                    help="dataset name")

parser.add_argument('--shift',
                    type=float,
                    default=0.1,
                    help='when not using random crop'
                    'apply shift augmentation.')

parser.add_argument('--scale',
                    type=float,
                    default=0.4,
                    help='when not using random crop'
                    'apply scale augmentation.')

parser.add_argument('--rotate',
                    type=float,
                    default=0,
                    help='when not using random crop'
                    'apply rotation augmentation.')

parser.add_argument('--flip',
                    type=float,
                    default=0.5,
                    help='probability of applying flip augmentation.')

# Model
parser.add_argument('--model', type=str, default='ViTPose', help='model name')

parser.add_argument('--model_size',
                    choices=['s', 'b', 'l', 'h'],
                    default='s',
                    help='Select the model scale')

parser.add_argument('--pretrain',
                    type=str,
                    default='.',
                    help='pre trian model path')

# Train
parser.add_argument('--save',
                    type=str,
                    default='ViTPose',
                    help='file name to save')

parser.add_argument('--resume',
                    type=str,
                    default='.',
                    help='file name to resume')

parser.add_argument('--epochs', type=int, default=210, help='number of epochs')

# Inference
parser.add_argument('--image_path',
                    type=str,
                    default='.',
                    help='image path to inference')

# Log

args = parser.parse_args()
