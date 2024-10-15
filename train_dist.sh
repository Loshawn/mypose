export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup torchrun --nproc_per_node=4 src/train.py --distributed   > out_dist.out & 