# export CUDA_VISIBLE_DEVICES=4,5,6,7
# nohup torchrun --nproc_per_node=4 src/train.py --save debug-b --distributed --model_size b > exp/out_dist-b.out & 

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup torchrun --nproc_per_node=4 --master_port=50000 src/train.py --save debug-l --distributed --model_size l > exp/out_dist-l.out & 