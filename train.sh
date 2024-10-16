export CUDA_VISIBLE_DEVICES=0 
nohup python src/train.py --save ViTPose --model ViTPose --model_size s > exp/out-s.out & 