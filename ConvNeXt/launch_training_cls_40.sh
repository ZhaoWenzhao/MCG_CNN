#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
#export NGPUS=8 
#export OMP_NUM_THREADS=18


python -m torch.distributed.launch --nproc_per_node=8 main0wt_fb.py \
--model convnext_small --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /home/bbb/imagenet/imagenet40/ \
--output_dir ./results40_small_fb 
