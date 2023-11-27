export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export OMP_NUM_THREADS=1

export TORCH_DISTRIBUTED_DEBUG=INFO

# base
python -m torch.distributed.launch --nproc_per_node=16 --nnodes=1 --use_env main.py \
            -c config/Pretrain/DINO_fast_itpn_o365_usedn_12e_bs8.py \
            --dataset_file o365 \
            --amp \
            --pt \
            --strong_aug \
            --coco_path /path/to/O365/ \
            --output_dir ./output_dir/EinsPT_itpn_base/ \
            --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 \

# large
# python -m torch.distributed.launch --nproc_per_node=16 --nnodes=1 --use_env main.py \
#             -c config/Pretrain/DINO_fast_itpn_large_o365_usedn_12e_bs6.py \
#             --dataset_file o365 \
#             --amp \
#             --pt \
#             --strong_aug \
#             --coco_path /path/to/O365/ \
#             --output_dir ./output_dir/EinsPT_itpn_large/ \
#             --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 \