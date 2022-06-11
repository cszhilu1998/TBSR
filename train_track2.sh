#!/bin/bash

echo "Start to train the model...."

name="track2"
dataroot="/mnt/disk10T/dataset/Burst_SR/"
# the path of dataset file

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


python train.py \
    --dataset_name burstsr       --model tbsrflow       --name $name           --dataroot $dataroot   \
    --niter 400                  --lr_decay_iters 200   --patch_size 480   \
    --batch_size 8               --print_freq 100       --calc_metrics True    --load_path ./ckpt/track2_model/EBSR_model_1.pth  \
    --gpu_ids 0,1,2,3,4,5,6,7    --save_imgs True       -j 4  --lr 1e-4  | tee $LOG    

