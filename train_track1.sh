#!/bin/bash

echo "Start to train the model...."

name="track1"
dataroot="/mnt/disk10T/dataset/Zurich-RAW-to-DSLR/"
# the path of dataset file

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


python train.py \
    --dataset_name zth    --model tbsr           --name $name           --dataroot $dataroot   \
    --niter 400           --lr_decay_iters 200   --patch_size 256   \
    --batch_size 32       --print_freq 100       --calc_metrics True    --load_path ./ckpt/track1_model/EBSR_model_1.pth \
    --gpu_ids 0,1,2,3,4,5,6,7                    --save_imgs True       -j 4       --lr 2e-4 | tee $LOG    