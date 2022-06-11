#!/bin/bash
echo "Start to test the model...."

name="track2_model"
dataroot="/Data/dataset/Burst_SR/realworld_test_2022"
device="0"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


python test.py \
    --dataset_name realtest  --model tbsr         --name $name         --dataroot $dataroot \
    --load_iter 1            --save_imgs True     --gpu_ids $device    --self_ensemble False 
