#!/bin/bash
echo "Start to test the model...."

name="track1_model"
dataroot="/Data/dataset/Burst_SR/synburst_test_2022"
device="0"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


python test.py \
    --dataset_name syntest   --model tbsr          --name $name         --dataroot $dataroot \
    --load_iter 1            --save_imgs True      --gpu_ids $device    --self_ensemble True 
