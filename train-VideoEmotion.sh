#!/bin/bash
mkdir log
for (( i = 1; i<=10; i=i+1 )); do
python3 train.py --feapath=./data/VideoEmotion/tsnv3_rgb_pool/,./data/VideoEmotion/tsnv3_flow_pool,./data/VideoEmotion/audioset/ --dataset=VideoEmotion --numclasses=8 --split=${i} --gpu=0 --seqlen=50 --hidden1=800 --hidden2=800 --numlayers=3 --batchsize=8 --numepochs=30 --learate=0.00001 --l2=1e-6 2>&1 |tee log/VE8_split=${i}.log
done
