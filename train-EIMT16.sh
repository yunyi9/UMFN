#!/bin/bash
mkdir log
python3 train.py --feapath=./data/EIMT16/me16tsn_v3_rgb,./data/EIMT16/me16tsn_v3_flow,./data/EIMT16/audioset --dataset=EIMT16 --numclasses=1 --gpu=0 --dtype=arousal --seqlen=18 --batchsize=32 --numepochs=17 --learate=0.00001 --l2=0.01 2>&1 |tee log/EIMT16_arousal.log

python3 train.py --feapath=./data/EIMT16/me16tsn_v3_rgb,./data/EIMT16/me16tsn_v3_flow,./data/EIMT16/audioset --dataset=EIMT16 --numclasses=1 --gpu=0 --dtype=valence --seqlen=18 --batchsize=32 --numepochs=46 --learate=0.00001 --l2=0.01 2>&1 |tee log/EIMT16_valence.log 
