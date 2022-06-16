#!/bin/bash
python3 test.py --feapath='./data/EIMT16/me16tsn_v3_rgb,./data/EIMT16/me16tsn_v3_flow,./data/EIMT16/audioset' --dataset=EIMT16 --numclasses=1 --dtype=arousal --seqlen=18 --numepochs=17

python3 test.py --feapath='./data/EIMT16/me16tsn_v3_rgb,./data/EIMT16/me16tsn_v3_flow,./data/EIMT16/audioset' --dataset=EIMT16 --numclasses=1 --dtype=valence --seqlen=18 --numepochs=46
