#!/bin/bash
python3 test.py --feapath=./data/VideoEmotion/tsnv3_rgb_pool/,./data/VideoEmotion/tsnv3_flow_pool,./data/VideoEmotion/audioset --dataset='VideoEmotion' --numclasses=8 --numepochs=30