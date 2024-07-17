#!/bin/bash
source ~/.bashrc
conda activate snnrec 
python ../train.py -network EHDR_network -path_to_pretrain_models pretrained_models/EHDR.pth \
-path_to_root_files "/home/s2491540/dataset/DSEC/train_sequences" \
-height 469 -width 640

