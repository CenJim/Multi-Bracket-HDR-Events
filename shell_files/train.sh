#!/bin/bash
source ~/.bashrc
conda activate snnrec 
python ../train.py -network EHDR_network \
-path_to_pretrain_models /home/s2491540/Pythonproj/Multi-Bracket-HDR-Events/pretrained_models/1.2-trained_on_smith_welding/EHDR_model_epoch_final.pth \
-path_to_root_files "/home/s2491540/dataset/HDM_HDR/sequences" \
-height 1060 -width 1900 -hdr_flag True

