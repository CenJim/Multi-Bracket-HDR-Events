#!/bin/bash
#SBATCH --nodelist=w7128
source ~/.bashrc
conda activate snnrec 
python ../train.py -network EHDR_network \
-path_to_pretrain_models /home/s2491540/Pythonproj/Multi-Bracket-HDR-Events/pretrained_models/2.1-trained_on_8_sequences/EHDR_model_epoch_final.pth \
-path_to_root_files "/localdisk/home/s2491540/HDM_HDR/sequences" \
-height 1060 -width 1900 -hdr_flag True

