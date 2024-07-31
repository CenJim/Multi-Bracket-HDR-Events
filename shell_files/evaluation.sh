#!/bin/bash
source ~/.bashrc
echo $CUDA_HOME
nvidia-smi
conda activate snnrec
echo $CUDA_HOME
python ../evaluation.py
