#!/bin/bash
#SBATCH --nodelist=w7128
source ~/.bashrc
conda activate snnrec
python ../utils/preprocessing.py
