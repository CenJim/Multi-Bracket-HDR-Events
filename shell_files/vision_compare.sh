#!/bin/bash
#SBATCH --nodelist=w7128
source ~/.bashrc
conda activate snnrec
python ../utils/vision_quality_compare.py