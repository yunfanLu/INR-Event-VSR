#!/bin/sh
#BSUB -J 'batch_unzip'
#BSUB -n 4
#BSUB -q normal
#BSUB -o sys_logs/out.%J.log
#BSUB -e sys_logs/err.%J.log

source /hpc/jhinno/unischeduler/exec/unisched

module load anaconda3			
module load cuda-11.1
source activate
conda activate nerf

cd /hpc/users/CONNECT/zipengwang/projects/EG-VSR/
python tools/extract_color_event_bag.py
