#!/bin/sh

#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -g gq54
#PJM -j
#PJM -N al11egnn
#-------Programexecution-------#
#module load fj
#module load nvidia
#module load cuda
#module load gcc
#module load fjmpi

source /work/01/gq54/p23002/.bashrc
conda activate megnetclone
cd /work/01/gq54/p23002/guyuhan/egnn
python egnn_inference11.py >> al11_egnn_rm.txt
