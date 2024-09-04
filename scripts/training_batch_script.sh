#!/bin/bash

# Configure the resources required
#SBATCH -p batch # partition (this is the queue your job will be added to)
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00 # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1 # generic resource required (here requires 1 GPU)
#SBATCH --mem=16GB # specify memory required per node (here set to 8 GB)

module use /hpcfs/apps/icl/modules/all/
module load Python/3.9.6-GCCcore-11.2.0
module load CUDAcompat/12.2-535.161.08
source activate predict_haq_env

which python

python /hpcfs/users/a1628977/predict_haq/bin/train_haq.py $MODCONFIG
