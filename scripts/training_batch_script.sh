# !/bin/bash -l

# SBATCH --nodes=1
# SBATCH --partition=skylake
# SBATCH --ntasks-per-node=8
# SBATCH --gpus-per-node=2
# SBATCH --time=0-24:00:00 ## Time format: DD-HH:MM:SS

source activate predict_haq_env

which python

python ../src/predict_haq/train.py
