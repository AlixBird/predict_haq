#!/bin/bash



for SEED in {1,2,3}; do

export MODCONFIG="--seed=${SEED}" # the script needs to take --seed as an argument

sbatch training_batch_script.sh
#sh training_batch_script.sh

done;