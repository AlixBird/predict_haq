#!/bin/bash

for IMAGE_SIZE in {800,1200}; do
MAX_EPOCHS=500
for LEARNING_RATE in {1e-4,1e-3}; do
for OUTCOME in {"HAQ","Future_HAQ"}; do
for SEED in {413,14213,954}; do

export MODCONFIG="
--csvpath=/hpcfs/users/a1628977/data/dataframes/
--imagepath=/hpcfs/users/a1628977/data/xray_images
--checkpointpath=/hpcfs/users/a1628977/predict_haq/checkpoints
--image_size=${IMAGE_SIZE}
--max_epochs=${MAX_EPOCHS}
--learning_rate=${LEARNING_RATE}
--outcome=${OUTCOME}
--seed=${SEED}"

sbatch training_batch_script.sh

done; 
done;
done;
done;