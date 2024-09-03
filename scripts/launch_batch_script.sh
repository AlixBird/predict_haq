#!/bin/bash

for SEED in {14021,21234,13412}; do
for IMAGE_SIZE in {500,1000}; do
for MAX_EPOCHS in {200,500}; do
for LEARNING_RATE in {1e-4,1e-3}; do

export MODCONFIG="
--csvpath=/hpcfs/users/a1628977/data/dataframes/xrays_train.csv
--imagepath=/hpcfs/users/a1628977/data/xray_images
--checkpointpath=/hpcfs/users/a1628977/predict_haq/checkpoints
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--max_epochs=${MAX_EPOCHS}
--learning_rate=${LEARNING_RATE}
"

sbatch training_batch_script.sh

done;
done;
done; 
done; 