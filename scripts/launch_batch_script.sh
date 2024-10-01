#!/bin/bash

#for IMAGE_SIZE in {900,1200}; do
#for LEARNING_RATE in {1e-4,1e-3,1e-5}; do
#for IMAGE_PATH in {"/hpcfs/users/a1628977/data/concatenatedjoints","/hpcfs/users/a1628977/data/xray_images"}; do
for OUTCOME in {"HAQ","Future_HAQ"}; do
for HANDSORFEET in {"Hands","Feet"}; do
for SEED in {443,1231,12341,1245,55}; do
IMAGE_SIZE=1200
MAX_EPOCHS=300
IMAGE_PATH="/hpcfs/users/a1628977/data/concatenatedjoints"
LEARNING_RATE=0.0001


export MODCONFIG="
--csvpath=/hpcfs/users/a1628977/data/dataframes
--imagepath=${IMAGE_PATH}
--checkpointpath=/hpcfs/users/a1628977/predict_haq/checkpoints
--image_size=${IMAGE_SIZE}
--max_epochs=${MAX_EPOCHS}
--learning_rate=${LEARNING_RATE}
--outcome=${OUTCOME}
--seed=${SEED}
--handsorfeet=${HANDSORFEET}"

sbatch training_batch_script.sh

done; 
done; 
done; 