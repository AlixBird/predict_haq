#!/bin/bash

#for IMAGE_SIZE in {900,1200}; do
#for LEARNING_RATE in {1e-4,1e-3,1e-5}; do
#for OUTCOME in {"HAQ","HAQ_change"}; do
for HANDSORFEET in {"Hands","Feet"}; do

OUTCOME="HAQ_change"
SEED=2308
IMAGE_SIZE=1200
MAX_EPOCHS=500
IMAGE_PATH="/hpcfs/users/a1628977/data/concatenatedjoints"
LEARNING_RATE=0.0001

export MODCONFIG="
--csvpath=/hpcfs/users/a1628977/data/dataframes
--imagepath=${IMAGE_PATH}
--checkpointpath=/hpcfs/users/a1628977/predict_haq/checkpoints
--figurepath=/hpcfs/users/a1628977/predict_haq/figures
--image_size=${IMAGE_SIZE}
--max_epochs=${MAX_EPOCHS}
--learning_rate=${LEARNING_RATE}
--outcome=${OUTCOME}
--seed=${SEED}
--handsorfeet=${HANDSORFEET}
--train"


sbatch training_batch_script.sh

done; 
