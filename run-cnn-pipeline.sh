#!/bin/bash

mkdir logs

echo "=== run baseline to prepare data folder and check that things work fine ==="
python task3_gmm_baseline.py &> logs/task3_gmm_baseline.log

echo "=== prepare 16kHz data ==="
src/make_downsample.sh 

echo "=== train CNN model using the original train data ==="
cp -rf task3_cnn.init.yaml task3_cnn.yaml
python task3_cnn.py &> logs/task3_cnn.init.log 

echo "=== train CNN using speed perturbed data ==="
src/make_speed.sh
cp -rf task3_cnn.augm.yaml task3_cnn.yaml
python task3_cnn.py &> logs/task3_cnn.augm.log


