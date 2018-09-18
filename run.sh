#!/bin/bash

# go to folder
# cd Generating_a_Training_Set

# run Step 2
# python Step2_ConvertingLabels2DataFrame.py
# python Step3_CheckLabels.py
# python Step4_GenerateTrainingFileFromLabelledData.py

# cd ../pose-tensorflow/models/pretrained
# ./download.sh

# cd ../..
# cp -r YOURexperimentNameTheDate-trainset95shuffle1 ../pose-tensorflow/models/
# cp -r UnaugmentedDataSet_YOURexperimentNameTheDate ../pose-tensorflow/models/

cd pose-tensorflow/models/reachingJan30-trainset95shuffle1/train
TF_CUDNN_USE_AUTOTUNE=0 CUDA_VISIBLE_DEVICES=0 python3 ../../../train.py
