#!/bin/bash

# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                              #
# BRIEF: Experiment with the randomized airplane dataset                       #
# ---------------------------------------------------------------------------- #




# ---  CONFIGURATION CONSTANTS  --- #
# --------------------------------- #
# VL3D framework components
VL3DENV="${HOME}/git/vl3d/venv/bin/activate"
VL3DPY="${HOME}/git/vl3d/vl3d.py"
VL3DPIPE='/tmp/vl3d_airplane_stratkfold.json'
# Directory where the point clouds of the dataset are located
DATADIR='/ext4/lidar_data/shapenet/pcloud/Airplane'
# Output directory for the pipeline
OUTDIR='/ext4/lidar_data/shapenet/vl3d/out/airplane/order2_only/'
# Output file for point cloud with mined features
OUTFEATS='training_airplane.las'
# Output file to store the trained model
OUTMODEL='rf_airplane.pipe'
# How many separation between objects
INTERNODE_SEPARATION=5.0
# Ratio of data for training
TRAINING_RATIO=0.8
# Class names (in JSON pipeline format)
CLASS_NAMES='["wing", "body", "tail" ,"engine", "other"]'
# Class-wise sampling for training distribution
TRAINING_DISTRIBUTION='[2000000, 2000000, 2000000, 2000000, 0]'





# ---  SOURCES  --- #
# ----------------- #
source experiment_template.sh
source "${VL3DENV}"




# ---   MAIN   --- #
# ---------------- #
# Recopilate airplane point clouds
find_pclouds

# Random shuffle the point clouds
shuffle_pclouds

# K-folding loop
determine_kfolds
echo "Running k-fold for k=${kfolds} with a dataset of ${#pclouds[@]} point clouds ..."
for(( it=0 ; it < ${kfolds} ; ++it )); do
    # Split point clouds into training and validation splits
    train_val_split
    # Prepare experiment
    prepare_experiment
    echo "Results will be exported to \"${outdir}\""
    # Craft VL3D++ training pipeline
    echo "Crafting ${fold}-fold pipeline with ${#pclouds_train[@]} training point clouds and ${#pclouds_val[@]} validation point clouds with validation split [${aidx}, ${bidx}) ..."
    source training_pipeline_template_second_order_only.sh
    # Launch the VL3D++ training pipeline
    python3 ${VL3DPY} --pipeline "${vl3d_training_pipeline}"
    # Craft VL3D++ validation pipeline
    source validation_pipeline_template.sh
    # Launch the VL3D++ validation pipeline
    python3 ${VL3DPY} --pipeline "${vl3d_validation_pipeline}"
    # Prepare next iteration
    fold=$(( ${fold} + 1 ))  # Move to next fold
done



