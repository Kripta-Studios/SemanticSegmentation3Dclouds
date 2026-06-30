#!/bin/bash

# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                              #
# BRIEF: Bash template to handle a k-folding experiment with VL3D++            #
# ---------------------------------------------------------------------------- #


# ---   METHODS   --- #
# ------------------- #
# Recopilate point clouds
function find_pclouds() {
    pclouds=()
    for las in $(find ${DATADIR} | grep "\.las"); do
        pclouds+=("${las}")
    done
}

# Random shuffle the point clouds
function shuffle_pclouds() {
    pclouds=($(shuf -e "${pclouds[@]}"))
}

# Determine the number of folds (KFOLDS) for current TRAINING_RATIO
function determine_kfolds() {
    validation_ratio=$(echo "1.0 - ${TRAINING_RATIO}" | bc)
    kfolds=$(echo "1.0 / ${validation_ratio}" | bc)
    fold=0
}

# Prepare the k-folding experiment
function prepare_experiment() {
    outdir="${OUTDIR}/fold_${fold}"
    mkdir -p "${outdir}"
    specdir="${outdir}"'/spec'
    mkdir -p "${specdir}"
}

# Split point clouds into training and validation splits for current fold
function train_val_split() {
    pclouds_train=()
    pclouds_val=()
    num_pclouds=${#pclouds[@]}
    # Start index of validation split (inclusive)
    aidx=$(echo "${validation_ratio} * ${fold} * ${num_pclouds}" | bc)
    aidx=${aidx%.*}
    # End index of validation split (exclusive)
    bidx=$(echo "${validation_ratio} * (${fold} + 1) * ${num_pclouds}" | bc)
    bidx=${bidx%.*}
    for(( i=0 ; i < ${aidx} ; ++i )); do
        pclouds_train+=("${pclouds[$i]}")
    done
    for(( i=${aidx} ; i < ${bidx} ; ++i )); do
        pclouds_val+=("${pclouds[$i]}")
    done
    for(( i=${bidx} ; i < ${num_pclouds} ; ++i )); do
        pclouds_train+=("${pclouds[$i]}")
    done
}
