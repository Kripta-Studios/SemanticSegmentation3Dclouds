#!/bin/bash

# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Split each ShapeNet dataset into training and validation to
#        generate a fixed split dataset for straightforward comparisons
# ---------------------------------------------------------------------------- #


# ---  CONSTANTS  --- #
# ------------------- 3
PCLOUD_DIR='/ext4/lidar_data/shapenet/pcloud/'
PCLOUD_SPLIT_DIR='/ext4/lidar_data/shapenet/pcloud_split/'
INTERNODE_SEPARATION=5


# ---   MAIN   --- #
# ---------------- #
# For each dataset
for dataset in $(ls ${PCLOUD_DIR}); do
    dataset_dir="${PCLOUD_DIR}/${dataset}"
    # Get the point clouds
    pclouds=()
    for pcloud in $(ls "${dataset_dir}" | grep -i "\.las"); do
        pclouds+=("${pcloud}")
    done
    # Split first 80% and last 20%
    num_pclouds=${#pclouds[@]}
    cut_idx=$(echo "${num_pclouds} * 0.8" | bc)
    cut_idx=${cut_idx%.*}
    # Generate training split
    train_dir="${PCLOUD_SPLIT_DIR}/${dataset}/train"
    echo "Generatining training split at \"${train_dir}\" ..."
    mkdir -p "${train_dir}"
    nodes_per_line=$(echo "sqrt(${cut_idx})" | bc)
    train_template_path="${train_dir}/template.json"
    cat << EOF > "${train_template_path}"
{
    "in_pcloud_concat": [
EOF
    for (( i=0 ; i < ${cut_idx} ; ++i )); do
        train_pcloud="${train_dir}/${pclouds[$i]}"
        cp "${dataset_dir}/${pclouds[$i]}" "${train_pcloud}"
        nodex=$(echo "$i / ${nodes_per_line}" | bc)
        nodex=$(( ${INTERNODE_SEPARATION} * ${nodex} ))
        nodey=$(echo "$i % ${nodes_per_line}" | bc)
        nodey=$(( ${INTERNODE_SEPARATION} * ${nodey} ))
        echo -e "\t\t{" >> "${train_template_path}"
        echo -e "\t\t\t"'"in_pcloud": "'"${train_pcloud}"'",' >> "${train_template_path}"
        echo -e "\t\t\t"'"conditions": null,' >> "${train_template_path}"
        echo -e "\t\t\t\"center\": [$nodex, $nodey, 0]" >> "${train_template_path}"
        if (( ${i} < (( ${cut_idx} - 1 )) )); then
            echo -e "\t\t}," >> "${train_template_path}"
        else
            echo -e "\t\t}" >> "${train_template_path}"
        fi
    done
    cat << EOF >> "${train_template_path}"
    ],
EOF
    # Generate validation split
    val_dir="${PCLOUD_SPLIT_DIR}/${dataset}/val"
    echo "Generatining validation split at \"${val_dir}\" ..."
    mkdir -p "${val_dir}"
    nodes_per_line=$(echo "sqrt(${cut_idx})" | bc)
    val_template_path="${val_dir}/template.json"
    cat << EOF > "${val_template_path}"
{
    "in_pcloud_concat": [
EOF
    for (( i=${cut_idx} ; i < ${num_pclouds} ; ++i )); do
        val_pcloud="${val_dir}/${pclouds[$i]}"
        cp "${dataset_dir}/${pclouds[$i]}" "${val_pcloud}"
        nodex=$(echo "$i / ${nodes_per_line}" | bc)
        nodex=$(( ${INTERNODE_SEPARATION} * ${nodex} ))
        nodey=$(echo "$i % ${nodes_per_line}" | bc)
        nodey=$(( ${INTERNODE_SEPARATION} * ${nodey} ))
        echo -e "\t\t{" >> "${val_template_path}"
        echo -e "\t\t\t"'"in_pcloud": "'"${val_pcloud}"'",' >> "${val_template_path}"
        echo -e "\t\t\t"'"conditions": null,' >> "${val_template_path}"
        echo -e "\t\t\t\"center\": [$nodex, $nodey, 0]" >> "${val_template_path}"
        if (( ${i} < (( ${num_pclouds} - 1 )) )); then
            echo -e "\t\t}," >> "${val_template_path}"
        else
            echo -e "\t\t}" >> "${val_template_path}"
        fi
    done
    cat << EOF >> "${val_template_path}"
    ],
EOF
done
