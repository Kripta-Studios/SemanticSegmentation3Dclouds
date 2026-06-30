#!/bin/bash

# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                              #
# BRIEF: Bash template to handle a single validation pipeline with VL3D++      #
# ---------------------------------------------------------------------------- #



# Craft the VL3D pipeline : INITIALIZE
cat << EOF > "${VL3DPIPE}"
{
EOF
# Craft the VL3D pipeline : INPUT
cat << EOF >> "${VL3DPIPE}"
    "in_pcloud_concat": [
EOF
num_pclouds_val=${#pclouds_val[@]}
nx=$(echo "sqrt(${num_pclouds_val})" | bc)
if (( ${nx} < 1 )); then
    nx=1
fi
ny=$(echo "( ${num_pclouds_val} + ${nx} - 1) / ${nx}" | bc)
for(( i=0 ; i < ${nx} ; ++i)); do
    for(( j=0 ; j < ${ny} ; ++j)); do
        # Pick current validation point cloud
        k=$(( ${i} * ${ny} + ${j} ))
        # End the loop if there are no more available validation point clouds
        if (( ${k} == ${num_pclouds_val} )); then
            i=${nx}
            break
        fi
        # Add current validation point cloud to pipeline but translated
cat << EOF >> "${VL3DPIPE}"
        {
            "in_pcloud": "${pclouds_val[$k]}",
            "conditions": null,
            "center": [
                $(echo "${i} * ${INTERNODE_SEPARATION}" | bc),
                $(echo "${j} * ${INTERNODE_SEPARATION}" | bc),
                0
            ]
        }
EOF
        # Append comma separator unless for the last point cloud
        if (( ${k} < (( ${num_pclouds_val} -1 )) )); then
            echo "," >> "${VL3DPIPE}"
        fi
    done
done
cat << EOF >> "${VL3DPIPE}"
    ],
EOF

# Craft the VL3D pipeline : OUTPUT
cat << EOF >> "${VL3DPIPE}"
    "out_pcloud": "${outdir}/*",
EOF

# Craft the VL3D pipeline : INIT SEQUENTIAL PIPELINE
cat << EOF >> "${VL3DPIPE}"
    "sequential_pipeline": [
EOF

# Craft the VL3D pipeline : PREDICTIVE PIPELINE
cat << EOF >> "${VL3DPIPE}"
        {
            "predict": "PredictivePipeline",
            "model_path": "${outdir}/model/rf.pipe",
            "nn_path": null
        },
EOF

# Craft the VL3D pipeline : CLASSIFICATION EVALUATION
cat << EOF >> "${VL3DPIPE}"
        {
            "eval": "ClassificationEvaluator",
            "class_names": ${CLASS_NAMES},
            "ignore_classes": ["other"],
            "ignore_predictions": true,
            "metrics": ["OA", "P", "R", "F1", "IoU", "wP", "wR", "wF1", "wIoU", "MCC", "Kappa"],
            "class_metrics": ["P", "R", "F1", "IoU"],
            "report_path": "*/eval/report/global_eval.log",
            "class_report_path": "*/eval/report/class_eval.log",
            "confusion_matrix_report_path" : "*/eval/report/confusion_matrix.log",
            "confusion_matrix_plot_path" : "*/eval/plot/confusion_matrix.svg",
            "confusion_matrix_normalization_strategy": "row",
            "class_distribution_report_path": "*/eval/report/class_distribution.log",
            "class_distribution_plot_path": "*/eval/plot/class_distribution.svg",
            "nthreads": -1
        },
EOF

# Craft the VL3D pipeline : UNCERTAINTY EVALUATION
cat << EOF >> "${VL3DPIPE}"
    {
        "eval": "ClassificationUncertaintyEvaluator",
        "class_names": ${CLASS_NAMES},
        "ignore_classes": ["other"],
        "include_probabilities": true,
        "include_weighted_entropy": false,
        "include_clusters": false,
        "weight_by_predictions": false,
        "num_clusters": 0,
        "clustering_max_iters": 0,
        "clustering_batch_size": 0,
        "clustering_entropy_weights": false,
        "clustering_reduce_function": null,
        "gaussian_kernel_points": 256,
        "report_path": "*/uncertainty/uncertainty.las",
        "plot_path": "*/uncertainty/"
    }
EOF

# Craft the VL3D pipeline : END SEQUENTIAL PIPELINE
cat << EOF >> "${VL3DPIPE}"
    ]
EOF

# Craft the VL3D pipeline : END OF FILE
cat << EOF >> "${VL3DPIPE}"
}
EOF

# Move pipeline spec to experiment directory
vl3d_validation_pipeline="${specdir}/validation.json"
mv "${VL3DPIPE}" "${vl3d_validation_pipeline}"
