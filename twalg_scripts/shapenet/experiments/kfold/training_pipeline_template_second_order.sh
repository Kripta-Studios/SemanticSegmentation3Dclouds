#!/bin/bash

# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                              #
# BRIEF: Bash template to handle a single training pipeline with VL3D++        #
# ---------------------------------------------------------------------------- #


# Craft the VL3D pipeline : INITIALIZE
cat << EOF > "${VL3DPIPE}"
{
EOF
# Craft the VL3D pipeline : INPUT
cat << EOF >> "${VL3DPIPE}"
    "in_pcloud_concat": [
EOF
num_pclouds_train=${#pclouds_train[@]}
nx=$(echo "sqrt(${num_pclouds_train})" | bc)
if (( ${nx} < 1 )); then
    nx=1
fi
ny=$(echo "( ${num_pclouds_train} + ${nx} - 1) / ${nx}" | bc)
for(( i=0 ; i < ${nx} ; ++i)); do
    for(( j=0 ; j < ${ny} ; ++j)); do
        # Pick current training point cloud
        k=$(( ${i} * ${ny} + ${j} ))
        # End the loop if there are no more available training point clouds
        if (( ${k} == ${num_pclouds_train} )); then
            i=${nx}
            break
        fi
        # Add current training point cloud to pipeline but translated
cat << EOF >> "${VL3DPIPE}"
        {
            "in_pcloud": "${pclouds_train[$k]}",
            "conditions": null,
            "center": [
                $(echo "${i} * ${INTERNODE_SEPARATION}" | bc),
                $(echo "${j} * ${INTERNODE_SEPARATION}" | bc),
                0
            ]
        }
EOF
        # Append comma separator unless for the last point cloud
        if (( ${k} < (( ${num_pclouds_train} -1 )) )); then
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

# Craft the VL3D pipeline : DATA MINING
cat << EOF >> "${VL3DPIPE}"
        {
            "miner": "GeometricFeaturesPP",
            "first_order_strategy": "pca",
            "second_order_strategy": "taubin_bias",
            "idw_p": 0,
            "idw_eps": 0.006,
            "foc_K": 0,
            "foc_sigma": 3.14159265358979323846264338327950288419716939937510,
            "neighborhood": {
              "type": "sphere",
              "radius": 0.03,
              "k": 64,
              "lower_bound": 0,
              "upper_bound": 0
            },
            "fnames": [
                "linearity", "planarity", "sphericity", "surface_variation", "anisotropy", "roughness"
              ],
            "frenames": [
                "linearity_r0_03", "planarity_r0_03", "sphericity_r0_03", "surface_variation_r0_03", "anisotropy_r0_03", "roughness_r0_03"
            ],
            "non_degenerate_eigenthreshold": 0.00001,
            "tikhonov_parameter": 0.0000001,
            "nthreads": -1
        },
        {
            "miner": "GeometricFeaturesPP",
            "first_order_strategy": "pca",
            "second_order_strategy": "taubin_bias",
            "idw_p": 0,
            "idw_eps": 0.012,
            "foc_K": 0,
            "foc_sigma": 3.14159265358979323846264338327950288419716939937510,
            "neighborhood": {
              "type": "sphere",
              "radius": 0.06,
              "k": 64,
              "lower_bound": 0,
              "upper_bound": 0
            },
            "fnames": [
                "linearity", "planarity", "sphericity", "surface_variation", "anisotropy", "roughness",
                "gauss_curv_full", "mean_curv_full", "full_quad_dev", "full_abs_algdist", "full_gradient_norm",
                "full_laplacian", "full_rmsc", "full_gauss_umbilicality", "full_shape_index", "saddleness",
                "full_normalized_gac"
              ],
            "frenames": [
                "linearity_r0_06", "planarity_r0_06", "sphericity_r0_06", "surface_variation_r0_06", "anisotropy_r0_06", "roughness_r0_06",
                "gauss_curv_full_r0_06", "mean_curv_full_r0_06", "full_quad_dev_r0_06", "full_abs_algdist_r0_06", "full_gradient_norm_r0_06",
                "full_laplacian_r0_06", "full_rmsc_r0_06", "full_gauss_umbilicality_r0_06", "full_shape_index_r0_06", "saddleness_r0_06",
                "full_normalized_gac_r0_06"
            ],
            "non_degenerate_eigenthreshold": 0.00001,
            "tikhonov_parameter": 0.0000001,
            "nthreads": -1
        },
        {
            "miner": "GeometricFeaturesPP",
            "first_order_strategy": "pca",
            "second_order_strategy": "taubin_bias",
            "idw_p": 0,
            "idw_eps": 0.02,
            "foc_K": 0,
            "foc_sigma": 3.14159265358979323846264338327950288419716939937510,
            "neighborhood": {
              "type": "sphere",
              "radius": 0.1,
              "k": 64,
              "lower_bound": 0,
              "upper_bound": 0
            },
            "fnames": [
                "linearity", "planarity", "sphericity", "surface_variation", "anisotropy", "roughness",
                "gauss_curv_full", "mean_curv_full", "full_quad_dev", "full_abs_algdist", "full_gradient_norm",
                "full_laplacian", "full_rmsc", "full_gauss_umbilicality", "full_shape_index", "saddleness",
                "full_normalized_gac"
              ],
            "frenames": [
                "linearity_r0_1", "planarity_r0_1", "sphericity_r0_1", "surface_variation_r0_1", "anisotropy_r0_1", "roughness_r0_1",
                "gauss_curv_full_r0_1", "mean_curv_full_r0_1", "full_quad_dev_r0_1", "full_abs_algdist_r0_1", "full_gradient_norm_r0_1",
                "full_laplacian_r0_1", "full_rmsc_r0_1", "full_gauss_umbilicality_r0_1", "full_shape_index_r0_1", "saddleness_r0_1",
                "full_normalized_gac_r0_1"
            ],
            "non_degenerate_eigenthreshold": 0.00001,
            "tikhonov_parameter": 0.0000001,
            "nthreads": -1
        },
        {
            "miner": "GeometricFeaturesPP",
            "first_order_strategy": "pca",
            "second_order_strategy": "taubin_bias",
            "idw_p": 0,
            "idw_eps": 0.06,
            "foc_K": 0,
            "foc_sigma": 3.14159265358979323846264338327950288419716939937510,
            "neighborhood": {
              "type": "sphere",
              "radius": 0.3,
              "k": 64,
              "lower_bound": 0,
              "upper_bound": 0
            },
            "fnames": [
                "linearity", "planarity", "sphericity", "surface_variation", "anisotropy", "roughness",
                "gauss_curv_full", "mean_curv_full", "full_quad_dev", "full_abs_algdist", "full_gradient_norm",
                "full_laplacian", "full_rmsc", "full_gauss_umbilicality", "full_shape_index", "saddleness",
                "full_normalized_gac"
              ],
            "frenames": [
                "linearity_r0_3", "planarity_r0_3", "sphericity_r0_3", "surface_variation_r0_3", "anisotropy_r0_3", "roughness_r0_3",
                "gauss_curv_full_r0_3", "mean_curv_full_r0_3", "full_quad_dev_r0_3", "full_abs_algdist_r0_3", "full_gradient_norm_r0_3",
                "full_laplacian_r0_3", "full_rmsc_r0_3", "full_gauss_umbilicality_r0_3", "full_shape_index_r0_3", "saddleness_r0_3",
                "full_normalized_gac_r0_3"
            ],
            "non_degenerate_eigenthreshold": 0.00001,
            "tikhonov_parameter": 0.0000001,
            "nthreads": -1
        },
        {
            "miner": "GeometricFeaturesPP",
            "first_order_strategy": "pca",
            "second_order_strategy": "taubin_bias",
            "idw_p": 0,
            "idw_eps": 0.12,
            "foc_K": 0,
            "foc_sigma": 3.14159265358979323846264338327950288419716939937510,
            "neighborhood": {
              "type": "sphere",
              "radius": 0.6,
              "k": 64,
              "lower_bound": 0,
              "upper_bound": 0
            },
            "fnames": [
                "linearity", "planarity", "sphericity", "surface_variation", "anisotropy", "roughness"
              ],
            "frenames": [
                "linearity_r0_6", "planarity_r0_6", "sphericity_r0_6", "surface_variation_r0_6", "anisotropy_r0_6", "roughness_r0_6"
            ],
            "non_degenerate_eigenthreshold": 0.00001,
            "tikhonov_parameter": 0.0000001,
            "nthreads": -1
        },
EOF

# Craft the VL3D pipeline : WRITER
cat << EOF >> "${VL3DPIPE}"
        {
            "writer": "Writer",
            "out_pcloud": "*/${OUTFEATS}"
        },
EOF

# Craft the VL3D pipeline : MODEL
cat << EOF >> "${VL3DPIPE}"
        {
            "train": "RandomForestClassifier",
            "fnames": ["AUTO"],
            "training_type": "base",
            "num_folds": 5,
            "autoval_metrics": ["F1", "OA", "MCC"],
            "random_seed": null,
            "shuffle_points": false,
            "model_args": {
                "n_estimators": 100,
                "criterion": "entropy",
                "max_depth": 25,
                "min_samples_split": 16,
                "min_samples_leaf": 4,
                "min_weight_fraction_leaf": 0.0,
                "max_features": "sqrt",
                "max_leaf_nodes": null,
                "min_impurity_decrease": 0.0,
                "bootstrap": true,
                "oob_score": false,
                "n_jobs": -1,
                "warm_start": false,
                "class_weight": null,
                "ccp_alpha": 0.0,
                "max_samples": 0.8
            },
            "hyperparameter_tuning": null,
            "importance_report_path": "*/RF_importance.log",
            "importance_report_permutation": false,
            "decision_plot_path": null,
            "decision_plot_trees": 0,
            "decision_plot_max_depth": 0,
            "stratkfold_report_path": "*stratkfold_report.log",
            "stratkfold_plot_path": "*startkfold_plot.svg",
            "training_data_pipeline": [
                {
                    "component": "ClasswiseSampler",
                    "component_args": {
                        "target_class_distribution": ${TRAINING_DISTRIBUTION},
                        "replace": false
                    }
                }
            ]
        },
EOF

# Craft the VL3D pipeline : PREDICTIVE PIPELINE
cat << EOF >> "${VL3DPIPE}"
    {
        "writer": "PredictivePipelineWriter",
        "out_pipeline": "*/model/rf.pipe",
        "include_writer": false,
        "include_imputer": true,
        "include_feature_transformer": true,
        "include_miner": true,
        "include_class_transformer": true,
        "include_clustering": false,
        "ignore_predictions": false
    }
EOF

# Craft the VL3D pipeline : EVALUATION
cat << EOF >> "${VL3DPIPE}"
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
vl3d_training_pipeline="${specdir}/training.json"
mv "${VL3DPIPE}" "${vl3d_training_pipeline}"
