#!/bin/bash


# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                              #
# BRIEF: Script to run the experiment of hybrid geometric descriptors          #
#        in the DALES dataset                                                  #
# ---------------------------------------------------------------------------- #


# ---   CONSTANTS   --- #
# --------------------- #
TRAIN_PCLOUDS=(
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5080_54435.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5085_54320.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5095_54440.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5095_54455.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5100_54495.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5105_54405.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5105_54460.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5110_54320.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5110_54460.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5110_54475.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5110_54495.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5115_54480.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5130_54355.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5135_54495.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5140_54445.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5145_54340.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5145_54405.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5145_54460.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5145_54470.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5145_54480.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5150_54340.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5160_54330.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5165_54390.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5165_54395.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5180_54435.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5180_54485.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5185_54390.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5185_54485.laz"
    "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/train/5190_54400.laz"
)
VAL_PCLOUDS=(
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5080_54400.laz"
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5080_54470.laz"
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5100_54440.laz"
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5100_54490.laz"
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5120_54445.laz"
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5135_54430.laz"
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5135_54435.laz"
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5140_54390.laz"
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5150_54325.laz"
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5155_54335.laz"
        "/oldext4/lidar_data/dales/las/dales_las/train/vl3d_framework/out/geom2_mined/test/5175_54395.laz"
)
TMP_JSON='/tmp/geom2_dales_tmp.json'
OUTDIR='/oldext4/lidar_data/dales/las/dales_las/vl3d/out/hybrid/'
VL3DPY='/home/uadmin/git/vl3d/vl3d.py'
VL3DENV='/home/uadmin/git/vl3d/venv/bin/activate'


# ---   MAIN   --- #
# ---------------- #
# Build training JSON
cat << EOF > "${TMP_JSON}"
{
    "in_pcloud_concat": [
EOF
num_train_pclouds=${#TRAIN_PCLOUDS[@]}
for (( i=0 ; i < ${num_train_pclouds} ; ++i )); do
    cat << EOF >> "${TMP_JSON}"
        {
            "in_pcloud": "${TRAIN_PCLOUDS[$i]}",
            "conditions": null,
            "fnames": [
                "linearity_r1", "planarity_r1", "sphericity_r1", "surfvar_r1", "anisotropy_r1",
                "roughness_r1", "verticality_r1", "PCA1_r1", "PCA2_r1",
                "linearity_r1_5", "planarity_r1_5", "sphericity_r1_5", "surfvar_r1_5", "anisotropy_r1_5",
                "roughness_r1_5", "verticality_r1_5", "PCA1_r1_5", "PCA2_r1_5",
                "gauss_curv_r1_5", "mean_curv_r1_5", "quad_dev_r1_5", "abs_algdist_r1_5", "gradnorm_r1_5",
                "laplacian_r1_5", "rmsc_r1_5", "gumbilicality_r1_5", "shpidx_r1_5", "saddleness_r1_5", "normal_gac_r1_5",
                "mean_qdev_r1_5", "linear_norm_r1_5", "spectral_r1_5", "frobenius_r1_5", "normal_bipgnorm_r1_5",
                "linearity_r2", "planarity_r2", "sphericity_r2", "surfvar_r2", "anisotropy_r2",
                "roughness_r2", "verticality_r2", "PCA1_r2", "PCA2_r2",
                "linearity_r3", "planarity_r3", "sphericity_r3", "surfvar_r3", "anisotropy_r3",
                "roughness_r3", "verticality_r3", "PCA1_r3", "PCA2_r3",
                "gauss_curv_r3", "mean_curv_r3", "quad_dev_r3", "abs_algdist_r3", "gradnorm_r3",
                "laplacian_r3", "rmsc_r3", "gumbilicality_r3", "shpidx_r3", "saddleness_r3", "normal_gac_r3",
                "mean_qdev_r3", "linear_norm_r3", "spectral_r3", "frobenius_r3", "normal_bipgnorm_r3",
                "linearity_r5", "planarity_r5", "sphericity_r5", "surfvar_r5", "anisotropy_r5",
                "roughness_r5", "verticality_r5", "PCA1_r5", "PCA2_r5",
                "gauss_curv_r5", "mean_curv_r5", "quad_dev_r5", "abs_algdist_r5", "gradnorm_r5",
                "laplacian_r5", "rmsc_r5", "gumbilicality_r5", "shpidx_r5", "saddleness_r5", "normal_gac_r5",
                "mean_qdev_r5", "linear_norm_r5", "spectral_r5", "frobenius_r5", "normal_bipgnorm_r5",
                "floordist_r10", "ceildist_r10", "floordist_r30", "ceildist_r30"
            ],
            "target_class_distribution": [
                0, 250000, 250000, 250000, 250000,
                250000, 250000, 250000, 250000
            ]
EOF
    if (( $i < (( ${num_train_pclouds} -1 )) )); then
        echo -e "\t\t}," >> "${TMP_JSON}"
    else
        echo -e "\t\t}" >> "${TMP_JSON}"
    fi
done
cat << EOF >> "${TMP_JSON}"
    ],
    "out_pcloud": "${OUTDIR}/*",
    "sequential_pipeline": [
        {
            "class_transformer": "ClassReducer",
            "on_predictions": false,
            "input_class_names": [
                "noclass", "ground", "vegetation", "car", "truck",
                "powerline", "fence", "pole", "building"
            ],
            "output_class_names": [
                "ground", "vegetation", "car", "truck", "powerline",
                "fence", "pole", "building", "noclass"
            ],
            "class_groups": [
                ["ground"],
                ["vegetation"],
                ["car"],
                ["truck"],
                ["powerline"],
                ["fence"],
                ["pole"],
                ["building"],
                ["noclass"]
            ],
            "report_path": "*/class_reduction.log",
            "plot_path": "*/class_reduction.svg"
        },
        {
            "train": "RandomForestClassifier",
            "fnames": [
                "linearity_r1", "planarity_r1", "sphericity_r1", "surfvar_r1", "anisotropy_r1",
                "roughness_r1", "verticality_r1", "PCA1_r1", "PCA2_r1",
                "linearity_r1_5", "planarity_r1_5", "sphericity_r1_5", "surfvar_r1_5", "anisotropy_r1_5",
                "roughness_r1_5", "verticality_r1_5", "PCA1_r1_5", "PCA2_r1_5",
                "gauss_curv_r1_5", "mean_curv_r1_5", "quad_dev_r1_5", "abs_algdist_r1_5", "gradnorm_r1_5",
                "laplacian_r1_5", "rmsc_r1_5", "gumbilicality_r1_5", "shpidx_r1_5", "saddleness_r1_5", "normal_gac_r1_5",
                "mean_qdev_r1_5", "linear_norm_r1_5", "spectral_r1_5", "frobenius_r1_5", "normal_bipgnorm_r1_5",
                "linearity_r2", "planarity_r2", "sphericity_r2", "surfvar_r2", "anisotropy_r2",
                "roughness_r2", "verticality_r2", "PCA1_r2", "PCA2_r2",
                "linearity_r3", "planarity_r3", "sphericity_r3", "surfvar_r3", "anisotropy_r3",
                "roughness_r3", "verticality_r3", "PCA1_r3", "PCA2_r3",
                "gauss_curv_r3", "mean_curv_r3", "quad_dev_r3", "abs_algdist_r3", "gradnorm_r3",
                "laplacian_r3", "rmsc_r3", "gumbilicality_r3", "shpidx_r3", "saddleness_r3", "normal_gac_r3",
                "mean_qdev_r3", "linear_norm_r3", "spectral_r3", "frobenius_r3", "normal_bipgnorm_r3",
                "linearity_r5", "planarity_r5", "sphericity_r5", "surfvar_r5", "anisotropy_r5",
                "roughness_r5", "verticality_r5", "PCA1_r5", "PCA2_r5",
                "gauss_curv_r5", "mean_curv_r5", "quad_dev_r5", "abs_algdist_r5", "gradnorm_r5",
                "laplacian_r5", "rmsc_r5", "gumbilicality_r5", "shpidx_r5", "saddleness_r5", "normal_gac_r5",
                "mean_qdev_r5", "linear_norm_r5", "spectral_r5", "frobenius_r5", "normal_bipgnorm_r5",
                "floordist_r10", "ceildist_r10", "floordist_r30", "ceildist_r30"
            ],
            "training_type": "base",
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
            "training_data_pipeline": [
                {
                    "component": "ClasswiseSampler",
                    "component_args": {
                        "target_class_distribution": [
                            10000000,   10000000,   10000000,   10000000,   10000000,
                            10000000,   10000000,   10000000,   0
                        ],
                        "replace": false
                    }
                }
            ]
        },
        {
            "writer": "PredictivePipelineWriter",
            "out_pipeline": "*/model/rf.pipe",
            "include_writer": false,
            "include_imputer": false,
            "include_feature_transformer": false,
            "include_miner": false,
            "include_class_transformer": false,
            "include_clustering": false,
            "ignore_predictions": false
        }
    ]
}
EOF
# Load VL3D++ environment
echo 'Loading VL3D++ environment ...'
if [[ -f ${VL3DENV} ]]; then
    source ${VL3DENV}
    echo 'Loaded VL3D++ environment!'
else
    echo 'Failed to load VL3D++ environment!'
    exit 1
fi
# Run training pipeline
python3 ${VL3DPY} --pipeline "${TMP_JSON}"
# Move TMP json to spec folder in output directory
mkdir -p "${OUTDIR}/spec"
mv ${TMP_JSON} "${OUTDIR}/spec/training.json"
# Build validation JSON
cat << EOF > ${TMP_JSON}
{
    "in_pcloud": [
EOF
num_val_pclouds=${#VAL_PCLOUDS[@]}
for (( i=0 ; i < ${num_val_pclouds} ; ++i )); do
    if (( $i < (( ${num_val_pclouds} -1 )) )); then
        echo -e "\t\t\"${VAL_PCLOUDS[$i]}\"," >> "${TMP_JSON}"
    else
        echo -e "\t\t\"${VAL_PCLOUDS[$i]}\"" >> "${TMP_JSON}"
    fi
done
cat << EOF >> ${TMP_JSON}
    ],
    "out_pcloud": [
EOF
for (( i=0 ; i < ${num_val_pclouds} ; ++i )); do
    subdir=$(basename "${VAL_PCLOUDS[$i]}" | sed 's/\..*//g')
    outpath="${OUTDIR}/${subdir}/"
    if (( $i < (( ${num_val_pclouds} -1 )) )); then
        echo -e "\t\t\"${outpath}*\"," >> "${TMP_JSON}"
    else
        echo -e "\t\t\"${outpath}*\"" >> "${TMP_JSON}"
    fi
done
cat << EOF >> ${TMP_JSON}
    ],
    "sequential_pipeline": [
        {
            "class_transformer": "ClassReducer",
            "on_predictions": false,
            "input_class_names": [
                "noclass", "ground", "vegetation", "car", "truck",
                "powerline", "fence", "pole", "building"
            ],
            "output_class_names": [
                "ground", "vegetation", "car", "truck", "powerline",
                "fence", "pole", "building", "noclass"
            ],
            "class_groups": [
                ["ground"],
                ["vegetation"],
                ["car"],
                ["truck"],
                ["powerline"],
                ["fence"],
                ["pole"],
                ["building"],
                ["noclass"]
            ],
            "report_path": "*/class_reduction.log",
            "plot_path": "*/class_reduction.svg"
        },
        {
            "predict": "PredictivePipeline",
            "model_path": "${OUTDIR}/model/rf.pipe",
            "nn_path": null
        },
        {
            "writer": "ClassifiedPcloudWriter",
            "out_pcloud": "*/predicted.las"
        },
        {
            "eval": "ClassificationEvaluator",
            "class_names": [
                "ground", "vegetation", "car", "truck", "powerline",
                "fence", "pole", "building", "noclass"
            ],
            "ignore_classes": ["noclass"],
            "ignore_predictions": false,
            "metrics": ["OA", "P", "R", "F1", "IoU", "wP", "wR", "wF1", "wIoU", "MCC", "Kappa"],
            "class_metrics": ["P", "R", "F1", "IoU"],
            "report_path": "*/eval/global_eval.log",
            "class_report_path": "*/eval/class_eval.log",
            "confusion_matrix_report_path" : "*/eval/confusion_matrix.log",
            "confusion_matrix_plot_path" : "*/eval/confusion_matrix.svg",
            "confusion_matrix_normalization_strategy": "row",
            "class_distribution_report_path": "*/eval/class_distribution.log",
            "class_distribution_plot_path": "*/eval/class_distribution.svg",
            "nthreads": -1
        },
        {
            "eval": "ClassificationUncertaintyEvaluator",
            "class_names": [
                "ground", "vegetation", "car", "truck",
                "powerline", "fence", "pole", "building", "noclass"
            ],
            "ignore_classes": ["noclass"],
            "include_probabilities": true,
            "include_weighted_entropy": false,
            "include_clusters": false,
            "weight_by_predictions": false,
            "num_clusters": 0,
            "clustering_max_iters": 0,
            "clustering_batch_size": 0,
            "clustering_entropy_weights": false,
            "clustering_reduce_function": null,
            "gaussian_kernel_points": 0,
            "report_path": "*/uncertainty/uncertainty.las",
            "plot_path": null
        }
    ]
}
EOF
# Run validation pipeline
python3 ${VL3DPY} --pipeline "${TMP_JSON}"
# Move TMP json to spec folder in output directory
mv ${TMP_JSON} "${OUTDIR}/spec/validation.json"
