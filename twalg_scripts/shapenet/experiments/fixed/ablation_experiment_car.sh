#!/bin/bash

# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Split each ShapeNet dataset into training and validation to
#        conduct an ablation study on the features for the car dataset.
#
# FEATURE IMPORTANCES:
# full_gauss_umbilicality_r0_6    ,             0.007525
# full_rmsc_r0_6                  ,             0.008238
# mean_curv_full_r0_6             ,             0.008869
# saddleness_r0_6                 ,             0.011603
# full_quad_dev_r0_6              ,             0.013045
# gauss_curv_full_r0_6            ,             0.018718
# full_laplacian_r0_6             ,             0.019913
# full_shape_index_r0_6           ,             0.021486
# full_abs_algdist_r0_6           ,             0.025099
# full_gradient_norm_r0_6         ,             0.068143
# full_normalized_gac_r0_6        ,             0.078881
#
# ---------------------------------------------------------------------------- #


# ---  CONSTANTS  --- #
# ------------------- 3
REF_TRAINING_JSON='/home/uadmin/git/twalg_scripts/shapenet/experiments/fixed/car_training_second_order.json'
REF_VAL_JSON='/home/uadmin/git/twalg_scripts/shapenet/experiments/fixed/car_validation_second_order.json'
INTERNODE_SEPARATION=5
FEATURES=(
    "full_normalized_gac"
    "full_gradient_norm"
    "full_abs_algdist"
    "full_shape_index"
    "full_laplacian"
    "gauss_curv_full"
    "full_quad_dev"
    "saddleness"
    "mean_curv_full"
    "full_rmsc"
    "full_gauss_umbilicality"
)
TMP_TRAIN_JSON='tmp_ablation_training_car.json'
TMP_VAL_JSON='tmp_ablation_predict_car.json'
RADII=(
    "0.06"
    "0.1"
    "0.3"
    "0.6"
)
OUTDIR='/ext4/lidar_data/shapenet/vl3d/out/car/ablation/'
VL3DPY='/home/uadmin/git/vl3d/vl3d.py'
VL3DENV='/home/uadmin/git/vl3d/venv/bin/activate'

# ---   MAIN   --- #
# ---------------- #
# Iterate features
nradii=${#RADII[@]}
nfeats=${#FEATURES[@]}
for (( i=0 ; i < ${nfeats} ; ++i )); do
    imax=$(( $i + 1 ))

    # ---  TRAINING  --- #
    # ------------------ #
    # Build training JSON
    train_dir="${OUTDIR}/nfeats_${imax}/"
    cat << EOF > ${TMP_TRAIN_JSON}
{
    "in_pcloud_concat": [   
EOF
    grep -B 1 -A 3 '"in_pcloud"' "${REF_TRAINING_JSON}" >> "${TMP_TRAIN_JSON}"
    cat << EOF >> ${TMP_TRAIN_JSON}
    ],
    "out_pcloud": "${train_dir}/*",
    "sequential_pipeline": [
EOF
    # Add geometric features for each radius
    for (( j=0 ; j < ${nradii} ; ++j )); do
        radius=${RADII[$j]}
        radius_alt=$(sed 's/\./_/g' <<< ${radius})
        cat << EOF >> ${TMP_TRAIN_JSON}
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
              "radius": ${radius},
              "k": 64,
              "lower_bound": 0,
              "upper_bound": 0
            },
            "fnames": [
EOF
        echo -ne "\t\t\t\t" >> ${TMP_TRAIN_JSON}
        for (( k=0 ; k < ${imax} ; ++k )); do
            echo -n "\"${FEATURES[$k]}\"" >> ${TMP_TRAIN_JSON}
            if (( ${k} < $(( ${imax} - 1 )) )); then
                echo -n ", " >> ${TMP_TRAIN_JSON}
            else
                echo >> ${TMP_TRAIN_JSON}
            fi
        done
        cat << EOF >> ${TMP_TRAIN_JSON}
            ],
            "frenames": [
EOF
        echo -ne "\t\t\t\t" >> ${TMP_TRAIN_JSON}
        for (( k=0 ; k < ${imax} ; ++k )); do
            echo -n "\"${FEATURES[$k]}_r${radius_alt}\"" >> ${TMP_TRAIN_JSON}
            if (( ${k} < $(( ${imax} - 1 )) )); then
                echo -n ", " >> ${TMP_TRAIN_JSON}
            else
                echo >> ${TMP_TRAIN_JSON}
            fi
        done
        cat << EOF >> ${TMP_TRAIN_JSON}
            ],
            "non_degenerate_eigenthreshold": 0.00001,
            "tikhonov_parameter": 0.0000001,
            "nthreads": -1
        },
EOF
    done
    cat << EOF >> ${TMP_TRAIN_JSON}
        {
            "writer": "Writer",
            "out_pcloud": "*/training_car.las"
        },
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
                        "target_class_distribution": [2000000, 2000000, 2000000, 0],
                        "replace": false
                    }
                }
            ]
        },
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
    ]
}
EOF

    # Run training JSON
    python3 ${VL3DPY} --pipeline "${TMP_TRAIN_JSON}"

    # ---  PREDICT AND EVAL --- #
    # ------------------------- #
    # Build predict JSON
    val_dir="${OUTDIR}/nfeats_${imax}/"
    cat << EOF > ${TMP_VAL_JSON}
{
    "in_pcloud_concat": [   
EOF
    grep -B 1 -A 3 '"in_pcloud"' "${REF_VAL_JSON}" >> "${TMP_VAL_JSON}"
    cat << EOF >> ${TMP_VAL_JSON}
    ],
    "out_pcloud": "${val_dir}/*",
    "sequential_pipeline": [
        {
            "predict": "PredictivePipeline",
            "model_path": "${train_dir}/model/rf.pipe",
            "nn_path": null
        },
        {
            "eval": "ClassificationEvaluator",
            "class_names": ["wheel", "hood", "roof", "other"],
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
		{
		    "eval": "ClassificationUncertaintyEvaluator",
		    "class_names": ["wheel", "hood", "roof", "other"],
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
    ]
}
EOF

    # Run validation JSON
    python3 ${VL3DPY} --pipeline "${TMP_VAL_JSON}"

done

