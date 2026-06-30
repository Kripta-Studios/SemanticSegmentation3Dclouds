#!/bin/bash


# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                              #
# BRIEF: Script to run the sensitivity analysis on the tikhonov parameter
#        in the DALES dataset                                                  #
# ---------------------------------------------------------------------------- #


# ---   CONSTANTS   --- #
# --------------------- #
TRAIN_PCLOUDS=(
    "5080_54435"
    "5085_54320"
    "5095_54440"
    "5095_54455"
    "5100_54495"
    "5105_54405"
    "5105_54460"
    "5110_54320"
    "5110_54460"
    "5110_54475"
    "5110_54495"
    "5115_54480"
    "5130_54355"
    "5135_54495"
    "5140_54445"
    "5145_54340"
    "5145_54405"
    "5145_54460"
    "5145_54470"
    "5145_54480"
    "5150_54340"
    "5160_54330"
    "5165_54390"
    "5165_54395"
    "5180_54435"
    "5180_54485"
    "5185_54390"
    "5185_54485"
    "5190_54400"
)
VAL_PCLOUD="5135_54435"
TIKHONOV=(
    "0.0000001" "0.0000005" "0.000001" "0.000005"
    "0.00001" "0.00005" "0.0001" "0.0005" "0.001"
    "0.005" "0.01" "0.05" "0.1" "0.5" "1.0"
    "5.0" "10.0" "50.0" "100.0"
)
JSONDIR='sensit_tikhonov_only2nd/'
OUTDIR='/mnt/netapp2/Store_uscciaep/lidar_data/dales/vl3d/out/'




# ---------------- #
# ---   MAIN   --- #
# ---------------- #
# Make directory to generate JSON files
mkdir -p ${JSONDIR}


# For each change on a parameter
num_param_changes=${#TIKHONOV[@]}
for (( i=0 ; i < ${num_param_changes} ; ++i )); do


    # ---  MINING JSON : CHANGING TIKHONOV PARAMETER  --- #
    # --------------------------------------------------- #
    mining_json="${JSONDIR}/tikhonov_only2nd_mining_$(( $i + 1 )).json"
    tmp_json="${mining_json}"
    tikhonov=${TIKHONOV[$i]}
    outdir="${OUTDIR}/sensit/tikhonov_only2nd_${tikhonov}/"
    num_train_pclouds=${#TRAIN_PCLOUDS[@]}
    cat << EOF > ${tmp_json}
{
    "in_pcloud": [
EOF
    for (( j=0 ; j < ${num_train_pclouds} ; ++j )); do
        echo -e "\t\t\"/mnt/netapp2/Store_uscciaep/lidar_data/dales/las/train/${TRAIN_PCLOUDS[$j]}.laz\"," >> ${tmp_json}
    done
    echo -e "\t\t\"/mnt/netapp2/Store_uscciaep/lidar_data/dales/las/test/${VAL_PCLOUD}.laz\"" >> ${tmp_json}
    cat << EOF >> ${tmp_json}
    ],
    "out_pcloud": [
EOF
    for (( j=0 ; j < ${num_train_pclouds} ; ++j )); do
        echo -e "\t\t\"${outdir}/mined/${TRAIN_PCLOUDS[$j]}.las\"," >> ${tmp_json}
    done
    echo -e "\t\t\"/${outdir}/mined/${VAL_PCLOUD}.las\"" >> ${tmp_json}
    cat << EOF >> ${tmp_json}
    ],
    "sequential_pipeline": [
    	{
            "miner": "GeometricFeaturesPP",
            "first_order_strategy": "pca",
			"second_order_strategy": "taubin_bias",
			"idw_p": 0,
			"idw_eps": 11,
			"foc_K": 0,
			"foc_sigma": 3.14159265358979323846264338327950288419716939937510,
			"neighborhood": {
			  "type": "sphere",
			  "radius": 1.0,
			  "k": 1024,
			  "lower_bound": 0,
			  "upper_bound": 0
			},
            "fnames": [
            	"gauss_curv_full", "mean_curv_full", "full_quad_dev", "full_abs_algdist", "full_gradient_norm",
            	"full_laplacian", "full_rmsc", "full_gauss_umbilicality", "full_shape_index", "saddleness", "full_normalized_gac",
            	"full_mean_qdev", "full_linear_norm", "full_spectral", "full_frobenius", "normalized_bipgnorm"
        	],
            "frenames": [
            	"gauss_curv_r1", "mean_curv_r1", "quad_dev_r1", "abs_algdist_r1", "gradnorm_r1",
            	"laplacian_r1", "rmsc_r1", "gumbilicality_r1", "shpidx_r1", "saddleness_r1", "normal_gac_r1",
            	"mean_qdev_r1", "linear_norm_r1", "spectral_r1", "frobenius_r1", "normal_bipgnorm_r1"
            ],
            "non_degenerate_eigenthreshold": 0.00001,
    		"tikhonov_parameter": ${tikhonov},
            "nthreads": -1
        },
        {
            "miner": "GeometricFeaturesPP",
            "first_order_strategy": "pca",
			"second_order_strategy": "taubin_bias",
			"idw_p": 0,
			"idw_eps": 11,
			"foc_K": 0,
			"foc_sigma": 3.14159265358979323846264338327950288419716939937510,
			"neighborhood": {
			  "type": "sphere",
			  "radius": 1.5,
			  "k": 1024,
			  "lower_bound": 0,
			  "upper_bound": 0
			},
            "fnames": [
            	"gauss_curv_full", "mean_curv_full", "full_quad_dev", "full_abs_algdist", "full_gradient_norm",
            	"full_laplacian", "full_rmsc", "full_gauss_umbilicality", "full_shape_index", "saddleness", "full_normalized_gac",
            	"full_mean_qdev", "full_linear_norm", "full_spectral", "full_frobenius", "normalized_bipgnorm"
        	],
            "frenames": [
            	"gauss_curv_r1_5", "mean_curv_r1_5", "quad_dev_r1_5", "abs_algdist_r1_5", "gradnorm_r1_5",
            	"laplacian_r1_5", "rmsc_r1_5", "gumbilicality_r1_5", "shpidx_r1_5", "saddleness_r1_5", "normal_gac_r1_5",
            	"mean_qdev_r1_5", "linear_norm_r1_5", "spectral_r1_5", "frobenius_r1_5", "normal_bipgnorm_r1_5"
            ],
            "non_degenerate_eigenthreshold": 0.00001,
    		"tikhonov_parameter": ${tikhonov},
            "nthreads": -1
        },
        {
            "miner": "GeometricFeaturesPP",
            "first_order_strategy": "pca",
			"second_order_strategy": "taubin_bias",
			"idw_p": 0,
			"idw_eps": 11,
			"foc_K": 0,
			"foc_sigma": 3.14159265358979323846264338327950288419716939937510,
			"neighborhood": {
			  "type": "sphere",
			  "radius": 2.0,
			  "k": 1024,
			  "lower_bound": 0,
			  "upper_bound": 0
			},
            "fnames": [
            	"gauss_curv_full", "mean_curv_full", "full_quad_dev", "full_abs_algdist", "full_gradient_norm",
            	"full_laplacian", "full_rmsc", "full_gauss_umbilicality", "full_shape_index", "saddleness", "full_normalized_gac",
            	"full_mean_qdev", "full_linear_norm", "full_spectral", "full_frobenius", "normalized_bipgnorm"
        	],
            "frenames": [
            	"gauss_curv_r2", "mean_curv_r2", "quad_dev_r2", "abs_algdist_r2", "gradnorm_r2",
            	"laplacian_r2", "rmsc_r2", "gumbilicality_r2", "shpidx_r2", "saddleness_r2", "normal_gac_r2",
            	"mean_qdev_r2", "linear_norm_r2", "spectral_r2", "frobenius_r2", "normal_bipgnorm_r2"
            ],
            "non_degenerate_eigenthreshold": 0.00001,
    		"tikhonov_parameter": ${tikhonov},
            "nthreads": -1
        },
        {
            "miner": "GeometricFeaturesPP",
            "first_order_strategy": "pca",
			"second_order_strategy": "taubin_bias",
			"idw_p": 0,
			"idw_eps": 11,
			"foc_K": 0,
			"foc_sigma": 3.14159265358979323846264338327950288419716939937510,
			"neighborhood": {
			  "type": "sphere",
			  "radius": 3.0,
			  "k": 1024,
			  "lower_bound": 0,
			  "upper_bound": 0
			},
            "fnames": [
            	"gauss_curv_full", "mean_curv_full", "full_quad_dev", "full_abs_algdist", "full_gradient_norm",
            	"full_laplacian", "full_rmsc", "full_gauss_umbilicality", "full_shape_index", "saddleness", "full_normalized_gac",
            	"full_mean_qdev", "full_linear_norm", "full_spectral", "full_frobenius", "normalized_bipgnorm"
        	],
            "frenames": [
            	"gauss_curv_r3", "mean_curv_r3", "quad_dev_r3", "abs_algdist_r3", "gradnorm_r3",
            	"laplacian_r3", "rmsc_r3", "gumbilicality_r3", "shpidx_r3", "saddleness_r3", "normal_gac_r3",
            	"mean_qdev_r3", "linear_norm_r3", "spectral_r3", "frobenius_r3", "normal_bipgnorm_r3"
            ],
            "non_degenerate_eigenthreshold": 0.00001,
    		"tikhonov_parameter": ${tikhonov},
            "nthreads": -1
        },
        {
            "miner": "FPSDecorated",
            "fps_decorator": {
                "num_points": "m/2",
                "fast": 2,
                "num_encoding_neighbors": 1,
                "num_decoding_neighbors": 1,
                "release_encoding_neighborhoods": false,
                "threads": -1,
                "representation_report_path": null
            },
            "decorated_miner": {
		        "miner": "GeometricFeaturesPP",
		        "first_order_strategy": "pca",
				"second_order_strategy": "taubin_bias",
				"idw_p": 0,
				"idw_eps": 11,
				"foc_K": 0,
				"foc_sigma": 3.14159265358979323846264338327950288419716939937510,
				"neighborhood": {
				  "type": "sphere",
				  "radius": 5.0,
				  "k": 1024,
				  "lower_bound": 0,
				  "upper_bound": 0
				},
		        "fnames": [
		        	"gauss_curv_full", "mean_curv_full", "full_quad_dev", "full_abs_algdist", "full_gradient_norm",
		        	"full_laplacian", "full_rmsc", "full_gauss_umbilicality", "full_shape_index", "saddleness", "full_normalized_gac",
		        	"full_mean_qdev", "full_linear_norm", "full_spectral", "full_frobenius", "normalized_bipgnorm"
		    	],
		        "frenames": [
		        	"gauss_curv_r5", "mean_curv_r5", "quad_dev_r5", "abs_algdist_r5", "gradnorm_r5",
		        	"laplacian_r5", "rmsc_r5", "gumbilicality_r5", "shpidx_r5", "saddleness_r5", "normal_gac_r5",
		        	"mean_qdev_r5", "linear_norm_r5", "spectral_r5", "frobenius_r5", "normal_bipgnorm_r5"
		        ],
                "non_degenerate_eigenthreshold": 0.00001,
				"tikhonov_parameter": ${tikhonov},
		        "nthreads": -1
	        }
        }
    ]
}
EOF


    # ---   TRAINING JSON   --- #
    # ------------------------- #
    train_json="${JSONDIR}/tikhonov_only2nd_train_$(( $i + 1 )).json"
    tmp_json="${train_json}"
    cat << EOF > "${tmp_json}"
{
    "in_pcloud_concat": [
EOF
    for (( j=0 ; j < ${num_train_pclouds} ; ++j )); do
        cat << EOF >> "${tmp_json}"
        {
            "in_pcloud": "${outdir}/mined/${TRAIN_PCLOUDS[$j]}.las",
            "conditions": null,
            "fnames": [
                "gauss_curv_r1", "mean_curv_r1", "quad_dev_r1", "abs_algdist_r1", "gradnorm_r1",
                "laplacian_r1", "rmsc_r1", "gumbilicality_r1", "shpidx_r1", "saddleness_r1", "normal_gac_r1",
                "mean_qdev_r1", "linear_norm_r1", "spectral_r1", "frobenius_r1", "normal_bipgnorm_r1",
                "gauss_curv_r1_5", "mean_curv_r1_5", "quad_dev_r1_5", "abs_algdist_r1_5", "gradnorm_r1_5",
                "laplacian_r1_5", "rmsc_r1_5", "gumbilicality_r1_5", "shpidx_r1_5", "saddleness_r1_5", "normal_gac_r1_5",
                "mean_qdev_r1_5", "linear_norm_r1_5", "spectral_r1_5", "frobenius_r1_5", "normal_bipgnorm_r1_5",
                "gauss_curv_r2", "mean_curv_r2", "quad_dev_r2", "abs_algdist_r2", "gradnorm_r2",
                "laplacian_r2", "rmsc_r2", "gumbilicality_r2", "shpidx_r2", "saddleness_r2", "normal_gac_r2",
                "mean_qdev_r2", "linear_norm_r2", "spectral_r2", "frobenius_r2", "normal_bipgnorm_r2",
                "gauss_curv_r3", "mean_curv_r3", "quad_dev_r3", "abs_algdist_r3", "gradnorm_r3",
                "laplacian_r3", "rmsc_r3", "gumbilicality_r3", "shpidx_r3", "saddleness_r3", "normal_gac_r3",
                "mean_qdev_r3", "linear_norm_r3", "spectral_r3", "frobenius_r3", "normal_bipgnorm_r3",
                "gauss_curv_r5", "mean_curv_r5", "quad_dev_r5", "abs_algdist_r5", "gradnorm_r5",
                "laplacian_r5", "rmsc_r5", "gumbilicality_r5", "shpidx_r5", "saddleness_r5", "normal_gac_r5",
                "mean_qdev_r5", "linear_norm_r5", "spectral_r5", "frobenius_r5", "normal_bipgnorm_r5"
            ],
            "target_class_distribution": [
                0, 250000, 250000, 250000, 250000,
                250000, 250000, 250000, 250000
            ]
EOF
        if (( $j < (( ${num_train_pclouds} -1 )) )); then
            echo -e "\t\t}," >> "${tmp_json}"
        else
            echo -e "\t\t}" >> "${tmp_json}"
        fi
    done

    cat << EOF >> "${tmp_json}"
        ],
        "out_pcloud": "${outdir}/*",
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
                "imputer": "UnivariateImputer",
                "fnames": [
                    "gauss_curv_r1", "mean_curv_r1", "quad_dev_r1", "abs_algdist_r1", "gradnorm_r1",
                    "laplacian_r1", "rmsc_r1", "gumbilicality_r1", "shpidx_r1", "saddleness_r1", "normal_gac_r1",
                    "mean_qdev_r1", "linear_norm_r1", "spectral_r1", "frobenius_r1", "normal_bipgnorm_r1",
                    "gauss_curv_r1_5", "mean_curv_r1_5", "quad_dev_r1_5", "abs_algdist_r1_5", "gradnorm_r1_5",
                    "laplacian_r1_5", "rmsc_r1_5", "gumbilicality_r1_5", "shpidx_r1_5", "saddleness_r1_5", "normal_gac_r1_5",
                    "mean_qdev_r1_5", "linear_norm_r1_5", "spectral_r1_5", "frobenius_r1_5", "normal_bipgnorm_r1_5",
                    "gauss_curv_r2", "mean_curv_r2", "quad_dev_r2", "abs_algdist_r2", "gradnorm_r2",
                    "laplacian_r2", "rmsc_r2", "gumbilicality_r2", "shpidx_r2", "saddleness_r2", "normal_gac_r2",
                    "mean_qdev_r2", "linear_norm_r2", "spectral_r2", "frobenius_r2", "normal_bipgnorm_r2",
                    "gauss_curv_r3", "mean_curv_r3", "quad_dev_r3", "abs_algdist_r3", "gradnorm_r3",
                    "laplacian_r3", "rmsc_r3", "gumbilicality_r3", "shpidx_r3", "saddleness_r3", "normal_gac_r3",
                    "mean_qdev_r3", "linear_norm_r3", "spectral_r3", "frobenius_r3", "normal_bipgnorm_r3",
                    "gauss_curv_r5", "mean_curv_r5", "quad_dev_r5", "abs_algdist_r5", "gradnorm_r5",
                    "laplacian_r5", "rmsc_r5", "gumbilicality_r5", "shpidx_r5", "saddleness_r5", "normal_gac_r5",
                    "mean_qdev_r5", "linear_norm_r5", "spectral_r5", "frobenius_r5", "normal_bipgnorm_r5"
                ],
                "target_val": "NaN",
                "strategy": "mean",
                "constant_val": 0,
                "impute_coordinates": false,
                "impute_references": false
            },
            {
                "train": "RandomForestClassifier",
                "fnames": [
                    "gauss_curv_r1", "mean_curv_r1", "quad_dev_r1", "abs_algdist_r1", "gradnorm_r1",
                    "laplacian_r1", "rmsc_r1", "gumbilicality_r1", "shpidx_r1", "saddleness_r1", "normal_gac_r1",
                    "mean_qdev_r1", "linear_norm_r1", "spectral_r1", "frobenius_r1", "normal_bipgnorm_r1",
                    "gauss_curv_r1_5", "mean_curv_r1_5", "quad_dev_r1_5", "abs_algdist_r1_5", "gradnorm_r1_5",
                    "laplacian_r1_5", "rmsc_r1_5", "gumbilicality_r1_5", "shpidx_r1_5", "saddleness_r1_5", "normal_gac_r1_5",
                    "mean_qdev_r1_5", "linear_norm_r1_5", "spectral_r1_5", "frobenius_r1_5", "normal_bipgnorm_r1_5",
                    "gauss_curv_r2", "mean_curv_r2", "quad_dev_r2", "abs_algdist_r2", "gradnorm_r2",
                    "laplacian_r2", "rmsc_r2", "gumbilicality_r2", "shpidx_r2", "saddleness_r2", "normal_gac_r2",
                    "mean_qdev_r2", "linear_norm_r2", "spectral_r2", "frobenius_r2", "normal_bipgnorm_r2",
                    "gauss_curv_r3", "mean_curv_r3", "quad_dev_r3", "abs_algdist_r3", "gradnorm_r3",
                    "laplacian_r3", "rmsc_r3", "gumbilicality_r3", "shpidx_r3", "saddleness_r3", "normal_gac_r3",
                    "mean_qdev_r3", "linear_norm_r3", "spectral_r3", "frobenius_r3", "normal_bipgnorm_r3",
                    "gauss_curv_r5", "mean_curv_r5", "quad_dev_r5", "abs_algdist_r5", "gradnorm_r5",
                    "laplacian_r5", "rmsc_r5", "gumbilicality_r5", "shpidx_r5", "saddleness_r5", "normal_gac_r5",
                    "mean_qdev_r5", "linear_norm_r5", "spectral_r5", "frobenius_r5", "normal_bipgnorm_r5"
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


    # ---   VALIDATION JSON   --- #
    # --------------------------- #
    # Build validation JSON
    val_json="${JSONDIR}/tikhonov_only2nd_val_$(( $i + 1 )).json"
    tmp_json="${val_json}"
    cat << EOF > ${tmp_json}
{
    "in_pcloud": ["${outdir}/mined/${VAL_PCLOUD}.las"],
    "out_pcloud": ["${outdir}/${VAL_PCLOUD}/*"],
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
            "imputer": "UnivariateImputer",
            "fnames": [
                "gauss_curv_r1", "mean_curv_r1", "quad_dev_r1", "abs_algdist_r1", "gradnorm_r1",
                "laplacian_r1", "rmsc_r1", "gumbilicality_r1", "shpidx_r1", "saddleness_r1", "normal_gac_r1",
                "mean_qdev_r1", "linear_norm_r1", "spectral_r1", "frobenius_r1", "normal_bipgnorm_r1",
                "gauss_curv_r1_5", "mean_curv_r1_5", "quad_dev_r1_5", "abs_algdist_r1_5", "gradnorm_r1_5",
                "laplacian_r1_5", "rmsc_r1_5", "gumbilicality_r1_5", "shpidx_r1_5", "saddleness_r1_5", "normal_gac_r1_5",
                "mean_qdev_r1_5", "linear_norm_r1_5", "spectral_r1_5", "frobenius_r1_5", "normal_bipgnorm_r1_5",
                "gauss_curv_r2", "mean_curv_r2", "quad_dev_r2", "abs_algdist_r2", "gradnorm_r2",
                "laplacian_r2", "rmsc_r2", "gumbilicality_r2", "shpidx_r2", "saddleness_r2", "normal_gac_r2",
                "mean_qdev_r2", "linear_norm_r2", "spectral_r2", "frobenius_r2", "normal_bipgnorm_r2",
                "gauss_curv_r3", "mean_curv_r3", "quad_dev_r3", "abs_algdist_r3", "gradnorm_r3",
                "laplacian_r3", "rmsc_r3", "gumbilicality_r3", "shpidx_r3", "saddleness_r3", "normal_gac_r3",
                "mean_qdev_r3", "linear_norm_r3", "spectral_r3", "frobenius_r3", "normal_bipgnorm_r3",
                "gauss_curv_r5", "mean_curv_r5", "quad_dev_r5", "abs_algdist_r5", "gradnorm_r5",
                "laplacian_r5", "rmsc_r5", "gumbilicality_r5", "shpidx_r5", "saddleness_r5", "normal_gac_r5",
                "mean_qdev_r5", "linear_norm_r5", "spectral_r5", "frobenius_r5", "normal_bipgnorm_r5"
            ],
            "target_val": "NaN",
            "strategy": "mean",
            "constant_val": 0,
            "impute_coordinates": false,
            "impute_references": false
        },
        {
            "predict": "PredictivePipeline",
            "model_path": "${outdir}/model/rf.pipe",
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


    # ---  SLURM BATCH SCRIPT  --- #
    # ---------------------------- #
    slurm_script="${JSONDIR}/tikhonov_sensit_only2nd_$(( $i + 1 )).sh"
    cat << EOF > ${slurm_script}
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH --mem 123GB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=albertoesmp@gmail.com

# ---   CONSTANTS   --- #
# --------------------- #
VL3DPY='/home/usc/ci/aep/git/vl3d/vl3d.py'
VL3DENV='/home/usc/ci/aep/git/vl3d/cesga/vl3d_cesga_env.sh'
PYTRUNCATE='/mnt/netapp2/Store_uscciaep/lidar_data/dales/vl3d/spec/laspy_truncate.py'


# ---   MAIN   --- #
# ---------------- #

# Load VL3D++ environment
source \${VL3DENV}

# Mine features
python3 \${VL3DPY} --pipeline "/mnt/netapp2/Store_uscciaep/lidar_data/dales/vl3d/spec/${mining_json}"

# Clip mined features by truncating extreme values
EOF

    for (( j=0 ; j < ${num_train_pclouds} ; ++j )); do
        lasfile="${outdir}/mined/${TRAIN_PCLOUDS[$j]}.las"
        echo 'python3 ${PYTRUNCATE} '"\"${lasfile}\"" >> ${slurm_script}
        echo 'echo -e "\n---------------------------------------------\n"' >> ${slurm_script}
    done

    cat << EOF >> ${slurm_script}
python3 \${PYTRUNCATE} "${outdir}/mined/${VAL_PCLOUD}.las"

# Train model
python3 \${VL3DPY} --pipeline "/mnt/netapp2/Store_uscciaep/lidar_data/dales/vl3d/spec/${train_json}"

# Validate model
python3 \${VL3DPY} --pipeline "/mnt/netapp2/Store_uscciaep/lidar_data/dales/vl3d/spec/${val_json}"

# Remove mined features
rm -fr "${outdir}/mined"

EOF
done

