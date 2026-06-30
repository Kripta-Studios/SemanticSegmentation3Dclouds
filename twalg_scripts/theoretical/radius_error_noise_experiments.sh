#!/bin/bash

# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Script to compute the second order geometric descriptors with VL3D++
#        exploring a radii interval to find the expected error under
#        realistically acceptable noise conditions.
# ---------------------------------------------------------------------------- #

# ---   CONSTANTS   --- #
# --------------------- #
VL3DENV="/home/uadmin/git/vl3d/venv/bin/activate"
VL3DPY="/home/uadmin/git/vl3d/vl3d.py"
RADII=("0.20" "0.25" "0.30" "0.35" "0.40" "0.45" "0.50" "0.55" "0.60" "0.65" "0.70" "0.75" "0.80" "0.85" "0.90" "0.95" "1.00")
TMPJSON="/home/uadmin/tmp/radius_noise_error_pipeline.json"
TMPOUT="/home/uadmin/tmp/radius_noise_error"


# ---   MAIN   --- #
# ---------------- # 
# Load environment
source "${VL3DENV}"

# Run one experiment per radius
for (( i=0 ; i < ${#RADII[@]} ; ++i )); do
    # Prepare variables
    radius=${RADII[$i]}
    # Craft pipeline JSON
    cat << EOF > "${TMPJSON}"
{
    "in_pcloud": [
        "/ext4/lidar_data/math/geomfeats2nd/noise/cylinder/sage_elc_h1_5_a0_6_b0_5_sigma0.0500000000000000.laz",
        "/ext4/lidar_data/math/geomfeats2nd/noise/hyperboloid/sage_osh_a0_5_b0_75_c1_25_sigma0.0500000000000000.laz",
        "/ext4/lidar_data/math/geomfeats2nd/noise/paraboloid/sage_hpa_h1_a0_6_b0_9_sigma0.0500000000000000.laz",
        "/ext4/lidar_data/math/geomfeats2nd/noise/toroid/sage_tor_r0_5_R1_0_sigma0.0500000000000000.laz",
        "/ext4/lidar_data/math/geomfeats2nd/noise/ellipsoid/sage_ell_a1_3_b1_1_c0_9_sigma0.0500000000000000.laz",
        "/ext4/lidar_data/math/geomfeats2nd/noise/catenoid/sage_cat_h1_3_a1_sigma0.0500000000000000.laz"
    ],
    "out_pcloud": [
        "${TMPOUT}/cylinder/*",
        "${TMPOUT}/hyperboloid/*",
        "${TMPOUT}/paraboloid/*",
        "${TMPOUT}/toroid/*",
        "${TMPOUT}/ellipsoid/*",
        "${TMPOUT}/catenoid/*"
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
        "radius": ${radius},
        "k": 1024,
        "lower_bound": 0,
    	"upper_bound": 0
      },
      "fnames": [
        "gauss_curv_full", "mean_curv_full", 
        "full_maxabscurv", "full_minabscurv",
        "full_umbilic_dev", "full_shape_index"
      ],
      "frenames": [
        "gauss_curv_tw", "mean_curv_tw", 
        "maxabscurv_tw", "minabscurv_tw",
        "umbilic_dev_tw", "shpidx_tw"
      ],
      "non_degenerate_eigenthreshold": 0.00001,
      "tikhonov_parameter": 0.0000001,
      "nthreads": -1
    },
    {
		"eval": "RegressionEvaluator",
		"metrics": [
			"RMSE", "MAE",
			"MeSE", "RMeSE", "MeAE",
			"DevSE", "RDevSE", "DevAE",
			"Pearson", "Spearman"
		],
		"cases": [
			["sage_gauss_curv", "gauss_curv_tw"],
			["sage_mean_curv", "mean_curv_tw"]
		],
		"cases_renames":[
			["twGaussC"],
			["twMeanC"]
		],
		"outer_correlations": null,
		"outlier_filter": null,
		"outlier_param": 0,
		"regression_report_path": "*/report/r${radius}_regression_eval.log",
		"outer_report_path": "*/report/r${radius}_outer_eval.log",
		"distribution_report_path": "*/report/r${radius}_regression_distrb.log",
		"regression_pcloud_path": "*/report/r${radius}_regression_eval.las",
		"regression_plot_path": "*/plot/r${radius}_regression_plot.png",
		"regression_hist2d_path": "*/plot/r${radius}_regression_hist2d.png",
		"residual_plot_path": "*/plot/r${radius}_residual_plot.png",
		"residual_hist2d_path": "*/plot/r${radius}_residual_hist2d.png",
		"scatter_plot_path": "*/plot/r${radius}_scatter_plot.png",
		"scatter_hist2d_path": "*/plot/r${radius}_scatter_hist2d.png",
		"qq_plot_path": "*/plot/r${radius}_qq_plot.png",
		"summary_plot_path": "*/plot/r${radius}_summary_plot.png",
		"nthreads": -1
	}
  ]
}
EOF
    python "${VL3DPY}" --pipeline "${TMPJSON}"
done
