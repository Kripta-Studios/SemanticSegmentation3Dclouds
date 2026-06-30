#!/bin/bash

# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Script to compute the second order geometric descriptors with VL3D++
#        and evaluate them including also the results from CloudCompare and the
#        analytic references computed with SageMath.
# ---------------------------------------------------------------------------- #

# ---   CONSTANTS   --- #
# --------------------- #
VL3DENV="/home/uadmin/git/vl3d/venv/bin/activate"
VL3DPY="/home/uadmin/git/vl3d/vl3d.py"


# ---   MAIN   --- #
# ---------------- #
source "${VL3DENV}"
python "${VL3DPY}" --pipeline curvature_computation_tw.json && \
    python "${VL3DPY}" --pipeline curvature_computation_pca.json && \
    python "${VL3DPY}" --pipeline curvature_computation_svd.json && \
    python "${VL3DPY}" --pipeline regression_evaluation.json
