#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH --mem 60GB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=albertoesmp@gmail.com


# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                              #
# BRIEF: Script to run the experiment of hybrid geometric descriptors          #
#        in the DALES dataset                                                  #
# ---------------------------------------------------------------------------- #


# ---   CONSTANTS   --- #
# --------------------- #
IN_JSON='/home/usc/ci/aep/git/twalg_scripts/hpc/geom2_data_mining.json'
VL3DPY='/home/usc/ci/aep/git/vl3d/vl3d.py'
VL3DENV='/home/usc/ci/aep/git/vl3d/cesga/vl3d_cesga_env.sh'
NTHREADS=10
MAX_TRIES=5
TMP_JSON='/home/usc/ci/aep/git/twalg_scripts/hpc/geom2_'${NTHREADS}'cores.json'


# ---   MAIN   --- #
# ---------------- #

# Load VL3D++ environment
source ${VL3DENV}
# Write TMP JSON
sed 's/"nthreads":\ -1/"nthreads":\ '${NTHREADS}'/g' "${IN_JSON}" > "${TMP_JSON}"
# Run data mining many tries
for (( try=0 ; ${try} < ${MAX_TRIES} ; ++try )); do
    python3 ${VL3DPY} --pipeline "${TMP_JSON}"
done
