# ---   IMPORTS   --- #
# ------------------- #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# ---   RESULTS   --- #
# ------------------- #
# Directory with the results of the sensitivity analysis for the eigenthreshold
EIGEN_RESULTS='/ext4/lidar_data/math/geomfeats2nd/sensitivity/eigen_only2nd/out'
# Directory with the results of the sensitivity analysis for the tikhonov
TIKHONOV_RESULTS='/ext4/lidar_data/math/geomfeats2nd/sensitivity/tikhonov_only2nd/out'
# Eigenvalue fnames
EIGEN_FNAMES = [
    'quad_dev', 'frobenius', 'spectral'
]
# Polynomial fnames
POLY_FNAMES = [
    'abs_algdist', 'linear_norm'
]
# Differential fnames
DIFF_FNAMES = [
    'mean_qdev', 'gradnorm', 'saddleness', 'laplacian', 'normal_gac',
    'normal_bipgnorm'
]
# Curvature fnames
CURV_FNAMES = [
    'rmsc', 'gumbilicality', 'gauss_curv', 'mean_curv', 'shpidx'
]

# ---  METHODS  --- #
# ----------------- #
def digest_results(results_dir, subplot_idx, hparam_name, hparam_math):
    # Iterate over results
    dirs = os.listdir(results_dir)
    vals, eigen_fimps, poly_fimps, diff_fimps, curv_fimps = [], [], [], [], []
    for dirpath in dirs:
        # Extract hyperparameter value
        val = np.log10(float(dirpath[dirpath.rfind('_')+1:]))
        # Extract feature importances
        dirpath = os.path.join(results_dir, dirpath, 'RF_importance.log')
        df = pd.read_csv(dirpath, sep=',', skiprows=2, header=None)
        # Aggregate feature importances
        fnames = np.array([
            x[:x.rfind('_r')]
            for x in [x.replace(' ', '') for x in df[0].to_list()]
        ])
        fimps = df[1].to_numpy()
        eigen_imp, poly_imp, diff_imp, curv_imp = 0, 0, 0, 0
        for i, fname in enumerate(fnames):
            fimp = fimps[i]
            if fname in EIGEN_FNAMES:
                eigen_imp += fimp
            elif fname in POLY_FNAMES:
                poly_imp += fimp
            elif fname in DIFF_FNAMES:
                diff_imp += fimp
            elif fname in CURV_FNAMES:
                curv_imp += fimp
        # Append hyperparameter values and eigen importances
        vals.append(val)
        eigen_fimps.append(eigen_imp)
        poly_fimps.append(poly_imp)
        diff_fimps.append(diff_imp)
        curv_fimps.append(curv_imp)
    # Prepare data (e.g., sort)
    S = np.argsort(vals)
    vals = np.array(vals)[S]
    eigen_fimps, poly_fimps = np.array(eigen_fimps)[S], np.array(poly_fimps)[S]
    diff_fimps, curv_fimps = np.array(diff_fimps)[S], np.array(curv_fimps)[S]
    fimps = np.hstack([eigen_fimps, poly_fimps, diff_fimps, curv_fimps])
    # Prepare subplot
    ax = fig.add_subplot(1, 2, subplot_idx)
    ax.set_title(hparam_name, fontsize=10)
    ax.plot(
        vals, eigen_fimps,
        label='Eigenvalue', color='tab:blue', lw=2,
        marker='X', markeredgecolor='black'
    )
    ax.plot(
        vals, poly_fimps,
        label='Polynomial', color='tab:green', lw=2,
        marker='v', markeredgecolor='black'
    )
    ax.plot(
        vals, diff_fimps,
        label='Differential', color='tab:red', lw=2,
        marker='s', markeredgecolor='black'
    )
    ax.plot(
        vals, curv_fimps,
        label='Curvature', color='tab:orange', lw=2,
        marker='o', markeredgecolor='black'
    )
    # Format figure
    ax.set_xticks(np.linspace(np.min(vals), np.max(vals), 4))
    ax.set_yticks(np.linspace(0, 0.6, 4))
    ax.set_xlabel(r'$\log_{10}('f'{hparam_math}'r')$')
    if subplot_idx==1:
        ax.set_ylabel('Feature importance')
        ax.legend(bbox_to_anchor=(1.1, 1.05))
    ax.grid('both')
    ax.set_axisbelow(True)


# ---   MAIN   --- #
# ---------------- #
if __name__ == '__main__':
    # Prepare figure
    fig = plt.figure(figsize=(8, 3))
    # Digest results
    digest_results(EIGEN_RESULTS, 1, 'Eigenthreshold', r'\epsilon')
    digest_results(TIKHONOV_RESULTS, 2, 'Tikhonov parameter', r'\tau')
    # Format figure
    fig.tight_layout()
    # Save and show figure
    plt.savefig('/tmp/sensitivity_on_dales.jpg', dpi=300)
    plt.show()
