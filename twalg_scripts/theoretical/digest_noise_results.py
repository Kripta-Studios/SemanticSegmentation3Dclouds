# ---   IMPORTS   --- #
# ------------------- #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# ---   RESULTS   --- #
# ------------------- #
# Directory for the noise experiments on the hyperbolic paraboloid
HYPERBOLIC_PARABOLOID_RESULTS='/ext4/lidar_data/math/geomfeats2nd/noise/paraboloid/'
# Directory for the noise experiments on the toroid
TOROID_RESULTS='/ext4/lidar_data/math/geomfeats2nd/noise/toroid/'
# Directory for the noise experiments on the one-sheet hyperboloid
ONE_SHEET_HYPERBOLOID_RESULTS='/ext4/lidar_data/math/geomfeats2nd/noise/hyperboloid/'
# Directory for the noise experiments on the catenoid
CATENOID_RESULTS='/ext4/lidar_data/math/geomfeats2nd/noise/catenoid/'


# ---  METHODS  --- #
# ----------------- #
def digest_results(results_dir, subplot_idx, variety, geom_feat):
    # Iterate over results
    dirs = os.listdir(results_dir)
    sigmas, maes, rmses, pearsons, spearmans = [], [], [], [], []
    for path in dirs:
        # Skip files
        fpath = os.path.join(results_dir, path)
        if not os.path.isdir(fpath):
            continue
        # Skip img directory
        if path == 'img':
            continue
        # Extract sigma
        sigma = np.log10(float(path[path.rfind('sigma')+len('sigma'):]))
        sigmas.append(sigma)
        # Read report
        rpath = os.path.join(fpath, 'report', 'regression_eval.log')
        df = pd.read_csv(rpath, sep=',', skiprows=1, header=0, index_col=0)
        df.columns = df.columns.str.strip()
        df.index = df.index.str.strip()
        mae = float(df['MAE'].loc[geom_feat])
        maes.append(mae)
        rmse = float(df['RMSE'].loc[geom_feat])
        rmses.append(rmse)
        pearson = float(df['Pearson'].loc[geom_feat])
        pearsons.append(pearson)
        spearman = float(df['Spearman'].loc[geom_feat])
        spearmans.append(spearman)
    # Prepare data (e.g., sort)
    S = np.argsort(sigmas)
    sigmas = np.array(sigmas)[S]
    maes = np.array(maes)[S]
    rmses = np.array(rmses)[S]
    pearsons = np.array(pearsons)[S]
    spearmans = np.array(spearmans)[S]
    errors = np.hstack([maes, rmses])
    correlations = np.hstack([pearsons, spearmans])
    # Prepare error subplot
    ax = fig.add_subplot(2, 4, subplot_idx)
    title = '## ERROR ##'
    if geom_feat == 'twGaussC':
        title = f'{variety}\nGaussian curvature'
    elif geom_feat == 'twMeanC':
        title = f'{variety}\nMean curvature'
    ax.set_title(title, fontsize=10)
    ax.plot(
        sigmas, maes, label='MAE', lw=2, color='tab:blue',
        marker='v', markeredgecolor='black', markersize=7
    )
    ax.plot(
        sigmas, rmses, label='RMSE', lw=2, color='tab:red',
        marker='X', markeredgecolor='black', markersize=7.5
    )
    # Prepare correlation subplot
    ax2 = fig.add_subplot(2, 4, subplot_idx+4)
    ax2.plot(
        sigmas, pearsons, label='Pearson', lw=2, color='tab:green',
        marker='o', markeredgecolor='black', markersize=6.33
    )
    ax2.plot(
        sigmas, spearmans, label='Spearman', lw=2, color='tab:orange',
        marker='s', markeredgecolor='black', markersize=6
    )
    # Format subplot
    ax.set_xticks(np.linspace(np.min(sigmas), np.max(sigmas), 6))
    ax2.set_xticks(np.linspace(np.min(sigmas), np.max(sigmas), 6))
    ax.set_yticks(np.linspace(0, np.max(errors), 6))
    ax2.set_yticks(np.linspace(-1, 1, 6))
    if subplot_idx == 1:
        ax.legend(loc='upper left')
        ax.set_ylabel('Error')
        ax2.legend(loc='lower left')
        ax2.set_ylabel('Correlation')
    ax2.set_xlabel(r'$\log_{10}(\sigma)$')
    ax.grid('both')
    ax2.grid('both')
    ax.set_axisbelow(True)
    ax2.set_axisbelow(True)


# ---   MAIN   --- #
# ---------------- #
if __name__ == '__main__':
    # Prepare figure
    fig = plt.figure(figsize=(12, 5))
    # Digest results
    digest_results(
        HYPERBOLIC_PARABOLOID_RESULTS, 1, 'Hyperbolic paraboloid', 'twMeanC'
    )
    digest_results(
        TOROID_RESULTS, 2, 'Toroid', 'twMeanC'
    )
    digest_results(
        ONE_SHEET_HYPERBOLOID_RESULTS, 3, 'One-sheet hyperboloid', 'twGaussC'
    )
    digest_results(
        CATENOID_RESULTS, 4, 'Catenoid', 'twGaussC'
    )
    # Format figure
    fig.tight_layout()
    # Save and show figure
    plt.savefig('/tmp/noise_on_varieties.jpg', dpi=300)
    plt.show()
