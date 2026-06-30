# ---   IMPORTS   --- #
# ------------------- #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ---   RESULTS   --- #
# ------------------- #
# Directory for the error-radius experiments with noise on the one-sheet hyperb.
HYPERBOLOID_RESULTS='/home/uadmin/tmp/radius_noise_error/hyperboloid/report'
# Directory for the error-radius experiments with noise on the hyperb. parabol.
PARABOLOID_RESULTS='/home/uadmin/tmp/radius_noise_error/paraboloid/report'
# Directory for the error-radius experiments with noise on the ellipsoid
ELLIPSOID_RESULTS='/home/uadmin/tmp/radius_noise_error/ellipsoid/report'
# Directory for the error-radius experiments with noise on the cylinder
CYLINDER_RESULTS='/home/uadmin/tmp/radius_noise_error/cylinder/report'
# Directory for the error-radius experiments with noise on the catenoid
CATENOID_RESULTS='/home/uadmin/tmp/radius_noise_error/catenoid/report'
# Directory for the error-radius experiments with noise on the toroid
TOROID_RESULTS='/home/uadmin/tmp/radius_noise_error/toroid/report'

# ---   CONSTANTS   --- #
# --------------------- #
# Colors for plots
COLORS = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:purple',
    'tab:red',
    'tab:olive'
]
# Line styles for plots
LINE_STYLES = [
    '-'
]


# ---  METHODS  --- #
# ----------------- #
def digest_results(results_dir, case_idx, variety, geom_feat):
    # Iterate over results
    dirs = os.listdir(results_dir)
    radii, meaes, pearsons = [], [], []
    for path in dirs:
        # Skip everything but evaluation result files
        fpath = os.path.join(results_dir, path)
        if not os.path.isfile(fpath) or not fpath[-8:] == 'eval.log':
            continue
        # Extract radius
        radii.append(float(path[1:path.find('_')]))
        # Read report
        df = pd.read_csv(fpath, sep=',', skiprows=1, header=0, index_col=0)
        df.columns = df.columns.str.strip()
        df.index = df.index.str.strip()
        meae = float(df['MeAE'].loc[geom_feat])
        meaes.append(meae)
        pearson = float(df['Pearson'].loc[geom_feat])
        pearsons.append(pearson)
    # Prepare data (e.g., sort)
    S = np.argsort(radii)
    radii = np.array(radii)[S]
    meaes = np.array(meaes)[S]
    pearsons = np.array(pearsons)[S]
    # Plot MeAE
    ax.plot(
        radii, meaes,
        lw=2,
        ls=LINE_STYLES[case_idx % len(LINE_STYLES)],
        color=COLORS[case_idx % len(COLORS)],
        label=variety,
        marker='o', markeredgecolor='black', markersize=4, markeredgewidth=0.5,
        zorder=5
    )
    # Plot Pearson
    ax2.plot(
        radii, pearsons,
        lw=2,
        ls=LINE_STYLES[case_idx % len(LINE_STYLES)],
        color=COLORS[case_idx % len(COLORS)],
        marker='o', markeredgecolor='black', markersize=4, markeredgewidth=0.5,
        zorder=5
    )
    # Return
    return radii, meaes, pearsons

def aggregate_results():
    # Find min MeAE radius
    sum_meaes = hyperboloid_meaes + paraboloid_meaes + ellipsoid_meaes \
        + cylinder_meaes + catenoid_meaes + toroid_meaes
    meae_argmin = np.argmin(sum_meaes)
    # Find max Pearson radius
    sum_pearsons = hyperboloid_pearsons + paraboloid_pearsons + \
        ellipsoid_pearsons + cylinder_pearsons + catenoid_pearsons + \
        toroid_pearsons
    pearson_argmax = np.argmax(sum_pearsons)
    # Plot vertical line on min MeAE radius
    ax.axvline(
        radii[meae_argmin], lw=2, ls='--', color='black',
        label=f'Min average MeAE',
        #label=f'Min average MeAE ({sum_meaes[meae_argmin]/6:.2f})',
        zorder=3
    )
    # Plot vertical line on max Pearson radius
    ax2.axvline(
        radii[pearson_argmax], lw=2, ls='--', color='black',
        label=f'Max average $r$',
        #label=f'Max average $r$ ({sum_pearsons[pearson_argmax]/6:.2f})',
        zorder=3
    )


# ---   MAIN   --- #
# ---------------- #
if __name__ == '__main__':
    # Prepare figure
    fig = plt.figure(figsize=(7, 3))
    # Prepare axes
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # Digest results
    radii, hyperboloid_meaes, hyperboloid_pearsons = digest_results(
        HYPERBOLOID_RESULTS, 0, 'One-sheet hyperboloid', 'twGaussC'
    )
    paraboloid_meaes, paraboloid_pearsons = digest_results(
        PARABOLOID_RESULTS, 1, 'Hyperbolic paraboloid', 'twMeanC'
    )[1:]
    ellipsoid_meaes, ellipsoid_pearsons = digest_results(
        ELLIPSOID_RESULTS, 2, 'Ellipsoid', 'twGaussC'
    )[1:]
    cylinder_meaes, cylinder_pearsons = digest_results(
        CYLINDER_RESULTS, 3, 'Cylinder', 'twMeanC'
    )[1:]
    catenoid_meaes, catenoid_pearsons = digest_results(
        CATENOID_RESULTS, 4, 'Catenoid', 'twGaussC'
    )[1:]
    toroid_meaes, toroid_pearsons = digest_results(
        TOROID_RESULTS, 5, 'Toroid', 'twMeanC'
    )[1:]
    # Aggregate results
    aggregate_results()
    # Format axes
    ax.grid('both')
    ax.set_axisbelow(True)
    ax.set_ylabel('Median absolute error (MeAE)')
    ax.set_xlabel('Radius')
    ax.legend(loc='best')
    ax2.grid('both')
    ax2.set_axisbelow(True)
    ax2.set_ylabel(r'Pearson correlation ($r$)')
    ax2.set_xlabel('Radius')
    ax2.legend(loc='best')
    # Format figure
    fig.tight_layout()
    # Save and show figure
    plt.savefig('/tmp/radius_noise_error.jpg', dpi=300)
    plt.show()
