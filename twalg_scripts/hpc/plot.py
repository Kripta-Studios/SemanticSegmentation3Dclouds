#!/bin/python3

# ---------------------------------------------------------------------------- #
# AUTHOR : Alberto M. Esmoris Pena                                             #
# BRIEF : Script to generate plots from the results in the CSV files           #
# ---------------------------------------------------------------------------- #

# ---   IMPORTS   --- #
# ------------------- #
import matplotlib.pyplot as plt
import numpy as np

# ---   CONSTANTS   --- #
# --------------------- #
CORES_COL_IDX=0
TEXEC_COL_IDX=1
SPEEDUP_COL_IDX=3

# ---   METHODS   --- #
# ------------------- #
def plot_case(fig, ax, xtwin, data, color, label=False):
    # Handle labels
    ax_label = None
    xtwin_label = None
    if label:
        ax_label = 'Runtime'
        xtwin_label = 'Speedup'
    # Handle plots
    ax.plot(
        data[:, CORES_COL_IDX], data[:, TEXEC_COL_IDX],
        lw=2, ls='--', zorder=4, color=color, label=ax_label
    )
    xtwin.plot(
        data[:, CORES_COL_IDX], data[:, SPEEDUP_COL_IDX],
        lw=2, zorder=5, color=color, label=xtwin_label
    )

# ---   MAIN   --- #
# ---------------- #
if __name__ == "__main__":
    # Load data
    small = np.loadtxt('results_5080_54435.csv', delimiter=',')
    mid = np.loadtxt('results_5185_54485.csv', delimiter=',')
    big = np.loadtxt('results_5110_54320.csv', delimiter=',')
    # Prepare plot
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Runtime (s) and speedup', fontsize=12)
    xtwin = ax.twinx()
    # Plot theoretical speedup limit
    xtwin.plot(
        [0, np.max(small[:, CORES_COL_IDX])],
        [0, np.max(small[:, CORES_COL_IDX])],
        color='black',
        lw=5,
        zorder=3,
        label='Theoretical speedup'
    )
    # Plot data
    plot_case(fig, ax, xtwin, small, color='tab:orange')
    plot_case(fig, ax, xtwin, mid, color='tab:green')
    plot_case(fig, ax, xtwin, big, color='tab:blue', label=True)
    # Handle legend
    labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    plots, labels = [sum(lol, []) for lol in zip(*labels)]
    fig.legend(
        plots, labels,
        loc='upper center', ncol=1, fontsize=12,
        bbox_to_anchor=[0.5, 0.9]
    )
    # Format plot
    ax.set_xlabel('Number of cores', fontsize=12)
    ax.set_ylabel('Runtime (s)', fontsize=12)
    xtwin.set_ylabel('Speedup', fontsize=12)
    ax.grid('both')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='both', labelsize=12)
    xtwin.tick_params(axis='both', which='both', labelsize=12)
    fig.tight_layout()
    # Show plot
    plt.savefig('speedup.jpg', dpi=300)
    plt.show()


