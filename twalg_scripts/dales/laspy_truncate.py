# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Script to truncate features in a LAS file.
# ARGUMENT 1: Path to the LAS file to be truncated
# ---------------------------------------------------------------------------- #

# ---   IMPORTS   --- #
# ------------------- #
import laspy
import numpy as np
import sys
import time

# ---  CONSTANTS  --- #
# ------------------- #
INTERVAL = np.array([-1000.0, 1000.0])

# ---   MAIN   --- #
# ---------------- #
start_main = time.perf_counter()

# Load LAS file
start = time.perf_counter()
las = laspy.read(sys.argv[1])
end = time.perf_counter()
print(f'Loaded LAS file "{sys.argv[1]}" in {end-start:.3f} seconds.')

# Obtain features
start = time.perf_counter()
fnames = [fname for fname in las.point_format.extra_dimension_names]
F = np.array([las[fname] for fname in fnames])
end = time.perf_counter()
print(
    f'Extracted {F.shape[1]} features for {F.shape[0]} points '
    f'in {end-start:.3f} seconds.'
)

# Clip values
start = time.perf_counter()
F = np.clip(F, INTERVAL[0], INTERVAL[1])
end = time.perf_counter()
print(f'Clipped features in {end-start:.3f} seconds.')

# Update LAS with truncated features
start = time.perf_counter()
for fname, fi in zip (fnames, F):
    las[[fname]] = fi
end = time.perf_counter()
print(f'LAS updated in {end-start:.3f} seconds.')

# Export updated LAS to file
start = time.perf_counter()
las.write(sys.argv[1])
end = time.perf_counter()
print(f'Updated LAS file written in {end-start:.3f} seconds.')

# Report total time
end_main = time.perf_counter()
print(f'The whole process took {end-start:.3f} seconds.')
