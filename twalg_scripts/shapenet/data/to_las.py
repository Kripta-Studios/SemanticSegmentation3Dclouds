# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                              #
# BRIEF: Script to convert ShapeNet to a collection of LAS files               #
# ---------------------------------------------------------------------------- #


# ---   IMPORTS   --- #
# ------------------- #
from tqdm import tqdm
import laspy
import numpy as np
import json
import os


# ---   CONSTANTS   --- #
# --------------------- #
METADATA_PATH='metadata.json'

# ---   MAIN   --- #
# ---------------- #
if __name__ == '__main__':
    # Load JSON data
    with open(METADATA_PATH) as jsonf:
        meta = json.load(jsonf)
    # Iterate over objects metadata
    for obj_key, obj_val in tqdm(
        meta.items(),
        desc='Object-wise datasets',
        position=0
    ):
        obj_dir = obj_val['directory']
        pts_dir = os.path.join(obj_dir, 'points')
        outdir = f'../{os.path.join("pcloud", obj_key)}'
        os.makedirs(outdir, exist_ok=True)
        for _pts_file in tqdm(
            os.listdir(pts_dir),
            desc=f'{obj_key} dataset',
            position=1,
            leave=False
        ):
            # Read structure space (X)
            pts_file = os.path.join(pts_dir, _pts_file)
            X = np.loadtxt(pts_file)
            # Read labels (y)
            labels = obj_val['lables']
            y = np.full((X.shape[0], ), len(labels), np.int32)
            for cidx, label in enumerate(labels):
                try:
                    ycidxpath = os.path.join(
                        obj_dir,
                        'points_label',
                        label, _pts_file.replace('.pts', '.seg')
                    )
                    ycidx = np.loadtxt(ycidxpath)
                    mask = ycidx == 1
                    y[mask] = cidx
                except FileNotFoundError as fnferr:
                    pass
            # Build point cloud as LAS (in main memory)
            header = laspy.LasHeader(point_format=6, version="1.4")
            header.offsets = np.min(X, axis=0)
            header.scales = np.array([0.001, 0.001, 0.001])
            las = laspy.LasData(header)
            las.x, las.y, las.z = X[:, 0], X[:, 1], X[:, 2]
            las.classification = y
            # Write output point cloud as LAS file
            outpath = _pts_file.replace('.pts', '.las')
            outpath = os.path.join(outdir, outpath)
            las.write(outpath)
