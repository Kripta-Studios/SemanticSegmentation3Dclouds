Taubin-Weingarten algorithm paper
======================================

This repository contains the files necessary to reproduce the experiments.




VL3D++ framework
-------------------

First, the VL3D++ framework is necessary to run the algorithms, train the models, and evaluate them. It is an open-source software that can be downloaded from [its official repository](https://gitlab.com/catallactical/vl3dpp).



Shapenet scripts
--------------------

***shapenet/data*** :
The scripts to transform the the shapenet datasets to LAS format


***shapenet/experiments/kfold*** :
The scripts to run the stratified k-folding experiments on the airplane, car, chair, guitar, knife, lamp, motorbike, and pistol datasets.

***shapenet/experiments/fixed*** :
The scripts to run the ablation study on the chair, pistol, motorbike, and car datasets.


DALES scripts
----------------

*dales* :
The scripts to run the first-order, second-order, and hybrid model experiments on the DALES dataset. It also contains the bash scripts to generate the SLURM scripts to run the sensitivity analysis in the FinisTerrae-III supercomputer.



HPC scripts
----------------

*hpc* :
The scripts to run the experiments of the scalability analysis from 1 to 32 cores using the FinisTerrae-III with SLURM.



Theoretical validation
------------------------

*theoretical* :
The script to run the theoretical experiments and the Jupyter notebook for Sagemath used to generate the analyitical references. It also contains an alternative version of the notebook where normal noise along the gradient direction can be modelled through its standard deviation.

*theoretical/data* :
The point clouds corresponding to the theoretical experiments.





Cite
------
**DOI**: https://doi.org/10.1016/j.cam.2026.117569

**Bibtex**:
```
@article{ESMORIS2026117569,
title = {{Introducing the Taubin-Weingarten algorithm to compute second-order geometric descriptors in 3D point clouds}},
journal = {{Journal of Computational and Applied Mathematics}},
volume = {485},
pages = {117569},
year = {2026},
issn = {0377-0427},
doi = {https://doi.org/10.1016/j.cam.2026.117569},
url = {https://www.sciencedirect.com/science/article/pii/S0377042726002177},
author = {Alberto M. Esmorís and Xabier García-Martínez and Manuel Ladra and José Carlos Cabaleiro and Francisco {Fernández Rivera}},
}
```

