# Airborne LiDAR Semantic Segmentation MVP - Updated Project Summary

Updated: 2026-06-14.

## Purpose

This folder contains the current technical and commercial material for the airborne LiDAR semantic segmentation and forest analytics MVP developed in `vl3d_galicia`.

The product converts airborne LiDAR point clouds into GIS-ready semantic layers for:

- ground;
- low vegetation;
- medium vegetation;
- high vegetation;
- buildings;
- water;
- forest structure and anomaly proxies.

The current deployable core is a **supervised point-wise segmentation model enhanced with label-free local geometric context, Taubin-Weingarten-inspired descriptors, robust normalization, focal loss and class-balanced training**.

Geo-JEPA and TW-JEPA are implemented research tracks. They are not the source of the strongest current downstream result. Frozen JEPA encoders and the tested DINOv2 fusion variants remained below the strongest supervised LiDAR models.

## Current model roles

There are now two important reference models rather than one universal "best model".

### 1. Internal mIoU and macro-F1 leader

Model:

`outputs/local_context_medium/geom_concat_h256_balancedval`

Internal test results:

| Metric | Value |
|---|---:|
| Overall accuracy | 86.49% |
| Macro-F1 | 82.60% |
| Mean IoU | 72.63% |

This remains the highest internal macro-F1 and mIoU run in the current comparison.

Against the original internal baseline:

| Metric | Baseline | Local-context model | Delta |
|---|---:|---:|---:|
| Overall accuracy | 83.03% | 86.49% | +3.45 pp |
| Macro-F1 | 74.82% | 82.60% | +7.78 pp |
| Mean IoU | 63.49% | 72.63% | +9.14 pp |

Per-class internal IoU:

| Class | Baseline IoU | Best IoU | Delta |
|---|---:|---:|---:|
| Ground | 72.63% | 75.60% | +2.97 pp |
| Low vegetation | 33.54% | 44.39% | +10.85 pp |
| Medium vegetation | 40.70% | 53.29% | +12.59 pp |
| High vegetation | 91.06% | 94.74% | +3.69 pp |
| Building | 45.75% | 69.88% | +24.13 pp |
| Water | 97.26% | 97.88% | +0.62 pp |

### 2. Deployment and geographic-generalization candidate

Model:

`outputs/domain_generalization/geom_robustnorm_cb20k_seed42`

Training configuration:

- 20,000 class-balanced training blocks;
- 4,000 class-balanced validation blocks;
- TW input features;
- 56-dimensional label-free geometric-context cache;
- robust coordinate normalization;
- block-robust spectral normalization;
- robust normalization of external spectral context;
- focal loss and class-balanced sampling;
- 35 epochs requested, 33 completed;
- early stopping enabled;
- best validation macro-F1: 0.816798.

Internal results:

| Metric | Robust model |
|---|---:|
| Overall accuracy | 86.71% |
| Macro-F1 | 82.48% |
| Mean IoU | 72.50% |

The robust model has the highest internal overall accuracy and nearly matches the original internal mIoU leader, while generalizing substantially better to geographically separate data.

Internal per-class metrics for the robust model:

| Class | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| Ground | 83.72% | 89.86% | 86.68% | 76.49% |
| Low vegetation | 67.98% | 55.36% | 61.02% | 43.91% |
| Medium vegetation | 64.72% | 73.96% | 69.03% | 52.71% |
| High vegetation | 98.79% | 95.63% | 97.19% | 94.53% |
| Building | 76.01% | 89.05% | 82.01% | 69.51% |
| Water | 98.55% | 99.29% | 98.92% | 97.86% |

## Three-seed internal stability

The original local-context configuration was evaluated with seeds 42, 1337 and 2026.

| Run | OA | Macro-F1 | mIoU |
|---|---:|---:|---:|
| Seed 42 | 86.49% | 82.60% | 72.63% |
| Seed 1337 | 85.92% | 82.00% | 71.87% |
| Seed 2026 | 84.61% | 79.97% | 69.52% |
| Mean | 85.67% | 81.52% | 71.34% |
| Worst run | 84.61% | 79.97% | 69.52% |

All three runs improved the tracked class IoUs against the original internal baseline.

## External geographic validation

### CAT-32 development benchmark

A stratified external set of 32 `CAT-2016` COL/CIR tile pairs was prepared outside Galicia:

- 48,696 blocks;
- 119,005,770 points;
- all six semantic classes represented;
- Galicia campaign prefixes excluded;
- TW normalization reused from the training pipeline;
- geometric context generated without labels.

This CAT-32 set was initially a clean external test, but it was later used to diagnose domain shift and design the robust-normalization strategy. It must therefore be described as an **external development and robustness benchmark**, not as the final untouched holdout.

Results:

| Model | OA | Macro-F1 | mIoU | Delta mIoU vs strong baseline |
|---|---:|---:|---:|---:|
| Strong TW baseline | 72.60% | 63.99% | 49.45% | - |
| Robust local-context model | 80.83% | 73.42% | 60.41% | +10.96 pp |

External development deltas:

- overall accuracy: +8.23 pp;
- macro-F1: +9.43 pp;
- mean IoU: +10.96 pp;
- class IoU improved in 6/6 classes.

### Fresh disjoint external holdout

A second external set was selected without tile overlap with the CAT-32 development benchmark:

- 32 `CAT-2016` tiles;
- 49,853 blocks;
- 130,588,254 points;
- 51.76% ground;
- 8.80% low vegetation;
- 10.51% medium vegetation;
- 28.07% high vegetation;
- 0.80% building;
- 0.064% water.

Results:

| Model | OA | Macro-F1 | mIoU | mIoU excluding water |
|---|---:|---:|---:|---:|
| Strong TW baseline | 77.33% | 51.38% | 40.62% | 48.33% |
| Robust local-context model | 83.27% | 61.80% | 50.98% | 60.62% |

Fresh-holdout improvement:

- overall accuracy: +5.94 pp;
- macro-F1: +10.42 pp;
- mean IoU: +10.35 pp;
- mIoU excluding water: +12.29 pp;
- class IoU improved in 6/6 classes against the strong baseline.

Per-class IoU on the fresh holdout:

| Class | Strong baseline | Robust model | Delta |
|---|---:|---:|---:|
| Ground | 73.43% | 79.97% | +6.54 pp |
| Low vegetation | 21.23% | 27.70% | +6.47 pp |
| Medium vegetation | 47.28% | 56.46% | +9.18 pp |
| High vegetation | 81.93% | 88.47% | +6.54 pp |
| Building | 17.80% | 50.50% | +32.70 pp |
| Water | 2.09% | 2.77% | +0.68 pp |

The fresh holdout demonstrates a real improvement in geographic generalization. The water result is not commercially representative because water accounts for only 0.064% of reliable points. Building is also sparse. Water and building require another external holdout or customer pilot with sufficient support before strong absolute claims are made for those classes.

## Domain-shift diagnosis

The external degradation of the first local-context model was traced mainly to geographic and sensor-domain shift rather than a total absence of useful signal.

Observed differences included:

- mean NIR: 0.5050 on the internal test vs 0.2955 on CAT-32;
- NIR Jensen-Shannon divergence: 0.2420;
- mean block z-range: 16.99 internally vs 61.58 externally;
- z-range p95: 42.54 internally vs 423.26 externally;
- different class priors across regions.

The effective correction was the combination of:

1. robust coordinate normalization;
2. robust spectral normalization;
3. label-free local geometric context;
4. genuine class-balanced selection over 20,000 training blocks.

Strong spectral augmentation alone and metric-height variants did not outperform this configuration.

## Forest intelligence layer

The forest-facing layer converts segmentation results and point geometry into practical indicators:

- vegetation ratio;
- high-vegetation ratio;
- forest-core ratio;
- canopy-cover proxy;
- canopy-height proxy;
- medium-high canopy proxy;
- canopy-gap flags;
- per-tile and per-cell metrics;
- rule-based anomaly candidates;
- review-priority layers.

These are operational proxies. They are not certified biomass estimates, species classifications or verified temporal-change products.

## Customer-facing deliverables

The current pipeline can produce:

- classified `.laz` or point-cloud outputs;
- semantic maps for ground, vegetation, buildings and water;
- CSV or GeoPackage-style grids;
- forest proxy layers;
- class-level and global metrics;
- confusion matrices;
- anomaly and disagreement maps;
- technical and executive reports;
- review queues for GIS or field teams.

## Commercial positioning

Recommended product wording:

> A supervised airborne LiDAR semantic-segmentation and forest-intelligence platform enhanced with label-free local geometry and robust normalization, with strong internal performance and demonstrated improvement over a strong baseline on a separate geographic holdout.

Recommended short wording:

> Turn airborne LiDAR into GIS-ready land-cover maps, forest indicators and prioritized review areas.

Do not present the current production model as a self-supervised Geo-JEPA winner. Geo-JEPA is a research track for future representation learning, domain adaptation and uncertainty-aware pretraining.

## Research findings

### Geo-JEPA and TW-JEPA

- implemented and evaluated;
- spatial masking and latent prediction are available;
- frozen downstream encoders did not beat the strongest supervised route;
- future work should reuse a stronger local/contextual 3D backbone.

### DINOv2

- direct concatenation was negative against the strong LiDAR baseline;
- gated fusion reduced part of the degradation but remained negative;
- DINOv2 should not be presented as a current performance gain.

### DINOv3 and V-JEPA-style work

Potential future directions include:

- aerial/satellite DINOv3 late fusion or distillation;
- dense predictive losses over LiDAR/raster tokens;
- self-supervised adaptation on unlabeled target-domain data;
- multitemporal modeling when aligned multi-date data exists.

Any transductive use of unlabeled external data must be documented as domain adaptation and evaluated on another untouched holdout.

## Current limitations and next steps

The most important next steps are:

1. obtain an additional external holdout with meaningful water and building support;
2. evaluate a stronger efficient local 3D backbone under the same protocol;
3. add calibrated uncertainty and confidence outputs;
4. package the full inference and merging pipeline for large raw tile collections;
5. validate forest proxies against field or authoritative forestry measurements;
6. continue self-supervised research over the stronger robust local-context backbone;
7. add multi-date change analysis only when aligned temporal data is available.

## Source-of-truth distinction

Use the following labels consistently:

- **Best internal mIoU run:** `geom_concat_h256_balancedval`.
- **Best deployment/generalization candidate:** `geom_robustnorm_cb20k_seed42`.
- **External development benchmark:** first CAT-32 set.
- **Fresh external holdout:** disjoint CAT-32 set.

This distinction prevents mixing internal leaderboard performance with deployment-oriented geographic robustness.
