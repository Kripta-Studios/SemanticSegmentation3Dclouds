# Geo-JEPA / Forest-JEPA MVP Demo

## Que es Geo-JEPA en este repo

Geo-JEPA es el pipeline autosupervisado sobre LiDAR/PNOA Galicia que aprende embeddings de bloques de puntos con una tarea JEPA: predice la representacion latente de una region espacial oculta usando contexto visible. El encoder se reutiliza despues en segmentacion semantica de puntos.

Componentes:

- Loaders PNOA `.laz` COL/CIR con XYZ, RGB, intensidad, clasificacion y NIR.
- GeoPointJEPA con masking espacial, random y por altura.
- TW-JEPA con tarea auxiliar Taubin-Weingarten.
- LeJEPA-style con SIGReg y target encoder compartido por defecto.
- Downstream: ground, low vegetation, medium vegetation, high vegetation, building, water.
- Validacion MVP: baseline, frozen linear probe, frozen MLP probe, semi-frozen y full fine-tune.

## Que es Forest-JEPA en este repo

Forest-JEPA es una primera capa forestal encima de Geo-JEPA. No estima biomasa cientifica ni especies. Evalua proxies forestales utiles a partir de LiDAR/PNOA:

- vegetacion agregada;
- forest core: medium + high vegetation;
- high vegetation;
- canopy height proxy;
- canopy cover proxy;
- canopy gaps;
- anomalias forestales basadas en reglas.

## Datos esperados

Por defecto:

```text
data/raw/pnoa_galicia/*-COL.laz
data/raw/pnoa_galicia/*-CIR.laz
```

Cada tesela debe tener su par COL/CIR.

## Quick Demo

```powershell
cd .\vl3d_galicia
powershell -ExecutionPolicy Bypass -File .\run_forest_demo.ps1 -Python python -Quick -Workers 8 -PrepareWorkers 8 -BatchSize 24 -JepaBatchSize 48
```

## Geo-JEPA Pilot Solo

```powershell
cd .\vl3d_galicia
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Python python -Quick -Workers 8 -PrepareWorkers 8 -BatchSize 24 -JepaBatchSize 48
```

## DINOv3 Raster Fusion

El baseline mas fuerte actual es `TW + focal loss + balanced sampler`. Para intentar superarlo sin tirar ese trabajo, el repo incluye una ruta DINOv3-raster-fusion:

- rasteriza cada bloque LiDAR/PNOA a una imagen multicanal;
- extrae dense features DINOv3/DINOv2 o features estadisticas `stat` para smoke;
- proyecta esas features a cada punto;
- concatena `XYZ + COL/CIR/intensidad/NIR + TW + DINO`;
- entrena con el mismo trainer, sampler y metricas que el baseline fuerte.

Smoke sin descargar DINO:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dino_fusion.ps1 `
  -Python python `
  -Backend stat `
  -MaxFeatureBlocksPerSplit 8 `
  -MaxTrainBlocks 8 `
  -MaxValBlocks 4 `
  -MaxTestBlocks 8 `
  -TrainEpochs 1 `
  -Workers 2 `
  -BatchSize 4 `
  -FeatureRoot data/processed/dino_smoke `
  -OutRoot outputs/dino_smoke
```

Experimento real DINOv3 ViT-S/16:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dino_fusion.ps1 `
  -Python python `
  -Backend hf `
  -DinoModel facebook/dinov3-vits16-pretrain-lvd1689m `
  -FeatureRoot data/processed/galicia_blocks_medium_dino_v3s16 `
  -OutRoot outputs/dino_fusion `
  -Workers 8 `
  -BatchSize 24 `
  -TrainEpochs 30 `
  -MaxTrainBlocks 12000 `
  -MaxValBlocks 2000
```

Nota: los checkpoints DINOv3 oficiales son gated. Este comando requiere acceso aceptado en Hugging Face y autenticacion local (`huggingface-cli login`) o pesos descargados via `-Backend torchhub`.

Alternativa publica inmediata con DINOv2 ViT-S/14:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dino_fusion.ps1 `
  -Python python `
  -Backend dinov2 `
  -DinoModel dinov2_vits14 `
  -FeatureRoot data/processed/galicia_blocks_medium_dinov2s14 `
  -OutRoot outputs/dinov2_fusion `
  -Workers 8 `
  -BatchSize 24 `
  -TrainEpochs 30
```

Experimento DINOv3 satelite, mas pesado:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dino_fusion.ps1 `
  -Python python `
  -Backend hf `
  -DinoModel facebook/dinov3-vitl16-pretrain-sat493m `
  -Normalize sat493m `
  -FeatureRoot data/processed/galicia_blocks_medium_dino_sat493m `
  -OutRoot outputs/dino_fusion_sat `
  -Workers 8 `
  -BatchSize 24 `
  -TrainEpochs 30
```

Si los pesos DINOv3 estan descargados desde el repo oficial de Meta:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dino_fusion.ps1 `
  -Python python `
  -Backend torchhub `
  -DinoRepoDir C:\ruta\a\dinov3 `
  -DinoModel dinov3_vits16 `
  -DinoWeights C:\ruta\a\dinov3_vits16.pth
```

## Experimentos Geo-JEPA

| Experimento | Encoder | Probe | Output |
|---|---|---|---|
| baseline | entrenado desde cero | MLP | `outputs/pilot/baseline` |
| jepa_frozen_linear | congelado | lineal | `outputs/pilot/jepa_frozen_linear` |
| jepa_frozen_mlp | congelado | MLP | `outputs/pilot/jepa_frozen_mlp` |
| jepa_semifrozen | LR bajo, default 0.01 | MLP | `outputs/pilot/jepa_semifrozen` |
| jepa_full_finetune | entrenable | MLP | `outputs/pilot/jepa_full_finetune` |

## Outputs Geo-JEPA

```text
outputs/pilot/comparison/test_comparison.csv
outputs/pilot/comparison/test_comparison.md
outputs/pilot/*/metrics.json
outputs/pilot/*/per_class_metrics.csv
outputs/pilot/*/confusion_matrix.csv
outputs/pilot/*/confusion_matrix.png
outputs/pilot/*/test_predictions.npy
outputs/pilot/*/run_config.json
```

## Outputs DINO Fusion

```text
data/processed/galicia_blocks_medium_dino*/feature_config.json
data/processed/galicia_blocks_medium_dino*/train/*.pt
data/processed/galicia_blocks_medium_dino*/val/*.pt
data/processed/galicia_blocks_medium_dino*/test/*.pt
outputs/dino_fusion/dino_tw_fusion/metrics.json
outputs/dino_fusion/dino_tw_fusion/per_class_metrics.csv
outputs/dino_fusion/comparison/test_comparison.csv
outputs/dino_fusion/comparison/test_comparison.md
```

`test_comparison.csv` y `test_comparison.md` incluyen veredicto automatico contra el baseline:

- `delta_OA_vs_baseline`
- `delta_macro_F1_vs_baseline`
- `delta_mIoU_vs_baseline`
- deltas IoU por clase;
- `improved_class_iou_count`;
- `worsened_class_iou_count`;
- `verdict_vs_baseline`: `POSITIVE`, `MIXED`, `NEGATIVE`.

## Outputs Demo Visual

```text
outputs/pilot_demo/maps/
outputs/pilot_demo/laz_exports/
outputs/pilot_demo/geopackage/geo_jepa_predictions_grid.csv
outputs/pilot_demo/geopackage/geo_jepa_predictions.gpkg   # solo si geopandas esta instalado
```

## Outputs Forest-JEPA

```text
outputs/forest_jepa/forest_metrics.json
outputs/forest_jepa/forest_classification_metrics.csv
outputs/forest_jepa/forest_metrics_by_tile.csv
outputs/forest_jepa/forest_grid_metrics.csv
outputs/forest_jepa/forest_anomalies.csv
outputs/forest_jepa/forest_report.md
outputs/forest_jepa/maps/
```

## Interpretacion

Geo-JEPA queda validado de forma fuerte si `jepa_frozen_linear` o `jepa_frozen_mlp` son competitivos o superan al baseline en macro-F1/mIoU. Si solo mejora `jepa_full_finetune`, JEPA ayuda como inicializacion, pero no demuestra reutilizacion lineal clara.

Forest-JEPA MVP inicial requiere que `forest_report.md` cumpla al menos 8/10 checks del checklist. Si cumple 5-7, debe presentarse como Geo-JEPA con modulo forestal parcial.

## Limitaciones

- Canopy height es proxy, no medicion forestal certificada.
- Biomasa real no esta implementada.
- Clasificacion de especies no esta implementada.
- Cambio temporal real requiere pares multifecha y no se evalua todavia.
- Los LAZ coloreados de demo se exportan desde bloques procesados, no desde inferencia exhaustiva sobre toda la tesela raw.

## Proximos pasos comerciales

- Inferencia tiled completa sobre LAZ raw y merge por tesela.
- GeoPackage con capas separadas por modelo y metrica.
- Validacion forestal con parcelas/inventario o datos externos.
- Report HTML/PDF con mapas incrustados.
- Calibracion y estimacion de incertidumbre por punto/celda.
