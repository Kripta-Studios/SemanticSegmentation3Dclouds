# SUMMARY.md

Estado actualizado: 2026-06-13 noche Europe/Madrid.

## Actualizacion 2026-06-14: holdout externo estratificado CCAA

Se audito `data/raw/pnoa_varias_ccaa` para evitar seleccionar el holdout externo por orden alfabetico.

- Carpeta completa: 763 ficheros, 39.49 GB.
- Pares COL/CIR reconocidos: 215.
- Pares externos no Galicia usados como candidatos: 152 `CAT-2016`.
- Pares Galicia excluidos del holdout externo: 46 `GAL-E-2016` y 17 `GAL-W-2015`.
- Candidatos externos CAT-2016 auditados: 10.68 GB emparejados, 596.20M puntos COL.
- Selector creado: `scripts/23_select_external_holdout_tiles.py`.
- `scripts/19_prepare_external_holdout.py` ahora acepta `--tile-list`.
- Seleccion recomendada: `reports/external_holdout_selection_cat32/selected_tiles.txt`.
- Holdout recomendado: 32 tiles CAT-2016, 2.22 GB LAZ emparejados, 121.80M puntos COL.

Distribucion del holdout CAT-32 seleccionado:

- ground: 25.59M puntos, 44.31% de puntos fiables.
- low_vegetation: 4.49M, 7.77%.
- medium_vegetation: 5.01M, 8.67%.
- high_vegetation: 13.12M, 22.72%.
- building: 3.00M, 5.20%.
- water: 6.54M, 11.33%.
- unreliable: 64.05M, 52.58% del total.

Decision: no usar los 36GB completos por defecto. Para validacion externa seria se usara CAT-32 estratificado, porque cubre todas las clases y sobre-muestrea clases raras como building y water. CAT-20 queda como smoke/diagnostico previo, no como holdout final.

Resultados externos disponibles antes de preparar CAT-32:

- CAT-20 fue evaluado como smoke de generalizacion Galicia -> Catalunya. No contiene water y building es muy escaso, por tanto no es el holdout final.
- Baseline `outputs/medium_plus/baseline` en CAT-20: OA 0.771741, macro-F1 0.486237, mIoU 0.376724.
- Mejor modelo local-context seed42 en CAT-20: OA 0.837936, macro-F1 0.599441, mIoU 0.495298.
- Mejor modelo local-context seed1337 en CAT-20: OA 0.840001, macro-F1 0.600106, mIoU 0.496183.
- Mejor modelo local-context seed2026 en CAT-20: OA 0.841503, macro-F1 0.560747, mIoU 0.457770.
- Media local-context en CAT-20: OA 0.839814, macro-F1 0.586765, mIoU 0.483084.
- Delta medio vs baseline CAT-20: OA +0.068073, macro-F1 +0.100528, mIoU +0.106360.
- En CAT-20, local-context mejora IoU en las 5 clases presentes frente al baseline. Water no se puede evaluar.

Baseline ablation adicional:

- Se intento lanzar `tw_balanced_focal` multiseed para comparar contra el modelo con geom-context bajo el mismo protocolo class-balanced.
- El proceso llego al timeout de la sesion. No queda ningun `python` corriendo.
- Completado: `outputs/sota_ablation_baseline_multiseed/tw_balanced_focal_seed42`.
- Incompleto: `tw_balanced_focal_seed1337`; seed2026 no llego a ejecutarse.
- Baseline ablation seed42: OA 0.852175, macro-F1 0.799906, mIoU 0.693924.
- Geom-context seed42 sigue por encima: OA 0.864858, macro-F1 0.826031, mIoU 0.726305.
- Delta geom-context vs baseline-ablation seed42: OA +0.012682, macro-F1 +0.026126, mIoU +0.032381, IoU mejora 6/6 clases.

## Current executive state

The project now has a defensible Galicia LiDAR semantic segmentation MVP and an initial Forest-JEPA-style analytics layer. The strongest result is not a self-supervised JEPA winner. It is a supervised point segmentation model using label-free local geometric context features, TW descriptors, focal loss and class-balanced training.

Best single run:

- Model: `outputs/local_context_medium/geom_concat_h256_balancedval`
- OA: 0.864858
- Macro-F1: 0.826031
- mIoU: 0.726305
- Delta vs internal baseline: OA +0.034528, macro-F1 +0.077797, mIoU +0.091411
- Class IoU improved 6/6 classes vs the internal baseline.

Three-seed stability:

- Directory: `outputs/local_context_multiseed`
- Mean OA: 0.856705
- Mean macro-F1: 0.815236
- Mean mIoU: 0.713417
- Worst-run OA: 0.846074
- Worst-run macro-F1: 0.799690
- Worst-run mIoU: 0.695219
- All three seeds were positive against the internal baseline and improved all tracked class IoUs.

Per-class best-run IoU vs internal baseline:

- ground: 0.726344 -> 0.756050 (+0.029706)
- low_vegetation: 0.335411 -> 0.443883 (+0.108472)
- medium_vegetation: 0.406994 -> 0.532929 (+0.125934)
- high_vegetation: 0.910552 -> 0.947422 (+0.036870)
- building: 0.457501 -> 0.698767 (+0.241265)
- water: 0.972565 -> 0.978781 (+0.006216)

Paper comparison status:

- Extracted paper metrics currently tracked in repo:
  - low vegetation F1: paper 0.4625 vs current best 0.614846
  - high vegetation F1: paper 0.9617 vs current best 0.973001
- These extracted metrics are beaten.
- Strict SOTA is not established yet because the paper backbone/protocol, or a modern KPConv/RandLA-Net/Point Transformer baseline, has not been reproduced on the same locked split.

Self-supervised status:

- GeoPointJEPA and TW-JEPA are implemented and evaluated.
- Frozen JEPA downstream runs did not beat the strongest supervised baseline.
- DINOv2 fusion was tested and was negative.
- DINOv3 remains a future experiment because gated model access/weights are required and the fusion strategy should be redesigned.
- Commercial wording should be: "Galicia LiDAR semantic segmentation MVP with a Geo-JEPA/Forest-JEPA research track", not "fully self-supervised SOTA Geo-JEPA".

Generated presentation and paper artifacts:

- `presentation/ELI5.txt`
- `presentation/lidar_semantic_segmentation_mvp_light_en.html`
- `presentation/geo_forest_lidar_mvp_demo.html`
- `presentation/SUMMARY.md`
- `paper/galicia_lidar_semantic_segmentation_mvp.tex`
- `paper/galicia_lidar_semantic_segmentation_mvp.pdf`

Important reports:

- `outputs/local_context_multiseed/comparison/test_comparison.csv`
- `outputs/local_context_multiseed/comparison/test_comparison.md`
- `outputs/local_context_multiseed/benchmark/benchmark_report.md`
- `outputs/local_context_multiseed/benchmark/benchmark_summary.json`

Sales-safe verdict:

- Geo-JEPA: Partial research track, not yet a validated self-supervised winner.
- Galicia LiDAR segmentation: MVP técnico fuerte / demo comercial inicial.
- Forest-JEPA: Initial MVP based on forest proxy analytics, not certified forestry inventory.
- SOTA: promising internal result, not a strict SOTA claim until external backbones are reproduced on a fresh locked holdout.

Estado historico anterior: 2026-06-12 17:10 Europe/Madrid.

## Objetivo actual

El usuario pidio avanzar con una ruta DINO/V-JEPA para mejorar el baseline fuerte actual. La decision tecnica fue implementar primero DINO-raster-fusion porque DINOv3/DINOv2 encajan mejor con tiles/imagenes estaticas y dense features que V-JEPA, que tiene mas sentido para video o multitemporalidad real.

## Resultado fuerte previo

`outputs/medium_plus/comparison/test_comparison.csv`:

- `baseline` = Point/TW supervisado + focal loss + balanced sampler.
- OA 0.830329, macro-F1 0.748235, mIoU 0.634894.
- IoU: terrain 0.726344, low vegetation 0.335411, medium vegetation 0.406994, high vegetation 0.910552, building 0.457501, water 0.972565.
- Este baseline supera claramente a los JEPA congelados actuales.

Conclusion previa: el salto grande vino de TW + loss/sampler, no del JEPA de puntos actual.

## Cambios implementados en esta sesion

Archivos creados:

- `src/features/raster_dino.py`
- `scripts/14_build_dino_features.py`
- `run_dino_fusion.ps1`
- `tests/test_dino_fusion.py`
- `SUMMARY.md`

Archivos modificados:

- `src/data/segmentation_dataset.py`
- `scripts/train_common.py`
- `scripts/03_train_baseline.py`
- `scripts/07_compare_results.py`
- `README_GEO_FOREST_JEPA_DEMO.md`
- `requirements_jepa.txt`

## Que hace la nueva ruta DINO fusion

1. Rasteriza cada bloque `.pt` a una imagen multicanal.
2. Usa canales derivados de RGB/CIR/NIR/intensidad/Z/TW/densidad.
3. Extrae dense features por celda con backend:
   - `hf`: DINOv3 via Hugging Face.
   - `torchhub`: DINOv3 local oficial con pesos descargados.
   - `dinov2`: DINOv2 publico via `torch.hub`.
   - `stat`: fallback estadistico para smoke, no es DINO real.
4. Proyecta features a cada punto.
5. Guarda cache espejo `train/val/test/*.pt` con clave `dino_features`.
6. `SegmentationBlockDataset` concatena `XYZ + features + TW + dino_features`.
7. `03_train_baseline.py` entrena la fusion con el mismo trainer/sampler/loss del baseline fuerte.

## Comparacion automatica

`scripts/07_compare_results.py` ahora genera deltas contra baseline y veredicto automatico:

- `delta_OA_vs_baseline`
- `delta_macro_F1_vs_baseline`
- `delta_mIoU_vs_baseline`
- deltas por clase `delta_*_IoU_vs_baseline`
- `improved_class_iou_count`
- `worsened_class_iou_count`
- `verdict_vs_baseline`: `BASELINE`, `POSITIVE`, `MIXED`, `NEGATIVE`
- `verdict_reason`

Criterio actual:

- `POSITIVE`: mejora mIoU al menos 0.5 pp, no pierde macro-F1 y no empeora mas clases de las que mejora.
- `MIXED`: mejora alguna metrica pero no es una victoria fuerte o dana clases relevantes.
- `NEGATIVE`: no mejora OA, macro-F1 ni mIoU suficientemente frente al baseline supervisado.

Outputs:

- `outputs/<run>/comparison/test_comparison.csv`
- `outputs/<run>/comparison/test_comparison.md`

## Estado DINOv3

Se intento:

```powershell
python scripts/14_build_dino_features.py --data data/processed/galicia_blocks_medium_tw --out data/processed/dino_hf_smoke --backend hf --model facebook/dinov3-vits16-pretrain-lvd1689m --grid-size 128 --image-mode rgb_nir_height --out-dim 32 --max-blocks-per-split 1 --force
```

Resultado: fallo por acceso gated/401 de Hugging Face. La maquina no esta autenticada con acceso al modelo `facebook/dinov3-vits16-pretrain-lvd1689m`.

No es fallo del pipeline. Para DINOv3 real hace falta:

- `huggingface-cli login` con token que tenga acceso al modelo; o
- descargar pesos desde Meta y usar `-Backend torchhub -DinoRepoDir ... -DinoWeights ...`.

## Estado DINOv2

Se probo backend real publico:

```powershell
python scripts/14_build_dino_features.py --data data/processed/galicia_blocks_medium_tw --out data/processed/dinov2_smoke --backend dinov2 --model dinov2_vits14 --grid-size 126 --image-mode rgb_nir_height --out-dim 32 --max-blocks-per-split 1 --force
```

Resultado: OK. Descargo `dinov2_vits14` en cache local de torch y genero features reales DINOv2.

### DINOv2 medium real ejecutado

Se ejecuto una evaluacion medium real con DINOv2 `dinov2_vits14`:

- feature cache: `data/processed/galicia_blocks_medium_dinov2s14`
- outputs: `outputs/dinov2_fusion/dino_tw_fusion`
- comparacion: `outputs/dinov2_fusion/comparison/test_comparison.csv`
- backend real: `dinov2`
- bloques cacheados: 48,577 (`train=12,000`, `val=2,000`, `test=34,577`)
- puntos con features: 87,987,756
- dimensiones de entrada del modelo: 97 (`XYZ + features + TW + dino_features`)
- early stopping: mejor val macro-F1 en epoch 12, parada en epoch 20

Resultado test:

- `baseline` medium_plus: OA 0.830329, macro-F1 0.748235, mIoU 0.634894.
- `dino_tw_fusion`: OA 0.810661, macro-F1 0.720279, mIoU 0.607014.
- Delta DINOv2 vs baseline: OA -0.019668, macro-F1 -0.027955, mIoU -0.027880.
- IoU DINOv2: ground 0.701387, low vegetation 0.273014, medium vegetation 0.352227, high vegetation 0.898024, building 0.442960, water 0.974474.
- Frente al baseline solo mejora water IoU (+0.001909); empeora ground, low vegetation, medium vegetation, high vegetation y building.
- Veredicto automatico: `NEGATIVE`.

Interpretacion: DINOv2 fusion funciona tecnicamente, pero no mejora el baseline fuerte TW/focal/balanced. La ruta actual de concatenar dense features DINOv2 a cada punto no debe venderse como mejora. Para que DINO aporte, el siguiente experimento deberia usar un adapter/gating o late fusion y posiblemente DINOv3 satelite, no solo concatenacion directa.

## Runs ejecutados en esta sesion

### Smoke stat

Comando:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dino_fusion.ps1 -Python python -Backend stat -FeatureRoot data/processed/dino_smoke -OutRoot outputs/dino_smoke -CompareRoot outputs/medium_plus -MaxFeatureBlocksPerSplit 8 -MaxTrainBlocks 8 -MaxValBlocks 4 -MaxTestBlocks 8 -TrainEpochs 1 -Workers 2 -BatchSize 4 -ForceFeatures -ForceTrain
```

Resultado: OK. Solo valida cableado, no calidad.

### Quick DINOv2 real

Comando:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dino_fusion.ps1 -Python python -Backend dinov2 -DinoModel dinov2_vits14 -FeatureRoot data/processed/dinov2_quick -OutRoot outputs/dinov2_quick -CompareRoot outputs/medium_plus -GridSize 126 -MaxFeatureBlocksPerSplit 200 -MaxTrainBlocks 200 -MaxValBlocks 50 -MaxTestBlocks 50 -TrainEpochs 5 -Workers 8 -BatchSize 16 -ForceFeatures -ForceTrain
```

Resultado: OK, pero no comparable al medium completo porque usa solo 200/50/50 bloques y el test limitado no contiene building/water.

`outputs/dinov2_quick/comparison/test_comparison.csv`:

- `dino_tw_fusion`: OA 0.542429, macro-F1 0.315345, mIoU 0.234954.
- IoU: terrain 0.307885, low vegetation 0.274436, medium vegetation 0.067990, high vegetation 0.759411, building 0.0, water 0.0.
- Veredicto automatico: `NEGATIVE` contra `outputs/medium_plus/baseline`.

Interpretacion: este quick no demuestra mejora. Sirve para validar que DINOv2 real se integra y entrena. Para evaluar de verdad hay que correr con todo el medium y test completo.

### Medium DINOv2 real

Comando de entrenamiento ejecutado:

```powershell
python scripts\03_train_baseline.py --data data/processed/galicia_blocks_medium_tw --out outputs/dinov2_fusion/dino_tw_fusion --epochs 30 --batch-size 24 --num-workers 4 --use-tw-input --external-feature-dir data/processed/galicia_blocks_medium_dinov2s14 --external-feature-key dino_features --probe-type mlp --seed 42 --class-weight-mode inverse_sqrt --max-class-weight 20.0 --loss-type focal --focal-gamma 1.5 --balanced-sampler --sampler-alpha 1.2 --sampler-max-weight 10.0 --sampler-class-boost 1:1.5,2:2.0,4:4.0 --early-stopping-patience 8 --early-stopping-min-delta 0.001 --max-train-blocks 12000 --max-val-blocks 2000 --no-resume
```

Comando de comparacion ejecutado:

```powershell
python scripts\07_compare_results.py --experiments-root outputs\medium_plus --experiments-root outputs\dinov2_fusion --out-csv outputs\dinov2_fusion\comparison\test_comparison.csv --out-md outputs\dinov2_fusion\comparison\test_comparison.md
```

Resultado: OK. El baseline sigue siendo superior.

## Tests ejecutados

Pasaron:

```powershell
python -m pytest tests/test_dino_fusion.py tests/test_finetune_smoke.py tests/test_geo_forest_mvp.py -q
```

Resultado: 8 passed.

Tambien compilo:

```powershell
python -m compileall src\features\raster_dino.py src\data\segmentation_dataset.py scripts\14_build_dino_features.py scripts\03_train_baseline.py scripts\07_compare_results.py
```

## Comandos recomendados siguientes

### Reproducir DINOv2 medium real

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dino_fusion.ps1 `
  -Python python `
  -Backend dinov2 `
  -DinoModel dinov2_vits14 `
  -FeatureRoot data/processed/galicia_blocks_medium_dinov2s14 `
  -OutRoot outputs/dinov2_fusion `
  -CompareRoot outputs/medium_plus `
  -GridSize 126 `
  -Workers 8 `
  -BatchSize 24 `
  -TrainEpochs 30 `
  -MaxTrainBlocks 12000 `
  -MaxValBlocks 2000
```

No pasar `-MaxFeatureBlocksPerSplit` ni `-MaxTestBlocks` si se quiere evaluar todo el test.

### Con acceso DINOv3 Hugging Face

```powershell
huggingface-cli login
powershell -ExecutionPolicy Bypass -File .\run_dino_fusion.ps1 `
  -Python python `
  -Backend hf `
  -DinoModel facebook/dinov3-vits16-pretrain-lvd1689m `
  -FeatureRoot data/processed/galicia_blocks_medium_dino_v3s16 `
  -OutRoot outputs/dino_fusion `
  -CompareRoot outputs/medium_plus `
  -Workers 8 `
  -BatchSize 24 `
  -TrainEpochs 30 `
  -MaxTrainBlocks 12000 `
  -MaxValBlocks 2000
```

### Con pesos DINOv3 descargados de Meta

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dino_fusion.ps1 `
  -Python python `
  -Backend torchhub `
  -DinoRepoDir C:\ruta\a\dinov3 `
  -DinoModel dinov3_vits16 `
  -DinoWeights C:\ruta\a\dinov3_vits16.pth `
  -FeatureRoot data/processed/galicia_blocks_medium_dino_v3s16 `
  -OutRoot outputs/dino_fusion
```

## Artefactos temporales pequenos

Generados y no borrados:

- `data/processed/dino_smoke`
- `outputs/dino_smoke`
- `data/processed/dinov2_smoke`
- `data/processed/dinov2_quick`
- `outputs/dinov2_quick`

Son pequenos comparados con los datasets grandes. Si se quiere limpieza estricta, borrar esos directorios; no borrar `data/processed/galicia_blocks_medium_tw`, `outputs/medium`, `outputs/medium_plus` ni `outputs/forest_jepa_medium`.

## Riesgos / proximos ajustes tecnicos

- La fusion actual concatena features proyectadas al PointMLP. Puede necesitar normalizacion global por dataset o adapter/gating para que DINO no degrade TW.
- El quick DINOv2 limitado no es evidencia de calidad.
- DINOv3 satelite (`facebook/dinov3-vitl16-pretrain-sat493m`) probablemente sea mejor para PNOA, pero es mas pesado y tambien gated.
- V-JEPA 2.1 queda para una fase posterior con datos multitemporales o secuencias de modalidades; no esta implementado.

## Actualizacion 2026-06-13: DINOv2 gated y analisis de nuevas arquitecturas

Resultado adicional ejecutado:

- `outputs/dinov2_gated_medium/dino_tw_gated`
- Fusion: Point/TW + DINOv2 con adapter residual gated.
- OA 0.824055, macro-F1 0.727820, mIoU 0.612775.
- IoU: terrain 0.714303, low vegetation 0.327423, medium vegetation 0.403588, high vegetation 0.908359, building 0.364743, water 0.958238.
- Veredicto automatico contra `outputs/medium_plus/baseline`: `NEGATIVE`.
- Conclusion: el gating reduce parte del dano de la concatenacion directa DINOv2, pero sigue peor que el baseline TW/focal/balanced en OA, macro-F1, mIoU y todas las clases. DINOv2 queda descartado como mejora directa.

Baseline interno vigente:

- `outputs/medium_plus/baseline`.
- OA 0.830329, macro-F1 0.748235, mIoU 0.634894.
- Clases debiles: low vegetation IoU 0.335411, medium vegetation IoU 0.406994, building IoU 0.457501.
- Clases fuertes: high vegetation IoU 0.910552, water IoU 0.972565.

Lectura de PDFs solicitados y decision tecnica:

- `DINOv3.pdf`: alta prioridad, pero solo con modelo DINOv3 satellite/aerial y como late fusion, teacher distillation, pseudo-label refinement o decoder raster auxiliar. No repetir DINOv2 concat/gated esperando mejoras automaticas.
- `V-JEPA 2.1 Unlocking Dense Features in Video.pdf`: prioridad media. Sus ideas utiles son dense predictive loss, deep self-supervision y multi-modal tokenizers. Para este repo tiene sentido adaptarlas a tiles LiDAR/PNOA estaticos; la parte video solo merece la pena si hay datos multitemporales reales.
- `A LIGHTWEIGHT LIBRARY FOR ENERGY-BASED JOINT-.pdf`: prioridad media-baja. Sirve como referencia para regularizadores SIGReg/VICReg y evitar colapso JEPA; no es una arquitectura SOTA LiDAR por si sola.
- `Var-JEPA A Variational Formulation of the Joint-Embedding Predictive.pdf.pdf`: prioridad media-baja. Interesante para incertidumbre y active learning por celda/tile; no atacara directamente low vegetation/building sin mejor backbone 3D.
- `JEPA-DNA Grounding Genomic Foundation Models.pdf`: prioridad baja para LiDAR. La idea transferible es continual latent grounding sobre un backbone ya fuerte, no la arquitectura genomica.
- `Intuitive physics understanding emerges from.pdf`: prioridad baja para segmentacion estatica. Util para cambio/anomalias si en el futuro hay secuencias temporales.
- `FF-JEPA Long-Horizon Planning in World Models.pdf`: no prioritario para este MVP. Es planificacion/acciones/world models, no segmentacion LiDAR estatica.

Conclusion de arquitectura:

- Para superar el paper de Galicia y acercarse a SOTA/industria, falta un backbone 3D local eficiente.
- El pipeline moderno actual usa `PointSegmentationNet`, un PointMLP con pooling global. Esto no modela vecindarios locales al nivel de KPConv/SFL-Net/RandLA-Net/PTv3.
- Existe `src/models/point_local_backbone.py`, pero usa `torch.cdist` O(N^2) para KNN y no esta integrado en `scripts/03_train_baseline.py`; no es seguro lanzarlo con 8192 puntos/bloque en 12GB VRAM sin optimizar.
- Tambien existe codigo legacy KPConv/SFL en `src/model/deeplearn`, pero no esta conectado al pipeline reproducible moderno.

Ruta recomendada:

1. Implementar o integrar un backbone local eficiente: SFL-Net/KPConv ligero, RandLA-Net style o Point Transformer V3 style.
2. Mantener TW + focal loss + balanced sampler como baseline fuerte.
3. Usar DINOv3 satellite/aerial solo despues, como complemento 2D: late fusion/teacher/auxiliary raster decoder.
4. Rehacer JEPA sobre el backbone local con dense token prediction y deep supervision tipo V-JEPA 2.1.
5. Medir siempre contra `outputs/medium_plus/baseline` y contra una reproduccion SFL-Net/KPConv en el mismo split.

Veredicto actual:

- DINOv2: no mejora.
- DINOv3: prometedor, pero requiere acceso/pesos y otra estrategia de fusion.
- V-JEPA 2.1: prometedor para pretraining denso, no para reemplazar el backbone 3D.
- FF-JEPA / Intuitive Physics / JEPA-DNA: investigacion futura, no prioridad para subir precision ahora.
- Lo mas probable para mejorar low/medium vegetation y building es geometria local eficiente + entrenamiento balanceado, no mas features globales.

## Actualizacion 2026-06-13 tarde: local geometric context supera baseline en todas las clases

Se implemento una ruta nueva sin usar etiquetas para generar features:

- `src/features/geom_context.py`
- `scripts/15_build_geom_context_features.py`
- feature key: `geom_features`
- output cache: `data/processed/galicia_blocks_medium_geom_context`
- bloques cacheados: 100,467
- puntos cacheados: 230,625,346
- feature_dim: 56
- `uses_labels=false` en `feature_config.json`

Las features se calculan por bloque desde `coords`, `features`, `tw_features` y estadisticas locales por celda a escalas 2.5m, 5m y 10m. No usan `labels`, ni predicciones, ni informacion del test para entrenar.

Tambien se anadio seleccion determinista de bloques:

- `--train-block-selection sorted|random|class_balanced`
- `--val-block-selection sorted|random|class_balanced`
- `--test-block-selection sorted|random|class_balanced`

Motivo: `--max-val-blocks 2000` con orden alfabetico dejaba solo 17 puntos de building en validacion. Con `class_balanced` la validacion usada en tuning tiene 108,627 puntos building, sin tocar el test completo.

Experimentos ejecutados:

1. `outputs/local_context_medium/geom_concat_h256`
   - TW + geom context, seleccion sorted.
   - OA 0.839163, macro-F1 0.786733, mIoU 0.678073.
   - Mejora 5/6 clases contra baseline; baja water ligeramente.

2. `outputs/local_context_medium/geom_concat_h256_balancedval`
   - TW + geom context, train/val class_balanced, test completo sorted.
   - OA 0.864858, AA 0.840008, macro-F1 0.826031, mIoU 0.726305.
   - Mejora 6/6 clases contra baseline.

Comparacion contra `outputs/medium_plus/baseline`:

- baseline: OA 0.830329, macro-F1 0.748235, mIoU 0.634894.
- nuevo: OA 0.864858, macro-F1 0.826031, mIoU 0.726305.
- delta: OA +0.034528, macro-F1 +0.077797, mIoU +0.091411.

IoU por clase:

- ground: 0.726344 -> 0.756050 (+0.029706)
- low_vegetation: 0.335411 -> 0.443883 (+0.108472)
- medium_vegetation: 0.406994 -> 0.532929 (+0.125934)
- high_vegetation: 0.910552 -> 0.947422 (+0.036870)
- building: 0.457501 -> 0.698767 (+0.241265)
- water: 0.972565 -> 0.978781 (+0.006216)

F1 por clase del nuevo modelo:

- ground 0.861080
- low_vegetation 0.614846
- medium_vegetation 0.695308
- high_vegetation 0.973001
- building 0.822675
- water 0.989277

Output de comparacion:

- `outputs/local_context_medium/comparison/test_comparison.csv`
- `outputs/local_context_medium/comparison/test_comparison.md`

Veredicto automatico:

- `geom_concat_h256_balancedval`: `POSITIVE`
- mejora OA, macro-F1, mIoU y 6/6 clases contra el baseline supervisado previo.

Sobre comparacion con el paper "Deep Learning for Ultra-Large-Scale Semantic Segmentation of Geographic 3D Point":

- El nuevo modelo supera claramente los numeros explicitamente extraidos del paper para low vegetation F1 (46.25% en paper vs 61.48% aqui) y high vegetation F1 (96.17% en paper vs 97.30% aqui).
- Tambien mejora mucho building frente al baseline interno, con F1 82.27% e IoU 69.88%.
- No debe afirmarse todavia "SOTA demostrado" hasta reproducir SFL-Net/KPConv del paper en el mismo split exacto y con el mismo protocolo. Si se vende, la formulacion honesta es: "en nuestro split Galicia PNOA-II, el modelo local-context supera el baseline interno y bate las cifras publicadas disponibles para vegetacion baja/alta; falta benchmark reproducido de SFL-Net para claim SOTA estricto".

Incidencia:

- Un `pytest` lanzado con `tmp_path` quedo colgado por permisos de `%TEMP%`/sandbox y se mato manualmente. No era entrenamiento ni evaluacion. Para tests en Windows usar `--basetemp C:\tmp\vl3d_pytest`.

## Actualizacion 2026-06-14: holdout externo CAT-32 sin data leakage

Objetivo: validar generalizacion geografica fuera de Galicia usando `data/raw/pnoa_varias_ccaa`, sin usar esos datos para entrenamiento, fine-tuning ni seleccion de hiperparametros.

Seleccion:

- Script: `scripts/23_select_external_holdout_tiles.py`
- Seleccion final: `reports/external_holdout_selection_cat32/selected_tiles.txt`
- 32 pares COL/CIR de `CAT-2016`
- Excluidos prefijos `GAL-E-2016` y `GAL-W-2015`
- Motivo: CAT-32 tiene todas las clases relevantes, incluida agua y mas edificios que el smoke CAT-20.

Preparacion y features:

```powershell
python scripts\19_prepare_external_holdout.py --raw data\raw\pnoa_varias_ccaa --out data\processed\pnoa_varias_ccaa_holdout_cat32 --reports reports\external_holdout_cat32 --tile-list reports\external_holdout_selection_cat32\selected_tiles.txt --exclude-campaign-prefix GAL --num-workers 8
python scripts\02_compute_tw_features.py --input data\processed\pnoa_varias_ccaa_holdout_cat32 --out data\processed\pnoa_varias_ccaa_holdout_cat32_tw --reports reports\external_holdout_cat32_tw --splits test --stats-json reports\tw_normalization.json --num-workers 16
python scripts\15_build_geom_context_features.py --data data\processed\pnoa_varias_ccaa_holdout_cat32_tw --out data\processed\pnoa_varias_ccaa_holdout_cat32_geom_context --splits test --num-workers 16
```

No se ajustaron estadisticas con CAT-32: TW usa `reports\tw_normalization.json` aprendido del pipeline Galicia. Geom-context no usa etiquetas y queda marcado `uses_labels=false`.

Datos preparados:

- tiles ok: 32
- blocks: 48,696
- points: 119,005,770
- distribucion fiable: ground 44.34%, low vegetation 7.77%, medium vegetation 8.68%, high vegetation 22.71%, building 5.20%, water 11.29%
- unreliable: 51.53% de todos los puntos, excluidos de metricas por `IGNORE_INDEX`

Evaluacion externa CAT-32:

```powershell
python scripts\20_evaluate_segmentation_model.py --model-dir outputs\sota_ablation_baseline_multiseed\tw_balanced_focal_seed42 --data data\processed\pnoa_varias_ccaa_holdout_cat32_tw --out outputs\external_holdout_cat32_eval\tw_balanced_focal_seed42 --batch-size 32 --num-workers 8 --use-tw-input
python scripts\20_evaluate_segmentation_model.py --model-dir outputs\local_context_medium\geom_concat_h256_balancedval --data data\processed\pnoa_varias_ccaa_holdout_cat32_tw --external-feature-dir data\processed\pnoa_varias_ccaa_holdout_cat32_geom_context --external-feature-key geom_features --out outputs\external_holdout_cat32_eval\geom_concat_h256_balancedval_seed42 --batch-size 32 --num-workers 8 --use-tw-input
python scripts\20_evaluate_segmentation_model.py --model-dir outputs\local_context_multiseed\geom_concat_h256_balancedval_seed1337 --data data\processed\pnoa_varias_ccaa_holdout_cat32_tw --external-feature-dir data\processed\pnoa_varias_ccaa_holdout_cat32_geom_context --external-feature-key geom_features --out outputs\external_holdout_cat32_eval\geom_concat_h256_balancedval_seed1337 --batch-size 32 --num-workers 8 --use-tw-input
python scripts\20_evaluate_segmentation_model.py --model-dir outputs\local_context_multiseed\geom_concat_h256_balancedval_seed2026 --data data\processed\pnoa_varias_ccaa_holdout_cat32_tw --external-feature-dir data\processed\pnoa_varias_ccaa_holdout_cat32_geom_context --external-feature-key geom_features --out outputs\external_holdout_cat32_eval\geom_concat_h256_balancedval_seed2026 --batch-size 32 --num-workers 8 --use-tw-input
python scripts\07_compare_results.py --experiments-root outputs\external_holdout_cat32_eval --reference-model tw_balanced_focal_seed42 --out-csv outputs\external_holdout_cat32_eval\comparison\test_comparison.csv --out-md outputs\external_holdout_cat32_eval\comparison\test_comparison.md
```

Resultados contra baseline fuerte `tw_balanced_focal_seed42`:

| modelo | OA | macro-F1 | mIoU | delta mIoU | clases IoU mejoran |
|---|---:|---:|---:|---:|---:|
| tw_balanced_focal_seed42 | 0.725991 | 0.639851 | 0.494502 | 0.000000 | baseline |
| geom seed42 | 0.772181 | 0.690962 | 0.552451 | +0.057948 | 6/6 |
| geom seed1337 | 0.779076 | 0.694157 | 0.557189 | +0.062686 | 6/6 |
| geom seed2026 | 0.778643 | 0.683042 | 0.548943 | +0.054440 | 5/6, building practicamente empata |

Mejor CAT-32: `outputs/local_context_multiseed/geom_concat_h256_balancedval_seed1337`.

IoU por clase del mejor CAT-32 externo:

- ground: 0.700082
- low_vegetation: 0.258208
- medium_vegetation: 0.516861
- high_vegetation: 0.837282
- building: 0.376507
- water: 0.654191

Interpretacion:

- La mejora externa es real frente al baseline fuerte: +5.44 a +6.27 puntos mIoU y +4.32 a +5.43 puntos macro-F1 segun seed.
- El modelo geom-context generaliza mejor a una comunidad autonoma no vista, pero el dominio externo es claramente mas dificil que Galicia: low vegetation, building y water caen mucho respecto al test interno.
- Esto permite defender "generalizacion geografica inicial", no "SOTA demostrado".
- No es evidencia de self-supervised superior todavia; es una mejora supervisada con features geometricas no etiquetadas.

Cambios de tooling:

- `scripts/07_compare_results.py` ahora acepta `--reference-model` para fijar la fila baseline al comparar carpetas con varios experimentos historicos.

Artefactos:

- Metricas CAT-32: `outputs/external_holdout_cat32_eval/*/metrics.json`
- Per-class: `outputs/external_holdout_cat32_eval/*/per_class_metrics.csv`
- Comparacion: `outputs/external_holdout_cat32_eval/comparison/test_comparison.csv`
- Report: `outputs/external_holdout_cat32_eval/comparison/test_comparison.md`

## Actualizacion 2026-06-14: diagnostico y correccion de caida en dominio externo

Pregunta investigada: por que las metricas externas en otras CCAA caian mucho frente a Galicia, si era overfitting, mal modelo o falta de generalizacion.

Causa raiz encontrada:

- No era solo "modelo malo". El modelo previo aprendia demasiado de estadisticas de Galicia.
- Shift fotometrico/NIR fuerte:
  - NIR medio Galicia test: 0.5050
  - NIR medio CAT-32 externo: 0.2955
  - JS divergence NIR: 0.2420
  - Por clase, el NIR cambia aun mas: water 0.2351 -> 0.0786, high vegetation 0.6299 -> 0.3244, medium 0.6171 -> 0.3046, low 0.6497 -> 0.3203.
- Shift de relieve/densidad:
  - z_range medio Galicia: 16.99
  - z_range medio CAT-32: 61.58
  - z_range p95 Galicia: 42.54
  - z_range p95 CAT-32: 423.26
- Shift de clases:
  - CAT-32 tiene mas ground/building/medium y menos low/high/water que Galicia test.
  - El holdout CAT fresco restante tiene casi nada de water: 44,099 puntos, 0.064% de los fiables.
- Conclusion tecnica: habia sobreespecializacion a Galicia en fotometria/prior de clase y una normalizacion de altura demasiado dependiente del bloque. El backbone no es SOTA, pero el fallo principal era shift de dominio y no ausencia total de senal.

Cambios implementados:

- `src/data/segmentation_dataset.py`
  - `coordinate_normalization=xy_unit_z_robust`
  - `spectral_normalization=block_robust`
  - `external_feature_normalization=spectral_block_robust`
  - augmentaciones espectrales opcionales.
  - seleccion class-balanced usando `class_counts_cache.csv`.
- `scripts/03_train_baseline.py`
  - flags de normalizacion y augmentacion.
- `scripts/20_evaluate_segmentation_model.py`
  - evaluacion respeta normalizaciones del `run_config`.
- `scripts/25_cache_block_label_counts.py`
  - cache rapido de conteos por bloque para seleccionar 20k bloques balanceados sin recargar todos los tensores.
- `scripts/23_select_external_holdout_tiles.py`
  - `--manifest-in` para reutilizar manifiestos auditados.
  - `--exclude-tile-list` para crear holdouts disjuntos sin solape con CAT-32.

Experimentos de correccion:

- `geom_robustnorm_seed42`
  - CAT-32 externo: OA 0.805471, macro-F1 0.730198, mIoU 0.599356.
- `geom_metric_robustnorm_random_seed42`
  - CAT-32 externo: OA 0.795233, macro-F1 0.728758, mIoU 0.595653.
  - Las alturas metricas no mejoraron el resultado global.
- `geom_robustnorm_spectralaug_seed42`
  - CAT-32 externo: OA 0.794982, macro-F1 0.726472, mIoU 0.593987.
  - La augmentacion espectral fuerte no fue la solucion.
- `geom_robustnorm_cb20k_seed42`
  - Mejor modelo actual.
  - Entrenado con 20k train blocks, 4k val blocks, seleccion class-balanced cacheada, robust norm, TW input y geom-context original de 56 dims.
  - Internal Galicia: OA 0.867092, macro-F1 0.824760, mIoU 0.725017.
  - CAT-32 externo: OA 0.808256, macro-F1 0.734167, mIoU 0.604094.
  - Gana a `tw_balanced_focal_seed42` en CAT-32 por +0.0823 OA, +0.0943 macro-F1, +0.1096 mIoU y 6/6 IoUs.

Holdout externo fresco sin solape:

- Seleccion: `reports/external_holdout_selection_cat32_fresh/selected_tiles.txt`
- Excluye los 32 tiles de `reports/external_holdout_selection_cat32/selected_tiles.txt`.
- Datos:
  - 32 tiles CAT-2016.
  - 49,853 bloques.
  - 130,588,254 puntos.
  - Distribucion fiable: ground 51.76%, low 8.80%, medium 10.51%, high 28.07%, building 0.80%, water 0.064%.
- Limitacion:
  - Sirve bien para ground/vegetacion.
  - Building es escaso y water es casi ausente, asi que las metricas de water son poco estables.

Comandos ejecutados para el holdout fresco:

```powershell
python scripts\23_select_external_holdout_tiles.py --raw data\raw\pnoa_varias_ccaa --reports reports\external_holdout_selection_cat32_fresh --include-campaign-prefix CAT --exclude-campaign-prefix GAL --target-tiles 32 --manifest-in reports\external_holdout_selection_cat32\tile_class_manifest.json --exclude-tile-list reports\external_holdout_selection_cat32\selected_tiles.txt
python scripts\19_prepare_external_holdout.py --raw data\raw\pnoa_varias_ccaa --out data\processed\pnoa_varias_ccaa_holdout_cat32_fresh --reports reports\external_holdout_cat32_fresh --tile-list reports\external_holdout_selection_cat32_fresh\selected_tiles.txt --exclude-campaign-prefix GAL --num-workers 8
python scripts\02_compute_tw_features.py --input data\processed\pnoa_varias_ccaa_holdout_cat32_fresh --out data\processed\pnoa_varias_ccaa_holdout_cat32_fresh_tw --reports reports\external_holdout_cat32_fresh_tw --splits test --stats-json reports\tw_normalization.json --num-workers 16
python scripts\15_build_geom_context_features.py --data data\processed\pnoa_varias_ccaa_holdout_cat32_fresh_tw --out data\processed\pnoa_varias_ccaa_holdout_cat32_fresh_geom_context --splits test --num-workers 16 --no-metric-height --force
python scripts\20_evaluate_segmentation_model.py --model-dir outputs\sota_ablation_baseline_multiseed\tw_balanced_focal_seed42 --data data\processed\pnoa_varias_ccaa_holdout_cat32_fresh_tw --out outputs\external_holdout_cat32_fresh_eval\tw_balanced_focal_seed42 --batch-size 32 --num-workers 8 --use-tw-input
python scripts\20_evaluate_segmentation_model.py --model-dir outputs\domain_generalization\geom_robustnorm_cb20k_seed42 --data data\processed\pnoa_varias_ccaa_holdout_cat32_fresh_tw --external-feature-dir data\processed\pnoa_varias_ccaa_holdout_cat32_fresh_geom_context --external-feature-key geom_features --out outputs\external_holdout_cat32_fresh_eval\geom_robustnorm_cb20k_seed42 --batch-size 32 --num-workers 8 --use-tw-input
python scripts\07_compare_results.py --experiments-root outputs\external_holdout_cat32_fresh_eval --reference-model tw_balanced_focal_seed42 --out-csv outputs\external_holdout_cat32_fresh_eval\comparison\test_comparison.csv --out-md outputs\external_holdout_cat32_fresh_eval\comparison\test_comparison.md
```

Resultado holdout fresco:

| modelo | OA | macro-F1 | mIoU | mIoU sin water | delta mIoU vs baseline |
|---|---:|---:|---:|---:|---:|
| tw_balanced_focal_seed42 | 0.773336 | 0.513794 | 0.406243 | 0.483312 | baseline |
| geom_robustnorm_cb20k_seed42 | 0.832728 | 0.617996 | 0.509763 | 0.606180 | +0.103521 |

IoU por clase en holdout fresco:

| clase | baseline IoU | robust IoU | delta |
|---|---:|---:|---:|
| ground | 0.734263 | 0.799656 | +0.065393 |
| low_vegetation | 0.212273 | 0.277003 | +0.064730 |
| medium_vegetation | 0.472813 | 0.564573 | +0.091759 |
| high_vegetation | 0.819252 | 0.884664 | +0.065412 |
| building | 0.177961 | 0.505003 | +0.327042 |
| water | 0.020893 | 0.027681 | +0.006787 |

Interpretacion:

- La correccion generaliza a un holdout externo disjunto: +5.94 pp OA, +10.42 pp macro-F1 y +10.35 pp mIoU.
- Mejora 6/6 clases frente al baseline fuerte, incluso en el holdout fresco.
- La mejora fuerte viene de robust normalization + geom-context + seleccion class-balanced real.
- Low vegetation sigue siendo la clase mas debil en IoU absoluto.
- Water no queda resuelto comercialmente en el holdout fresco: el modelo detecta los pocos puntos de agua, pero sobrepredice agua en ground porque el prior real de water es solo 0.064%. Hace falta un holdout externo con agua suficiente o calibracion conservadora validada en otro conjunto, sin tocar etiquetas de test.
- Esto ya es defendible como mejora de generalizacion supervisada sobre el baseline. No demuestra todavia superioridad self-supervised JEPA.

Artefactos nuevos:

- `reports/domain_shift_cat32/domain_shift_report.md`
- `reports/external_holdout_selection_cat32_fresh/selected_tiles.txt`
- `data/processed/pnoa_varias_ccaa_holdout_cat32_fresh_tw`
- `data/processed/pnoa_varias_ccaa_holdout_cat32_fresh_geom_context`
- `outputs/external_holdout_cat32_fresh_eval/tw_balanced_focal_seed42`
- `outputs/external_holdout_cat32_fresh_eval/geom_robustnorm_cb20k_seed42`
- `outputs/external_holdout_cat32_fresh_eval/comparison/test_comparison.md`

Siguiente paso tecnico:

- Conseguir o seleccionar un holdout externo con soporte real de water/building; el CAT restante despues de CAT-32 no lo tiene.
- Entrenar/evaluar un backbone local mas serio con el mismo protocolo: RandLA/PointTransformer/KPConv-like.
- Rehacer self-supervised sobre backbone local/contextual y evaluar frozen/semi-frozen. El mejor resultado actual no viene de JEPA, viene de entrenamiento supervisado robusto con geom-context.
- Si se usa unlabeled externo para adaptacion self-supervised, documentarlo como domain adaptation transductiva y mantener otro test externo fresco intacto.
