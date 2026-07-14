# Consolidación M0-M4 y Auditoría

## 1. Verificación 6-Logits (Provenance)

Se ha verificado la matriz de confusión (6x7 reales) y el flag `segmentation-metrics-v2-pred-ignore-is-fn` en todos los runs:

| Model Name | Provenance |
|---|---|
| M0_geometric_seed13 | `verified_six_logit` |
| M0_geometric_seed21 | `verified_six_logit` |
| M0_geometric_seed7 | `verified_six_logit` |
| M1_dinov2_concat_seed13 | `verified_six_logit` |
| M1_dinov2_concat_seed21 | `verified_six_logit` |
| M1_dinov2_concat_seed7 | `verified_six_logit` |
| M2_dinov3_concat_seed13 | `verified_six_logit` |
| M2_dinov3_concat_seed21 | `verified_six_logit` |
| M2_dinov3_concat_seed7 | `verified_six_logit` |
| M3_dinov3_gated_seed13 | `verified_six_logit` |
| M3_dinov3_gated_seed21 | `verified_six_logit` |
| M3_dinov3_gated_seed7 | `verified_six_logit` |
| M4_dinov3_multiview_seed13 | `verified_six_logit` |
| M4_dinov3_multiview_seed21 | `verified_six_logit` |
| M4_dinov3_multiview_seed7 | `verified_six_logit` |

## 2. Comparación Agregada por Familia (Emparejada por Seed contra M0)

| Family | mIoU Mean ± SD | mIoU Min-Max | Delta mIoU vs M0 (Mean ± SD) | OA Mean ± SD | Delta OA (Mean ± SD) |
|---|---|---|---|---|---|
| M0_geometric | 0.6183 ± 0.0117 | [0.6026, 0.6307] | +0.0000 ± 0.0000 | 0.8248 ± 0.0104 | +0.0000 ± 0.0000 |
| M1_dinov2_concat | 0.6024 ± 0.0287 | [0.5632, 0.6313] | -0.0160 ± 0.0170 | 0.8209 ± 0.0075 | -0.0039 ± 0.0046 |
| M2_dinov3_concat | 0.6180 ± 0.0040 | [0.6132, 0.6231] | -0.0003 ± 0.0134 | 0.8260 ± 0.0012 | +0.0013 ± 0.0101 |
| M3_dinov3_gated | 0.6115 ± 0.0156 | [0.5947, 0.6323] | -0.0068 ± 0.0065 | 0.8286 ± 0.0055 | +0.0038 ± 0.0092 |
| M4_dinov3_multiview | 0.6203 ± 0.0051 | [0.6152, 0.6274] | +0.0020 ± 0.0075 | 0.8268 ± 0.0046 | +0.0021 ± 0.0071 |

## 3. Tabla Única de Métricas Individuales (M0 - M4)

| Model | OA | mIoU | macro-F1 | Bal.Acc | Cov. | Ign.Rate | Params | Best Epoch | VRAM | Gate Stats |
|---|---|---|---|---|---|---|---|---|---|---|
| M0_geometric_seed13 | 0.8299 | 0.6217 | 0.7477 | 0.7653 | 1.0 | 0.000 | 167622 | 0 | unknown | nan |
| M0_geometric_seed21 | 0.8103 | 0.6026 | 0.7280 | 0.7651 | 1.0 | 0.000 | 167622 | 0 | unknown | nan |
| M0_geometric_seed7 | 0.8341 | 0.6307 | 0.7559 | 0.7511 | 1.0 | 0.000 | 167622 | 0 | unknown | nan |
| M1_dinov2_concat_seed13 | 0.8204 | 0.6125 | 0.7347 | 0.7451 | 1.0 | 0.000 | 175814 | 0 | unknown | nan |
| M1_dinov2_concat_seed21 | 0.8120 | 0.5632 | 0.6956 | 0.7535 | 1.0 | 0.000 | 175814 | 0 | unknown | nan |
| M1_dinov2_concat_seed7 | 0.8304 | 0.6313 | 0.7516 | 0.7697 | 1.0 | 0.000 | 175814 | 0 | unknown | nan |
| M2_dinov3_concat_seed13 | 0.8247 | 0.6231 | 0.7459 | 0.7702 | 1.0 | 0.000 | 175814 | 0 | unknown | nan |
| M2_dinov3_concat_seed21 | 0.8258 | 0.6179 | 0.7412 | 0.7704 | 1.0 | 0.000 | 175814 | 0 | unknown | nan |
| M2_dinov3_concat_seed7 | 0.8275 | 0.6132 | 0.7372 | 0.7507 | 1.0 | 0.000 | 175814 | 0 | unknown | nan |
| M3_dinov3_gated_seed13 | 0.8234 | 0.6075 | 0.7281 | 0.7607 | 1.0 | 0.000 | 242886 | 0 | unknown | Not recorded |
| M3_dinov3_gated_seed21 | 0.8262 | 0.5947 | 0.7218 | 0.7438 | 1.0 | 0.000 | 242886 | 0 | unknown | Not recorded |
| M3_dinov3_gated_seed7 | 0.8362 | 0.6323 | 0.7564 | 0.7547 | 1.0 | 0.000 | 242886 | 0 | unknown | Not recorded |
| M4_dinov3_multiview_seed13 | 0.8252 | 0.6185 | 0.7371 | 0.7559 | 1.0 | 0.000 | 175814 | 0 | unknown | nan |
| M4_dinov3_multiview_seed21 | 0.8222 | 0.6152 | 0.7425 | 0.7631 | 1.0 | 0.000 | 175814 | 0 | unknown | nan |
| M4_dinov3_multiview_seed7 | 0.8331 | 0.6274 | 0.7524 | 0.7466 | 1.0 | 0.000 | 175814 | 0 | unknown | nan |

## 4. Auditoría de la Purga de 76.038 Bloques

- **Comando Ejecutado**: `.\.venv\Scripts\python.exe scripts\purge_incompatible_blocks.py`
- **Fecha**: 14/07/2026 09:27 CET
- **Manifest Utilizado**: `protocols/block_compatibility_matrix.csv`
- **Tiles Afectados**: 51 tiles.
- **Split Original**: Sampling balanceado para uso en Train.
- **Split Nuevo**: Requerían sampling uniforme (movidos a validation/test). 1 tile fue excluido por caer en el buffer geográfico (blind-split).
- **Incompatibilidad**: Sampling balanceado de entrenamiento utilizado accidentalmente en tiles de validación, lo cual invalida las métricas uniformes.
- **Bytes Liberados**: Aprox 76.038 archivos `.pt` (varios GB).
- **Capacidad de Regeneración**: `01_prepare_tiles.py` está preparado para regenerar los bloques faltantes desde los archivos LAZ raw en la siguiente pasada.
- **Estado de M0-M4**: Los runs M0-M4 **sí** fueron entrenados antes de esta purga, usando la versión del dataset `galicia_blocks_geo_v1_d6b98dd` que incluía esos 51 tiles con sampling erróneo en el validation set. El hash del dataset utilizado fue `d6b98dd`. Se conserva este hash como registro histórico para los runs M0-M4. No se simulará que usaron el dataset corregido.


## 5. Conclusión Provisional Permitida

* **M1 DINOv2 concatenado**: `mixed/unstable`.
* **M2 DINOv3 concatenado**: `no robust improvement observed`.
* **M3 gated**: `pending three-seed aggregate` (Seed 7 fue favorable, pero la agregación muestra que las seeds 13 y 21 no sostienen la mejora).
* **M4 multiview**: `pending three-seed aggregate` (No mejora el baseline).

> **Veredicto general M0-M4**: Ninguna forma de incorporar DINO (concat, gated, o multiview top-view) mejora de manera consistente y robusta el baseline geométrico M0 en las tres seeds ni en las métricas por clase simultáneamente. Las varianzas entre seeds son mayores que las ganancias arquitectónicas en los casos favorables.
