# Results invalidation

Fecha de corte: 2026-07-13. Estado normativo: `invalid_pre_fix` salvo que un
artefacto posterior demuestre todos los gates de `RERUN_PLAN.md`.

## Resultados invalidados

- Todas las tablas DINO/DINOv2 anteriores al schema `pnoa-spectral-v2`: el
  raster interpretaba las columnas PNOA en un orden distinto al escrito.
- Todas las métricas de segmentación anteriores a
  `segmentation-metrics-v2-pred-ignore-is-fn`: una predicción `ignore` sobre un
  target válido no contaba como falso negativo.
- Todos los checkpoints con siete logits. El índice 6 es target-only `ignore` y
  no puede ser una salida entrenable.
- Comparaciones construidas con selección class-aware en validación o test.
- Features externas sin `feature_schema_sha256` consistente entre manifest y
  payload, o con `input_feature_schema_sha256` distinto al bloque PNOA.
- Un artefacto `backend=stat` no es DINO. Solo puede figurar como
  `stat_raster_baseline`/smoke, nunca como candidato oficial.

Las cifras históricas se conservan como evidencia del trabajo previo, pero no
son resultados válidos bajo el protocolo corregido y no se deben combinar con
nuevas corridas.

## Inventario físico

No había cachés ni checkpoints que retirar físicamente en este checkout:

- `data/processed/galicia_blocks`: ausente.
- `outputs/`: ausente.
- `feature_config.json`, `*.pt`, `*.pth`, `*.ckpt` fuera de raw: ninguno.

Por tanto no se borró ningún archivo. Los raw se preservaron intactos: 594 LAZ
Galicia (17.159.019.393 bytes) y 763 LAZ multirregión (39.493.520.843 bytes).

## Criterio de rehabilitación

Una cifra solo cambia a `valid_post_fix` si el registro contiene
commit, comandos, seed, split hash label-blind, schema PNOA, schema externo,
backbone real cuando se declare DINO, checkpoint de seis logits y métricas v2.
