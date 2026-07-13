# Rerun plan

## Gates obligatorios

1. Congelar el commit corregido y ejecutar la suite completa en `.venv`.
2. Inventariar/hashing de pares COL/CIR sin modificar `data/raw`.
3. Preparar bloques con `pnoa-block-v2-label-blind-eval`; train puede ser
   class-balanced, val/test deben ser uniformes y label-blind.
4. Guardar el manifest de split con tile IDs, seed, cobertura, conteos y SHA256.
   La pertenencia al test externo se decide por `label_blind_hash`.
5. Regenerar TW/geom y validar exactamente schema 56 o 73 por hash.
6. Regenerar features externas. Para DINO oficial se exige
   `used_real_dino=true`, `requested_backbone`, `actual_backbone` y
   `promotion_eligible=true`. `stat` requiere opt-in y queda como smoke.
7. Entrenar desde cero con seis logits; nunca reanudar checkpoints antiguos.
8. Ejecutar seeds `0, 1, 2` para baseline, DINO-concat y DINO-gated usando el
   mismo split congelado. Reportar media, desviación y cada seed individual.
9. Evaluar una sola vez el test final con
   `segmentation-metrics-v2-pred-ignore-is-fn`, incluyendo
   `predicted_ignore_count` y la matriz 6x7 cuando se evalúen modelos legacy.

## Comandos previstos

```powershell
.venv\Scripts\python.exe -m pytest -q -rs tests
.venv\Scripts\python.exe scripts/01_prepare_tiles.py --help
.venv\Scripts\python.exe scripts/14_build_dino_features.py --backend dinov2 --model dinov2_vits14 --force
```

Los comandos exactos de preparación dependen de fijar antes las rutas y tamaño
del experimento. No se inició un entrenamiento largo sobre un worktree aún sin
commit base. En este checkout, además, faltan bloques procesados y pesos DINO
locales; por ello no es lícito publicar métricas nuevas todavía.

## Presupuesto de cómputo

- GPU: una corrida a la vez para no mezclar OOM/throughput entre variantes.
- CPU: preparación de tiles paralela, conservando manifests deterministas.
- Seeds: pueden encolarse, pero no compartir directorios de salida ni cachés de
  schema distinto.
