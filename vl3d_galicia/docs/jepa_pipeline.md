# GeoPoint/TW LeJEPA Pipeline

## Diagnostico de datos

Auditoria ejecutada sobre `data/raw/pnoa_galicia`:

- 594 ficheros `.laz`.
- 297 pares `COL/CIR`; no falta ningun par.
- 50 pares tienen distinto numero de puntos entre `COL` y `CIR`; el preparador usa join espacial por XYZ para recuperar NIR.
- 1 fichero `COL` falla al descomprimir y se salta registrando error: `PNOA_2016_GAL_E_620-4696_ORT-CLA-COL.laz`.
- Puntos leidos correctamente: 1,015,887,315 de 1,021,091,123 declarados en cabecera.

Distribucion ASPRS leida en `COL`:

| Clase ASPRS | Clase | Puntos | % |
|---:|---|---:|---:|
| 2 | ground | 242,166,709 | 23.84 |
| 3 | low vegetation | 82,072,332 | 8.08 |
| 4 | medium vegetation | 32,702,017 | 3.22 |
| 5 | high vegetation | 195,077,079 | 19.20 |
| 6 | building | 16,690,952 | 1.64 |
| 9 | water | 63,123,852 | 6.21 |
| 12 | overlap/unreliable | 377,897,908 | 37.20 |

Conclusion: el entrenamiento debe ignorar la clase no fiable (`ignore_index=6` en el mapeo interno) y usar pesos de clase. `building` y `medium_vegetation` son minoritarias; `water` esta muy sesgada por campana.

## Pipeline

Instala dependencias en el Python que vayas a usar para preparar datos:

```powershell
python -m pip install -r requirements_jepa.txt
```

Si ese Python no tiene PyTorch CUDA, instala PyTorch CUDA antes de entrenar. En esta maquina el Python global detecta `NVIDIA GeForce RTX 5070 Ti Laptop GPU` con `torch 2.10.0+cu128`, pero no tiene `laspy`; la `.venv` tiene `laspy` pero PyTorch CPU.

Ejecuta auditoria completa:

```powershell
python scripts/00_audit_pnoa_pairs.py `
  --raw data/raw/pnoa_galicia `
  --reports reports `
  --manifest data/manifests/pnoa_galicia_tiles.csv
```

### Piloto rapido

Antes de lanzar el pipeline completo, ejecuta un piloto balanceado por campana y split. Usa 24 teselas, 4096 puntos por bloque y pocas epocas, guardando todo separado en `data/processed/galicia_blocks_pilot*` y `outputs/pilot`.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 `
  -Python python `
  -Quick `
  -Workers 8 `
  -PrepareWorkers 8 `
  -BatchSize 24 `
  -JepaBatchSize 48
```

El resultado rapido que debes mirar es `outputs/pilot/comparison/test_comparison.csv` y la tabla final por terminal. Si `tw_jepa` mejora `macro_iou` o `macro_f1` frente al baseline en este piloto, merece la pena lanzar el completo. Si empata o pierde por poco, no es concluyente: sube `-BaselineEpochs`, `-JepaEpochs`, `-FinetuneEpochs` o `-MaxTiles`.

Prepara bloques:

```powershell
python scripts/01_prepare_tiles.py `
  --raw data/raw/pnoa_galicia `
  --out data/processed/galicia_blocks_mixed `
  --reports reports `
  --tile-size 50 `
  --stride 50 `
  --points-per-block 8192 `
  --min-points 100 `
  --val-ratio 0.1 `
  --test-ratio 0.1 `
  --split-mode mixed `
  --num-workers 4
```

`mixed` mezcla teselas de `GAL-W-2015` y `GAL-E-2016/17` en train/val/test usando hash estable por tesela. Usa `--split-mode campaign` solo para medir generalizacion a una campana no vista.

Calcula TW:

```powershell
python scripts/02_compute_tw_features.py `
  --input data/processed/galicia_blocks_mixed `
  --out data/processed/galicia_blocks_mixed_tw `
  --reports reports `
  --k-neighbors 32 `
  --fit-stats-blocks 1000 `
  --sample-points-per-block 256 `
  --num-workers 8
```

Entrena baseline supervisado:

```powershell
python scripts/03_train_baseline.py `
  --data data/processed/galicia_blocks_mixed_tw `
  --out outputs/baseline `
  --epochs 40 `
  --batch-size 24 `
  --num-workers 8
```

Preentrena TW-LeJEPA autosupervisado:

```powershell
python scripts/04_pretrain_jepa.py `
  --data data/processed/galicia_blocks_mixed_tw `
  --out outputs/tw_jepa_pretrain `
  --epochs 80 `
  --batch-size 48 `
  --num-workers 8 `
  --tw-target `
  --sigreg-weight 0.1 `
  --sigreg-slices 256
```

Fine-tuning:

```powershell
python scripts/05_finetune_jepa.py `
  --data data/processed/galicia_blocks_mixed_tw `
  --checkpoint outputs/tw_jepa_pretrain/best_jepa.pt `
  --out outputs/tw_jepa_finetune `
  --epochs 60 `
  --batch-size 24 `
  --num-workers 8
```

Evaluacion:

```powershell
python scripts/06_evaluate.py `
  --data data/processed/galicia_blocks_mixed_tw `
  --split test `
  --checkpoint outputs/tw_jepa_finetune/best_model.pt `
  --out outputs/tw_jepa_finetune/test_metrics.json `
  --batch-size 24 `
  --num-workers 8
```

## Estimacion RTX 5070 Ti 12 GB / 32 GB RAM

- Auditoria raw: medida local, ~1 minuto.
- Preparacion de bloques completa: 45-120 min con 4 workers, muy dependiente de SSD.
- TW completo: 3-7 h con 8 workers. El coste es CPU y disco, no GPU.
- Baseline 40 epochs: 4-8 h en RTX 5070 Ti con batch 24.
- TW-LeJEPA 80 epochs: 8-16 h con batch 48.
- Fine-tuning 60 epochs: 6-12 h con batch 24.

Espacio recomendado antes de lanzar todo: al menos 250 GB libres. Los bloques base deberian rondar 55-70 GB; los bloques con TW pueden anadir 100-140 GB segun numero final de puntos/bloques.
