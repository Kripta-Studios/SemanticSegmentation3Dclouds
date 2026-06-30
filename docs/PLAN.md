# PLAN.md — GeoPoint-JEPA para mejorar la segmentación semántica ultra-large-scale de nubes LiDAR/ALS con etiquetas ausentes

## 0. Resumen ejecutivo

Objetivo: implementar, evaluar y documentar una extensión tipo **JEPA** para el proyecto del artículo **"Deep Learning for Ultra-Large-Scale Semantic Segmentation of Geographic 3D Point Clouds With Missing Labels"**. La mejora propuesta se llama provisionalmente:

> **GeoPoint-JEPA / TW-JEPA: Self-supervised latent predictive pretraining for ultra-large-scale ALS point cloud segmentation with missing labels**

La idea no es sustituir SFL-Net, KPConv o el pipeline original, sino añadir una fase de **preentrenamiento autosupervisado** sobre todos los puntos, incluidos los no etiquetados, y después hacer fine-tuning supervisado con la misma pérdida ponderada usada en el paper original.

La hipótesis principal es:

> Si se preentrena un encoder 3D con una tarea JEPA de predicción latente de regiones ocultas, usando todos los puntos ALS disponibles, el modelo debería aprender mejores representaciones geométricas y mejorar la segmentación final, especialmente en clases difíciles como baja vegetación, vegetación media, bordes de edificios y regiones con RGB/NIR ausente.

La variante más ambiciosa incorpora el algoritmo **Taubin-Weingarten** como señal geométrica auxiliar:

> **TW-JEPA**: JEPA regularizada por descriptores geométricos de segundo orden: curvaturas principales, curvatura media, curvatura gaussiana, gradiente, Hessiano y features de cuádrica local.

---

## 1. Artículos que Codex debe tener en cuenta

### 1.1. Paper base que se quiere mejorar

1. **Deep Learning for Ultra-Large-Scale Semantic Segmentation of Geographic 3D Point Clouds With Missing Labels**  
   - Leer completo antes de programar.
   - Extraer de aquí:
     - Dataset PNOA-II Galicia.
     - Problema de etiquetas ausentes.
     - Baselines: PointNet, PointNet++, KPConv, SFL-Net, Random Forest.
     - Métricas: OA, IoU, Precision, Recall, F1, weighted metrics, MCC, Cohen's kappa.
     - Métrica de incertidumbre: entropy vs class ambiguity.
     - Tareas experimentales: vegetación, edificios, baja/media/alta vegetación, edificios+vegetación, full classification.
     - Framework: fork de VirtuaLearn3D++.

### 1.2. Paper geométrico complementario

2. **Introducing the Taubin-Weingarten algorithm to compute second-order geometric descriptors in 3D point clouds**  
   - Leer para implementar o integrar descriptores geométricos de segundo orden.
   - Extraer:
     - Algoritmo de ajuste cuadrático local.
     - Cálculo de curvaturas principales.
     - Curvatura gaussiana y media.
     - Descriptores diferenciales y de autovalores.
     - Estrategia paralelizable para datasets grandes.
   - Usar como base para la variante **TW-JEPA**.

### 1.3. Papers JEPA / SSL que hay que revisar

3. **I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**.
4. **V-JEPA / Video JEPA**.
5. **DINO / DINOv2** como referencia de preentrenamiento autosupervisado robusto.
6. **VICReg / Barlow Twins / BYOL** para pérdidas anticolapso y regularización de embeddings.
7. **Point-BERT**, **Point-MAE**, **Point-M2AE**, **MaskPoint** o equivalentes de masked modeling en nubes de puntos.
8. **PointNet**, **PointNet++**, **KPConv**, **SFL-Net**.
9. Trabajos de segmentación LiDAR large-scale: Semantic3D, DALES, SensatUrban, Toronto-3D, SemanticKITTI, FRACTAL.

---

## 2. Datos que hay que descargar

### 2.1. Dataset principal: PNOA-II Galicia

Dataset objetivo:

- **PNOA-II Galicia ALS**.
- Fuente indicada en el paper base: página PNOA LiDAR, segunda cobertura, Ministerio de Transportes y Movilidad Sostenible / IGN.
- URL indicada en el artículo: `https://pnoa.ign.es/pnoa-lidar/segunda-cobertura`
- Región: Galicia, España.
- Campañas:
  - Galicia West: LEICA ALS60, julio-septiembre 2015.
  - Galicia East: LEICA ALS80, agosto 2016-febrero 2017.
- Escala del paper:
  - Aproximadamente 36.369 millones de puntos.
  - Aproximadamente 29.557 km².
  - Densidad mínima garantizada: 0.5 pts/m².
  - Densidad media aproximada: 1.2 pts/m².
  - Error vertical: 20 cm.
  - Error planimétrico: 30 cm.
  - Etiquetas fiables de entrenamiento: ~61.52%.

No intentar descargar todo al principio. Implementar primero en subconjuntos.

### 2.2. Subconjuntos recomendados

Crear tres niveles de datos:

#### Nivel A — Smoke test local

Objetivo: ejecutar en portátil o workstation.

- 1-2 teselas pequeñas.
- Máximo: 1-5 millones de puntos.
- Usar para:
  - pruebas de lectura LAZ/LAS;
  - sampling;
  - máscaras JEPA;
  - forward/backward;
  - tests de métricas;
  - generación de figuras.

#### Nivel B — Experimento piloto

Objetivo: demostrar si JEPA ayuda antes de ir a escala Galicia.

- 4-8 teselas por provincia.
- Incluir variedad:
  - zona urbana: Santiago / A Coruña / Vigo / Lugo / Ourense;
  - zona rural con edificios dispersos;
  - zona forestal;
  - zona con baja vegetación;
  - zona con RGB/NIR faltante si se detecta.
- Usar para primera comparación seria.

#### Nivel C — Reproducción parcial del paper

Objetivo: aproximar los experimentos del artículo.

- Usar una nube masiva por provincia como hace el paper.
- Entrenar por provincias con secuencia similar a la del artículo.
- Requiere GPU/HPC.

#### Nivel D — Escala completa

Objetivo: solo al final.

- Galicia completa.
- Requiere almacenamiento, SQL/LAZ, entrenamiento distribuido o HPC.
- No bloquear el proyecto en este nivel.

### 2.3. Datasets secundarios opcionales

Usar solo si PNOA-II no está disponible o para depurar:

- Semantic3D.
- DALES.
- SensatUrban.
- Toronto-3D.
- SemanticKITTI.
- Vaihingen 3D.
- ShapeNetPart para validar TW-JEPA en object part segmentation.

---

## 3. Objetivo experimental exacto

El artículo original ya obtiene buenos resultados con SFL-Net, especialmente en:

- vegetación binaria;
- edificios binarios;
- high vegetation.

Los puntos débiles del artículo son:

1. baja vegetación;
2. vegetación media;
3. regiones con RGB/NIR ausente;
4. límites de edificios;
5. clases minoritarias;
6. uso pasivo de puntos sin etiqueta: se introducen como geometría, pero no generan pérdida supervisada.

Por tanto, el objetivo de JEPA debe ser:

> Aprovechar los puntos sin etiqueta fiable para aprender una representación 3D autosupervisada que mejore la segmentación final, sobre todo en baja/media vegetación y regiones de incertidumbre alta.

---

## 4. Arquitectura propuesta

### 4.1. Baseline que hay que reproducir

Antes de implementar JEPA:

1. Clonar el fork original:

```bash
git clone https://github.com/albertoesmp/vl3d_galicia.git
cd vl3d_galicia
```

2. Crear entorno:

```bash
conda create -n geopoint-jepa python=3.10 -y
conda activate geopoint-jepa
pip install -r requirements.txt || true
```

3. Si el repo no trae requirements completos, crear `requirements-dev.txt` con:

```txt
numpy
scipy
pandas
scikit-learn
matplotlib
seaborn
laspy[lazrs]
pyproj
shapely
geopandas
open3d
tqdm
pyyaml
rich
torch
torchvision
pytest
pytest-cov
ruff
black
mypy
```

4. Reproducir al menos un baseline SFL-Net en un subset pequeño.

### 4.2. GeoPoint-JEPA básico

Entrada de una muestra:

```text
P = [x, y, z, NIR, R, G, B, optional_features]
```

Se toma una ventana espacial, por ejemplo:

- bloque rectangular de 50 m, 100 m o 200 m;
- o vecindario esférico de radio configurable;
- con `N` puntos muestreados.

Partición JEPA:

```text
P_context = puntos visibles
P_target  = región oculta / máscara espacial
```

Encoder:

```text
z_context = E_context(P_context)
z_target  = E_target(P_target)
```

Predictor:

```text
z_pred = Predictor(z_context, mask_token, target_position_encoding)
```

Pérdida:

```text
L_JEPA = SmoothL1(z_pred, stopgrad(z_target))
```

Regularización anticolapso:

```text
L_varcov = variance_covariance_regularization(z_context, z_target)
```

Pérdida total de pretraining:

```text
L_pretrain = L_JEPA + lambda_reg * L_varcov
```

Notas:

- Usar `E_target` como copia EMA de `E_context` o como encoder no-grad actualizado periódicamente.
- Empezar con EMA porque suele estabilizar el entrenamiento.
- Implementar opción `--target-ema 0.996` configurable.

### 4.3. TW-JEPA: variante geométrica recomendada

Añadir descriptores Taubin-Weingarten por punto o por región:

```text
TW = [kappa_min, kappa_max, mean_curvature, gaussian_curvature,
      gradient_norm, laplacian, hessian_frobenius,
      quadratic_deviation, saddleness, ...]
```

Dos modos:

#### Modo 1 — TW como entrada

```text
P = [x, y, z, NIR, R, G, B, TW_features]
```

Ventaja: simple.

Riesgo: coste de calcular TW para todos los puntos.

#### Modo 2 — TW como teacher geométrico

El encoder no recibe TW. JEPA debe predecir también los descriptores de la zona oculta.

```text
tw_pred = TWHead(z_pred)
L_TW = SmoothL1(tw_pred, tw_target)
L_pretrain = L_JEPA + lambda_tw * L_TW + lambda_reg * L_varcov
```

Ventaja: durante inferencia no es necesario recalcular TW.

Recomendación: implementar ambos, pero priorizar Modo 2 para una contribución más publicable.

---

## 5. Estructura de carpetas recomendada

Crear o adaptar esta estructura:

```text
project/
  PLAN.md
  README.md
  configs/
    data/
      pnoa_galicia_smoke.yaml
      pnoa_galicia_pilot.yaml
    pretrain/
      geopoint_jepa_sflnet.yaml
      tw_jepa_sflnet.yaml
    finetune/
      sflnet_baseline.yaml
      sflnet_jepa_finetune.yaml
      sflnet_tw_jepa_finetune.yaml
    eval/
      full_classification.yaml
      vegetation_lmh.yaml
      building_binary.yaml
  data/
    raw/                 # no subir a git
    processed/           # no subir a git
    manifests/
      pnoa_galicia_tiles.csv
      pilot_tiles.csv
  src/
    data/
      pnoa_download.py
      laz_io.py
      tiling.py
      sampling.py
      masks.py
      transforms.py
    geometry/
      tw_features.py
      pca_features.py
      neighborhoods.py
    models/
      encoders/
        sflnet_encoder.py
        kpconv_encoder.py
      jepa/
        geopoint_jepa.py
        predictor.py
        losses.py
        ema.py
      segmentation/
        heads.py
        losses.py
    train/
      pretrain_jepa.py
      finetune_segmentation.py
      train_baseline.py
    eval/
      metrics.py
      uncertainty.py
      calibration.py
      benchmark.py
    viz/
      plots.py
      pointcloud_figures.py
      latex_tables.py
  scripts/
    00_download_pnoa_subset.py
    01_prepare_tiles.py
    02_compute_tw_features.py
    03_train_baseline.py
    04_pretrain_jepa.py
    05_finetune_jepa.py
    06_evaluate.py
    07_plot_results.py
    08_make_latex_report.py
  tests/
    test_masks.py
    test_jepa_loss.py
    test_ema.py
    test_weighted_ce_missing_labels.py
    test_metrics.py
    test_uncertainty.py
    test_tw_features_synthetic.py
    test_overfit_tiny_tile.py
  paper/
    main.tex
    sections/
      01_introduction.tex
      02_related_work.tex
      03_method.tex
      04_experiments.tex
      05_results.tex
      06_discussion.tex
      07_conclusion.tex
    figures/
    tables/
    refs.bib
  outputs/
    checkpoints/
    logs/
    metrics/
    figures/
```

---

## 6. Scripts que hay que implementar

### 6.1. `scripts/00_download_pnoa_subset.py`

Objetivo: crear un mecanismo reproducible para descargar o registrar teselas PNOA-II.

Funcionalidad mínima:

- Leer `data/manifests/pilot_tiles.csv`.
- Descargar teselas si hay URL directa.
- Si no hay URL directa, permitir modo manual:
  - validar que los ficheros `.las`, `.laz`, `.txt`, `.csv`, `.tif` existen en `data/raw`;
  - generar checksum SHA256;
  - escribir `data/processed/dataset_index.json`.

CLI:

```bash
python scripts/00_download_pnoa_subset.py \
  --manifest data/manifests/pilot_tiles.csv \
  --out data/raw/pnoa_galicia \
  --mode manual-or-download
```

### 6.2. `scripts/01_prepare_tiles.py`

Objetivo: convertir LAZ/LAS a formato eficiente.

Debe:

- leer coordenadas XYZ;
- leer intensidad si existe;
- leer RGB/NIR si existe;
- leer labels si existen;
- convertir labels a clases del paper:
  - ground;
  - water;
  - low vegetation;
  - mid vegetation;
  - high vegetation;
  - building;
  - unreliable/noisy/overlap;
- normalizar coordenadas por tile o bloque;
- crear ventanas espaciales;
- guardar en Zarr, Parquet, HDF5 o formato propio binario.

CLI:

```bash
python scripts/01_prepare_tiles.py \
  --raw data/raw/pnoa_galicia \
  --out data/processed/pnoa_galicia_smoke \
  --tile-size 100 \
  --points-per-sample 8192 \
  --include-rgb \
  --include-nir
```

### 6.3. `scripts/02_compute_tw_features.py`

Objetivo: calcular descriptores Taubin-Weingarten.

Debe:

- construir vecindarios con kNN o radio;
- calcular PCA first-order features;
- calcular TW second-order features;
- guardar features por punto o por bloque;
- soportar multiproceso.

CLI:

```bash
python scripts/02_compute_tw_features.py \
  --input data/processed/pnoa_galicia_smoke \
  --out data/processed/pnoa_galicia_smoke_tw \
  --radius 3.0 \
  --min-neighbors 16 \
  --num-workers 16
```

Tests obligatorios:

- plano sintético: curvatura aproximadamente 0;
- esfera sintética: curvaturas aproximadamente 1/R;
- cilindro sintético: una curvatura aproximadamente 1/R y otra aproximadamente 0;
- superficie silla: curvaturas con signos opuestos.

### 6.4. `scripts/03_train_baseline.py`

Objetivo: reproducir baseline SFL-Net/KPConv.

CLI:

```bash
python scripts/03_train_baseline.py \
  --config configs/finetune/sflnet_baseline.yaml
```

Debe guardar:

```text
outputs/checkpoints/baseline_sflnet.pt
outputs/metrics/baseline_sflnet_val.json
outputs/logs/baseline_sflnet.csv
```

### 6.5. `scripts/04_pretrain_jepa.py`

Objetivo: preentrenar encoder JEPA sin usar etiquetas.

CLI:

```bash
python scripts/04_pretrain_jepa.py \
  --config configs/pretrain/geopoint_jepa_sflnet.yaml
```

Debe soportar:

- masking espacial por bloques;
- masking aleatorio por puntos;
- masking por altura;
- masking por modalidad: simular RGB/NIR ausente;
- EMA target encoder;
- regularización anticolapso;
- checkpointing;
- mixed precision.

Outputs:

```text
outputs/checkpoints/geopoint_jepa_pretrained.pt
outputs/metrics/geopoint_jepa_pretrain.json
outputs/figures/pretrain_loss_curve.png
```

### 6.6. `scripts/05_finetune_jepa.py`

Objetivo: cargar encoder preentrenado y entrenar cabeza de segmentación.

CLI:

```bash
python scripts/05_finetune_jepa.py \
  --config configs/finetune/sflnet_jepa_finetune.yaml \
  --pretrained outputs/checkpoints/geopoint_jepa_pretrained.pt
```

Debe implementar la misma pérdida del paper para missing labels:

```text
L_supervised = <theta, y> * CrossEntropy(y, y_hat)
```

con peso cero para la clase no fiable.

### 6.7. `scripts/06_evaluate.py`

Objetivo: evaluación completa.

CLI:

```bash
python scripts/06_evaluate.py \
  --config configs/eval/full_classification.yaml \
  --checkpoint outputs/checkpoints/sflnet_jepa_finetuned.pt \
  --out outputs/metrics/sflnet_jepa_full.json
```

Métricas:

- OA;
- P/R/F1 macro;
- weighted P/R/F1;
- IoU macro;
- weighted IoU;
- class-wise F1;
- class-wise IoU;
- MCC;
- Cohen's kappa;
- confusion matrix;
- entropy;
- class ambiguity;
- Pearson/Spearman correlation between uncertainty cuts and F1;
- ECE/Brier score opcional;
- runtime por tile;
- GPU memory;
- RAM peak.

### 6.8. `scripts/07_plot_results.py`

Debe generar:

- barras de F1 por clase;
- barras de IoU por clase;
- matriz de confusión;
- curva de pérdida JEPA;
- curva de fine-tuning;
- scatter uncertainty vs error;
- curva F1 vs threshold de class ambiguity;
- mapas visuales:
  - referencia;
  - predicción baseline;
  - predicción JEPA;
  - error baseline;
  - error JEPA;
  - diferencia de errores;
  - class ambiguity.

### 6.9. `scripts/08_make_latex_report.py`

Objetivo: generar tablas `.tex` y copiar figuras a `paper/figures`.

CLI:

```bash
python scripts/08_make_latex_report.py \
  --metrics outputs/metrics \
  --figures outputs/figures \
  --out paper/
```

---

## 7. Tests obligatorios

### 7.1. Tests unitarios

Implementar con pytest.

#### `tests/test_masks.py`

- comprobar que context y target no se solapan;
- comprobar que target no queda vacío;
- comprobar reproducibilidad con seed;
- comprobar ratios de máscara.

#### `tests/test_jepa_loss.py`

- si `z_pred == z_target`, la loss debe ser ~0;
- si se permuta el target, la loss debe subir;
- comprobar que `stopgrad` no propaga gradiente al target encoder.

#### `tests/test_ema.py`

- comprobar que EMA actualiza pesos;
- comprobar que EMA no iguala exactamente al encoder tras un paso si `tau < 1`;
- comprobar guardado/carga de estado.

#### `tests/test_weighted_ce_missing_labels.py`

- clase no fiable con peso 0 no debe contribuir a la loss;
- gradiente debe ser 0 para muestras no fiables;
- gradiente debe existir para muestras fiables.

#### `tests/test_metrics.py`

- verificar OA, F1, IoU, MCC, kappa con arrays pequeños conocidos;
- verificar clase ausente sin NaN.

#### `tests/test_uncertainty.py`

- class ambiguity debe ser 0 cuando top1=1 y top2=0;
- class ambiguity debe ser alta cuando top1 y top2 están cerca;
- comparar entropy y class ambiguity en casos sintéticos.

#### `tests/test_tw_features_synthetic.py`

- plano: curvaturas ~0;
- esfera: curvaturas ~1/R;
- cilindro: k1 ~1/R, k2 ~0;
- silla: k1 y k2 de signo opuesto.

### 7.2. Tests de integración

#### `tests/test_overfit_tiny_tile.py`

- seleccionar una tesela mínima;
- entrenar 10-50 steps;
- verificar que la loss baja;
- verificar que el checkpoint se guarda;
- verificar que la evaluación produce JSON válido.

#### Smoke command

```bash
pytest -q
python scripts/03_train_baseline.py --config configs/finetune/sflnet_baseline_smoke.yaml
python scripts/04_pretrain_jepa.py --config configs/pretrain/geopoint_jepa_smoke.yaml
python scripts/05_finetune_jepa.py --config configs/finetune/sflnet_jepa_smoke.yaml
python scripts/06_evaluate.py --config configs/eval/smoke.yaml
```

---

## 8. Experimentos principales

### 8.1. Baselines

Ejecutar en el mismo split:

1. **Random Forest** con features geométricas clásicas.
2. **SFL-Net baseline** igual al paper.
3. **SFL-Net + TW features** sin JEPA.
4. **GeoPoint-JEPA + SFL-Net fine-tuning**.
5. **TW-JEPA + SFL-Net fine-tuning**.

### 8.2. Ablations

Ablaciones mínimas:

| Experimento | Descripción |
|---|---|
| Baseline | SFL-Net supervisado |
| JEPA-no-TW | Pretraining JEPA estándar |
| TW-input | SFL-Net con TW como entrada |
| TW-teacher | JEPA predice embeddings + TW targets |
| Mask-random | máscara aleatoria por puntos |
| Mask-block | máscara espacial por bloques |
| Mask-modality | simular RGB/NIR ausente |
| No-EMA | target encoder sin EMA |
| No-reg | sin regularización anticolapso |

### 8.3. Tareas a evaluar

Reproducir las del paper:

1. Vegetación binaria.
2. Edificios binarios.
3. Baja/media/alta vegetación.
4. Edificios + vegetación + other.
5. Full classification:
   - ground;
   - water;
   - low vegetation;
   - mid vegetation;
   - high vegetation;
   - building.

### 8.4. Criterios de éxito

El proyecto se considera exitoso si se cumple al menos una de estas condiciones:

1. **Mejora macro-F1** en full classification ≥ +2 puntos frente a SFL-Net baseline.
2. **Mejora F1 en low vegetation** ≥ +5 puntos.
3. **Mejora F1 en mid vegetation** ≥ +5 puntos.
4. **Mejora IoU en clases minoritarias** sin empeorar weighted-F1 más de 0.5 puntos.
5. **Mejor incertidumbre**: class ambiguity correlaciona mejor con error que baseline.
6. **Mejor robustez a RGB/NIR faltante**: menor caída de F1 al ocultar modalidades.
7. **Menos etiquetas necesarias**: con 10%, 25% o 50% de etiquetas fiables, JEPA supera claramente al baseline entrenado desde cero.

Si no mejora F1 pero mejora calibración, robustez o rendimiento con pocas etiquetas, también puede ser una contribución válida.

---

## 9. Evaluación de si JEPA mejora o no el artículo

Codex debe producir una tabla final como esta:

| Modelo | Full F1 | Full IoU | Low-Veg F1 | Mid-Veg F1 | High-Veg F1 | Building F1 | MCC | Kappa | Runtime | Mejora clara |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RF original | | | | | | | | | | |
| SFL-Net baseline | | | | | | | | | | referencia |
| SFL-Net + TW | | | | | | | | | | sí/no |
| GeoPoint-JEPA | | | | | | | | | | sí/no |
| TW-JEPA | | | | | | | | | | sí/no |

También debe producir una tabla de robustez:

| Modelo | Normal F1 | RGB missing F1 | NIR missing F1 | 25% labels F1 | 10% labels F1 |
|---|---:|---:|---:|---:|---:|
| SFL-Net | | | | | |
| GeoPoint-JEPA | | | | | |
| TW-JEPA | | | | | |

Interpretación esperada:

- Si JEPA mejora sobre todo con pocas etiquetas, decir que aporta representación autosupervisada.
- Si JEPA mejora low/mid vegetation, decir que captura geometría contextual mejor que el baseline.
- Si TW-JEPA mejora bordes/curvatura/edificios, decir que los descriptores de segundo orden aportan señal geométrica útil.
- Si no mejora, analizar:
  - masking mal diseñado;
  - encoder demasiado pequeño;
  - pretraining insuficiente;
  - pérdida colapsada;
  - target demasiado fácil/difícil;
  - ruido/aleatoric uncertainty irreducible por baja densidad y error vertical.

---

## 10. LaTeX final que hay que redactar

Crear `paper/main.tex` con estructura de paper científico.

### 10.1. Título provisional

```tex
\title{GeoPoint-JEPA: Self-Supervised Latent Predictive Pretraining for Ultra-Large-Scale ALS Point Cloud Segmentation With Missing Labels}
```

Variante con Taubin-Weingarten:

```tex
\title{TW-JEPA: Curvature-Aware Joint Embedding Predictive Learning for Ultra-Large-Scale Geographic 3D Point Clouds}
```

### 10.2. Secciones

```tex
\section{Introduction}
\section{Related Work}
\section{Dataset and Baseline}
\section{Method}
\subsection{GeoPoint-JEPA}
\subsection{Taubin-Weingarten Geometry Targets}
\subsection{Semi-supervised Fine-tuning With Missing Labels}
\section{Experimental Setup}
\section{Results}
\section{Ablation Study}
\section{Uncertainty and Robustness Analysis}
\section{Discussion}
\section{Limitations}
\section{Conclusion}
```

### 10.3. Diagramas obligatorios

Generar con TikZ o exportar como PNG/SVG desde Python.

1. **Figura 1 — Pipeline general**
   - PNOA-II → preprocessing → JEPA pretraining → fine-tuning → evaluation.

2. **Figura 2 — Arquitectura GeoPoint-JEPA**
   - context block;
   - target block;
   - context encoder;
   - EMA target encoder;
   - predictor;
   - latent loss.

3. **Figura 3 — TW-JEPA**
   - vecindario 3D;
   - ajuste cuadrático Taubin;
   - curvaturas/Weingarten;
   - loss geométrica auxiliar.

4. **Figura 4 — Máscaras espaciales**
   - random point mask;
   - block mask;
   - height mask;
   - modality mask.

5. **Figura 5 — Resultados cualitativos**
   - referencia;
   - baseline;
   - JEPA;
   - error baseline;
   - error JEPA;
   - mejora/error-diff.

6. **Figura 6 — Incertidumbre**
   - class ambiguity baseline;
   - class ambiguity JEPA;
   - F1 vs ambiguity threshold.

### 10.4. Gráficas obligatorias

Generar desde `scripts/07_plot_results.py`:

- `fig_pretrain_loss.png`
- `fig_finetune_loss.png`
- `fig_class_f1_bars.png`
- `fig_class_iou_bars.png`
- `fig_confusion_baseline.png`
- `fig_confusion_jepa.png`
- `fig_uncertainty_f1_curve.png`
- `fig_missing_rgb_robustness.png`
- `fig_label_efficiency.png`
- `fig_runtime_memory.png`

### 10.5. Tablas obligatorias

- Tabla 1: datasets y subsets.
- Tabla 2: modelos comparados.
- Tabla 3: resultados globales.
- Tabla 4: resultados por clase.
- Tabla 5: ablation study.
- Tabla 6: robustez a etiquetas escasas.
- Tabla 7: robustez a RGB/NIR ausente.
- Tabla 8: coste computacional.

---

## 11. Plan de trabajo por fases

### Fase 1 — Reproducción mínima

- [ ] Clonar repo base.
- [ ] Crear entorno.
- [ ] Descargar/registrar subset pequeño PNOA-II.
- [ ] Preparar teselas.
- [ ] Ejecutar baseline SFL-Net o KPConv en smoke subset.
- [ ] Verificar métricas y visualizaciones.

Criterio de salida:

```text
Baseline entrena, evalúa y genera metrics.json + figuras.
```

### Fase 2 — Implementar JEPA básico

- [ ] Implementar spatial masking.
- [ ] Implementar context encoder.
- [ ] Implementar target encoder EMA.
- [ ] Implementar predictor.
- [ ] Implementar JEPA loss.
- [ ] Implementar regularización anticolapso.
- [ ] Tests unitarios.
- [ ] Pretraining en smoke subset.

Criterio de salida:

```text
La loss JEPA baja, los embeddings no colapsan y el checkpoint carga en fine-tuning.
```

### Fase 3 — Fine-tuning supervisado

- [ ] Añadir cabeza de segmentación.
- [ ] Implementar pérdida con peso cero para missing labels.
- [ ] Comparar baseline vs JEPA en smoke subset.
- [ ] Verificar que no hay leakage de etiquetas en pretraining.

Criterio de salida:

```text
JEPA fine-tune funciona y produce métricas comparables al baseline.
```

### Fase 4 — Implementar TW features

- [ ] Implementar/desacoplar `src/geometry/tw_features.py`.
- [ ] Tests sintéticos.
- [ ] Calcular TW features en subset.
- [ ] Entrenar SFL-Net + TW input.
- [ ] Entrenar TW-JEPA teacher.

Criterio de salida:

```text
TW features son geométricamente razonables y no rompen memoria/tiempo.
```

### Fase 5 — Experimento piloto serio

- [ ] Ejecutar en varias teselas por provincia.
- [ ] Medir por región, clase y modalidad.
- [ ] Ejecutar ablations.
- [ ] Generar figuras del paper.

Criterio de salida:

```text
Hay resultados suficientes para decir si JEPA mejora o no.
```

### Fase 6 — Reporte LaTeX

- [ ] Crear `paper/main.tex`.
- [ ] Generar tablas desde JSON/CSV.
- [ ] Generar figuras.
- [ ] Escribir discusión honesta.
- [ ] Compilar PDF.

Comando:

```bash
cd paper
latexmk -pdf main.tex
```

---

## 12. Configuraciones mínimas

### 12.1. `configs/pretrain/geopoint_jepa_sflnet.yaml`

```yaml
seed: 42
experiment_name: geopoint_jepa_sflnet

data:
  root: data/processed/pnoa_galicia_smoke
  points_per_sample: 8192
  features: [x, y, z, nir, r, g, b]
  use_labels: false

masking:
  type: block
  context_ratio: 0.65
  target_ratio: 0.25
  min_target_points: 512
  num_targets: 4

model:
  encoder: sflnet
  embedding_dim: 256
  predictor_hidden_dim: 512
  target_ema: 0.996

loss:
  jepa: smooth_l1
  lambda_reg: 0.05
  lambda_tw: 0.0

train:
  epochs: 100
  batch_size: 8
  lr: 0.0003
  weight_decay: 0.0001
  amp: true
  num_workers: 8
  save_every: 5
```

### 12.2. `configs/pretrain/tw_jepa_sflnet.yaml`

```yaml
seed: 42
experiment_name: tw_jepa_sflnet

data:
  root: data/processed/pnoa_galicia_smoke_tw
  points_per_sample: 8192
  features: [x, y, z, nir, r, g, b]
  tw_features: [kappa_min, kappa_max, mean_curvature, gaussian_curvature, gradient_norm, hessian_frobenius]
  use_tw_as_input: false
  use_tw_as_target: true

masking:
  type: block
  context_ratio: 0.65
  target_ratio: 0.25
  min_target_points: 512
  num_targets: 4

model:
  encoder: sflnet
  embedding_dim: 256
  predictor_hidden_dim: 512
  target_ema: 0.996
  tw_head_hidden_dim: 128

loss:
  jepa: smooth_l1
  lambda_reg: 0.05
  lambda_tw: 0.25

train:
  epochs: 100
  batch_size: 8
  lr: 0.0003
  weight_decay: 0.0001
  amp: true
  num_workers: 8
  save_every: 5
```

---

## 13. Riesgos y mitigaciones

### Riesgo 1 — JEPA no mejora el baseline

Mitigación:

- probar label efficiency;
- probar robustness con RGB/NIR ausente;
- probar low/mid vegetation específicamente;
- probar masks más difíciles;
- añadir TW teacher.

### Riesgo 2 — Colapso de embeddings

Mitigación:

- EMA target encoder;
- variance/covariance regularization;
- predictor MLP no demasiado fuerte;
- monitorizar varianza media de embeddings;
- test de nearest-neighbor retrieval.

### Riesgo 3 — Demasiado coste computacional

Mitigación:

- empezar con KPConv/SFL-Net pequeño;
- AMP;
- gradient accumulation;
- precalcular TW offline;
- usar subsets;
- no calcular TW en inferencia si se usa como teacher.

### Riesgo 4 — TW features inestables en ALS de baja densidad

Mitigación:

- usar radios mayores;
- exigir mínimo de vecinos;
- normalizar features robustamente;
- winsorizar outliers;
- comparar radii 3m, 5m, 10m;
- incluir PCA first-order fallback.

### Riesgo 5 — Mejora aparente por split incorrecto

Mitigación:

- separar geográficamente train/val/test;
- evaluación por provincia;
- evaluación por campaña Galicia West/East;
- no mezclar puntos de la misma tesela entre train y test.

---

## 14. Resultado final esperado

Al acabar, el repositorio debe contener:

1. Código funcional de GeoPoint-JEPA.
2. Código funcional de TW-JEPA o al menos TW teacher.
3. Scripts reproducibles.
4. Tests pasando.
5. Resultados JSON/CSV.
6. Figuras comparativas.
7. Un PDF LaTeX final que diga claramente:
   - qué mejora;
   - cuánto mejora;
   - dónde no mejora;
   - si el coste computacional compensa;
   - si la mejora es científicamente defendible.

Conclusión esperada si sale bien:

> GeoPoint-JEPA mejora la segmentación ultra-large-scale con etiquetas ausentes al aprovechar puntos no etiquetados durante el preentrenamiento. TW-JEPA aporta mejoras adicionales en regiones geométricamente ambiguas al introducir objetivos de curvatura y estructura diferencial local.

Conclusión esperada si no sale bien:

> JEPA no mejora de forma significativa el baseline SFL-Net en este escenario, probablemente porque el error principal en baja vegetación es aleatorio/físico por baja densidad ALS y error vertical, no puramente representacional. Aun así, puede aportar robustez en condiciones de pocas etiquetas o modalidades faltantes.

---

## 15. Checklist final para Codex

- [ ] `pytest -q` pasa.
- [ ] Baseline reproduce resultados razonables en subset.
- [ ] JEPA pretraining no colapsa.
- [ ] Fine-tuning carga encoder preentrenado.
- [ ] Missing labels tienen gradiente cero.
- [ ] Métricas coinciden con definiciones del paper.
- [ ] Class ambiguity implementada.
- [ ] Figuras generadas.
- [ ] Tablas generadas.
- [ ] LaTeX compila.
- [ ] README explica cómo ejecutar todo.
- [ ] No se suben datos pesados a git.
- [ ] No se afirma mejora si no hay evidencia estadística.

---

## 16. Plan de Inferencia a Ultra-Gran Escala (Escala Completa Galicia)

Para escalar la inferencia de los modelos entrenados (Fase 6) a toda la extensión geográfica de Galicia (36.369 millones de puntos, ~400GB en crudo), usaremos un enfoque diseñado para ser eficiente en workstations locales (ej. RTX 5070 Ti) minimizando el cuello de botella de I/O y memoria RAM.

### 16.1. Metodología de Inferencia por Streaming
- **Tiling al vuelo:** No cargar la provincia entera en RAM. Leer archivos `.laz` individuales y procesarlos por baldosas (ej. 100x100m).
- **Procesamiento en Batch:** Computar características (TW, JEPA) de forma local. Mover un batch de baldosas a la VRAM, aplicar `torch.no_grad()` para la inferencia, extraer las predicciones (clase predicha) y liberar memoria. Esto mantiene el consumo de VRAM de la GPU estable y por debajo de 4GB-8GB.

### 16.2. Estrategia de Almacenamiento (Evitar 800GB)
Dado que duplicar los 400GB de `.laz` crudos no es logísticamente razonable, se utilizará una de las siguientes opciones:
1. **El Truco del Mapeo (Numpy int8):** No duplicar la geometría ni RGB/NIR. Guardar únicamente un array con la etiqueta predicha para cada punto original. Al usar `int8` para 6 clases, los 36.369 millones de puntos ocupan teóricamente apenas **36 GB** (sin comprimir).
2. **Edición In-Place (LAZ Classification Field):** Usar librerías modernas de LiDAR (`laspy`, `pdal`) para modificar de forma exclusiva el campo `Classification` del archivo LAZ original que ya se encuentra en el disco, sobreescribiendo etiquetas con predicciones en la misma posición de memoria del archivo.

### 16.3. Evaluación frente a SFL-Net
Para calcular la verdadera precisión (mIoU) del modelo sobre toda Galicia y compararlo con el 56.06% del artículo original:
- A medida que se procesan bloques, cotejar las predicciones del modelo *exclusivamente* contra los puntos que poseen etiqueta real humana (el 61.5% de los puntos). Los puntos sin etiqueta fiable se ignoran en el cálculo estadístico.
- Mantener una única **Matriz de Confusión Global** en memoria que acumula todos los aciertos y fallos de cada archivo `.laz` progresivamente.
- Al final de la inferencia, aplicar las fórmulas de mIoU, F1 y Kappa sobre la matriz acumulada final. El número arrojado podrá compararse 1:1 con el estado del arte publicado, validando o desmintiendo la robustez de TW-JEPA en el dominio global y no solo en la muestra del Benchmark.

---

## 17. Configuración de Hardware Óptima para Windows (Ryzen 9, 32GB RAM, RTX 5070 Ti)

Para evitar colapsos de RAM y maximizar el rendimiento computacional durante el entrenamiento, se debe emplear la siguiente configuración:

1. **`batch_size: 16`**: Aumenta la carga en la GPU, llenando aproximadamente 11-12 GB de la VRAM de la RTX 5070 Ti sin provocar OOM.
2. **`num_workers: 2` (Máximo 4)**: En Windows, PyTorch usa `spawn`, lo que clona el entorno y consume RAM física rápidamente. Más de 2 *workers* saturarán los 32GB de RAM, forzando la paginación al disco (Swapping) y asfixiando el CPU.
3. **`prefetch_factor: 2`**: (Opcional) Mantiene dos lotes de datos preparados en RAM listos para alimentar a la GPU sin esperas de I/O.
4. **`Resiliencia (Checkpointing por Época)`**: Obligatorio programar en `segmentation_trainer.py` un guardado de `latest_checkpoint.pt` al final de cada época y una función de carga al inicio. Esto blinda el entrenamiento contra errores *Out of Memory* (OOM) o reinicios, permitiendo retomar el progreso exacto sin perder horas de computación.

---

## 18. Tareas Futuras (Fase 7)

### 18.1. Implementación Estricta del Taubin-Weingarten (Algoritmo 3)
Actualmente, el pipeline extrae los descriptores geométricos asumiendo la forma de Monge ($z = \hat{z}(x,y)$) y calculando parábolas en 2.5D mediante regresión lineal (estilo CloudCompare). Para la próxima gran iteración del modelo, es imperativo **reescribir el archivo `src/features/taubin_weingarten.py`** para que implemente con exactitud el "Algoritmo 3" matemático expuesto en el artículo científico:

* **Modelo Geométrico:** Sustituir la regresión de 6 variables por una **ecuación cuadrática implícita puramente en 3D** ($g(x,y,z) = 0$).
* **Monomios:** El modelo debe escalar a las **10 variables** exigidas por el paper ($x, y, z, xy, xz, yz, x^2, y^2, z^2, 1$).
* **Optimización Matemática:** Reemplazar el cálculo actual por el auténtico **Método de Taubin**, lo que requiere plantear y resolver un **Problema de Eigenvalores Generalizados** minimizando la distancia geométrica real, no la distancia algebraica sesgada.

Esta mejora promete erradicar la distorsión matemática en objetos complejos (farolas, vegetación, filos de edificios) y proporcionar al TW-JEPA una señal de curvatura drásticamente más pura.
