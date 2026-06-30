# Fase 3: Diseño Técnico GeoPoint-JEPA

Este documento especifica el diseño para la implementación limpia de una arquitectura autosupervisada JEPA adaptada a nubes de puntos ALS (GeoPoint-JEPA) utilizando la técnica SIGReg para evitar el colapso, según las alineaciones con *LeWorldModel* y *LeJEPA*.

## 1. Formato de Entrada

*   **Bloques/Patches:** La entrada de red son los bloques espaciales normalizados generados en la Fase 1.
*   **Features:** `[x, y, z, intensity, r, g, b, nir]` (8 dimensiones).
*   **Uso de Datos:** Se usarán **todos los puntos** en la fase de preentrenamiento, ignorando si son confiables o no, y sin utilizar las etiquetas semánticas (`labels`).
*   Las características TW-Lite *no* se incluirán en esta fase.

## 2. Estrategia de Masking Espacial

El mecanismo de JEPA requiere predecir regiones espaciales ocultas a partir de regiones visibles.
Se implementarán dos estrategias (y un fallback iterativo para evitar regiones vacías):

1.  **`random_point_mask`:** Enmascara puntos de manera aleatoria. Útil como control pero ineficiente espacialmente.
2.  **`spatial_block_mask`:** (Principal). Divide el bloque en una grilla o selecciona centros aleatorios con un radio determinado.
    *   **Contexto (`x_context`):** Puntos visibles para el encoder (ej. 50-70% del bloque).
    *   **Target (`x_target`):** Puntos ocultos que deben ser predichos (ej. 15-30% del bloque).
    *   **Restricciones:** Garantizar `min_context_points` y `min_target_points`. Si una máscara genera regiones vacías, se regenerará hasta cumplir los requisitos con un máximo de intentos.

## 3. Arquitectura del Encoder (Context Encoder & Target Encoder)

Para el *smoke test*, usaremos un **PointMLP** simplificado (estilo PointNet) por su velocidad.
*   **Red de Extracción:** MLPs compartidos sobre las 8 características iniciales (ej. `[8 -> 64 -> 128 -> 256]`).
*   **Agregación/Pooling:** Max-pooling o Average-pooling espacial sobre sub-bloques o el bloque entero, para producir un vector embedding latente por región.
*   **Dimensión Latente ($D$):** 256.
*   **Target Encoder:** En esta implementación (SIGReg limpio), la rama Target comparte los mismos pesos que el Context Encoder, *sin* usar promedios exponenciales móviles (EMA).

## 4. Predictor

Una red neuronal ligera que toma el embedding del contexto y (opcionalmente) tokens posicionales del target para predecir el embedding del target.
*   **Arquitectura:** MLP de 2 o 3 capas (ej. `[256 -> 128 -> 256]`).

## 5. Función de Pérdida JEPA & SIGReg

La pérdida final es: $\mathcal{L} = \mathcal{L}_{pred} + \lambda \mathcal{L}_{SIGReg}$

### 5.1. Pérdida Predictiva ($\mathcal{L}_{pred}$)
Error Cuadrático Medio (MSE) entre el embedding predicho por el predictor ($\hat{s}_y$) y el embedding codificado por el target encoder ($s_y$):
$$\mathcal{L}_{pred} = \frac{1}{N} \sum ||\hat{s}_y - s_y||^2_2$$

### 5.2. Regularización SIGReg ($\mathcal{L}_{SIGReg}$)
Para evitar el colapso representacional (donde el encoder colapsa todo a un punto o una dimensión), aplicamos SIGReg (basado en Variance-Covariance pero simplificado como maximización de entropía y descorrelación, o maximizando la distancia entre muestras).
*   **Varianza/Hinge:** Asegurar que la desviación estándar de cada dimensión en el batch se mantenga por encima de un umbral $\gamma$. $\mathcal{L}_{var} = \frac{1}{D} \sum \max(0, \gamma - \text{std}_j)$
*   **Covarianza:** Penalizar las correlaciones fuera de la diagonal en la matriz de covarianza del batch para desentrelazar las características.
*   **Nota:** SIGReg elimina la necesidad de EMA, acelerando el aprendizaje.

## 6. Pruebas Unitarias (Tests)

Se crearán los siguientes tests para validar el diseño antes del entrenamiento masivo:
1.  `test_jepa_dataset.py`: Verificar carga de tensores y ausencia de etiquetas.
2.  `test_spatial_mask.py`: Verificar que las máscaras respetan `min_context` y no generan vacíos.
3.  `test_geopoint_jepa_forward.py`: Garantizar dimensiones y flujos correctos del encoder y predictor.
4.  `test_sigreg_loss.py`: Verificar penalizaciones de colapso con datos sintéticos colapsados y dispersos.
5.  `test_jepa_no_nan.py`: Asegurar estabilidad numérica.
6.  `test_jepa_deterministic.py`: Probar reproducibilidad bajo una misma seed.
7.  `test_jepa_no_collapse.py`: Entrenar un micro-batch para verificar que los embeddings se repelen/separan.

## 7. Plan de Pretraining (Smoke Test)

*   **Script:** `09_pretrain_geopoint_jepa_smoke.py`
*   **Dataset:** 100 bloques de `train_smoke`.
*   **Métricas a Loguear:**
    *   `loss_total`, `loss_pred`, `loss_sigreg`.
    *   Varianza media del embedding y norma media.
    *   Rango efectivo (dimensiones activas).
    *   Similitud coseno entre `pred` y `target`.
*   **Éxito:** La pérdida baja, la varianza no decae a cero, los embeddings no son idénticos.

## 8. Plan de Fine-Tuning (Smoke Test)

*   **Script:** `10_finetune_jepa_encoder_smoke.py`
*   **Metodología:** Congelar (o usar learning rate muy bajo) el encoder preentrenado, añadir una cabeza lineal/MLP encima, y entrenar supervisadamente con las etiquetas `labels` y usando máscara `reliable_mask`.
*   **Comparativa:** Se comparará este modelo preentrenado contra las métricas del Baseline MLP y Baseline MLP + TW obtenidas en la Fase 2.
*   **Éxito:** Macro-F1 competitivo, demostrando que la representación latente autosupervisada JEPA capta semántica geométrica útil del LiDAR sin haber visto etiquetas.
