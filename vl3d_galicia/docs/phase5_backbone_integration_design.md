# Fase 5: Integración de Backbone de Segmentación Real

Este documento especifica el diseño para la Fase 5 del proyecto, donde evaluamos si el pre-entrenamiento de GeoPoint-JEPA y TW-JEPA (Fase 4) ofrece ganancias empíricas sobre una red de segmentación densa real que ya es intrínsecamente capaz de modelar geometría local (a diferencia del MLP punto-a-punto previo).

## Objetivo
Responder a la pregunta: ¿Mejoran las representaciones autosupervisadas JEPA y TW-JEPA el rendimiento final (F1, IoU) y la generalización (West $\rightarrow$ East) de una red de segmentación 3D frente a entrenarla desde cero?

## Experimentos y Baselines (Smoke Benchmark)
Evaluar de forma escalonada:
- **E0**: MLP Baseline (Entrenado desde cero).
- **E1**: MLP + TW-Lite (Entrenado desde cero).
- **E2**: Backbone Real (KPConv/SFL-Net lite) Baseline (Entrenado desde cero).
- **E3**: Backbone Real + TW-Lite input (Entrenado desde cero).
- **E4**: Backbone Real inicializado con el *Encoder* de GeoPoint-JEPA pre-entrenado.
- **E5**: Backbone Real inicializado con el *Encoder* de TW-JEPA pre-entrenado.

## Diseño del Backbone Real ("Lite KPConv / PointNet++")

Para mantener la compatibilidad con los tensores `[B, N, C]` actuales generados en `03_make_blocks.py` y evitar introducir dependencias masivas o refactorizaciones complejas del Dataloader (como collations de grafos de tamaño variable en `torch_geometric`), implementaremos una versión *reducida* del backbone inspirada en **PointNet++ / KPConv**:

- **Local Neighborhood Aggregation**: Uso de KNN (k-Nearest Neighbors) dentro de cada bloque para agregar características locales.
- **Arquitectura Base (`PointLocalBackbone`)**:
  1. `in_channels` (8 o 20) $\rightarrow$ MLP puntual $\rightarrow$ $F_1$.
  2. Local Aggregation 1: KNN graph ($k=16$). Para cada punto, agregar características de sus vecinos usando un pseudo-KPConv o EdgeConv simplificado $\rightarrow$ $F_2$.
  3. Local Aggregation 2: KNN graph ($k=16$) iterativo $\rightarrow$ $F_3$.
  4. Global Context: Max-pooling global + expansión $\rightarrow$ $F_{global}$.
  5. Concatenación: $[F_1, F_2, F_3, F_{global}]$ $\rightarrow$ Segmentation Head $\rightarrow$ `out_classes=7`.

## Integración con JEPA (E4 y E5)
En los experimentos E4 y E5, el objetivo es utilizar el Encoder preentrenado.
- **Opción A (Feature Extraction / JEPA as Frontend)**: El backbone real toma las características emitidas por el Encoder de JEPA y luego aplica su propia agregación espacial. Es decir: `features = JEPA_Encoder(x)`, luego `Backbone(features, x)`.
- **Opción B (JEPA as Backbone)**: Si el Encoder de JEPA ya fuera el backbone jerárquico. 

*Dado que en la Fase 3 diseñamos el `PointMLPEncoder` como el encoder de JEPA (que no usa agregaciones vecinas jerárquicas sino global/local simplificado), para una evaluación justa la **Opción A** es la más modular.* La representación JEPA enriquecida entra como input semántico al backbone real.

## Métricas
El pipeline generará reportes detallados usando un `SegmentationTrainer` y `SegmentationMetrics` que registrará:
- `Macro-F1`
- `Weighted-F1`
- `IoU Macro`
- `IoU por clase` (y `F1 por clase`, en especial Low, Mid, High veg, Building, Water)
- `Matriz de Confusión`
- `MCC` y `Cohen Kappa`

## Estructura de Archivos
- `src/models/backbones/point_local_net.py`: El backbone real simplificado.
- `src/training/segmentation_trainer.py`: Motor de entrenamiento estandarizado para los 6 experimentos.
- `src/eval/segmentation_metrics.py`: Cálculo riguroso de métricas por clase.
- Scripts de ejecución `14` a `18` para orquestar cada experimento y generar la tabla de comparación final.
