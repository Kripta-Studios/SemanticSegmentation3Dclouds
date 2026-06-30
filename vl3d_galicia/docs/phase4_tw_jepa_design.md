# Fase 4: TW-JEPA Design

Este documento especifica el diseño técnico para la Fase 4 del proyecto, donde evaluamos la integración de las características geométricas locales (Taubin-Weingarten Lite) en la arquitectura autosupervisada JEPA.

## Objetivo
Evaluar rigurosamente si y cómo las características estructurales explícitas (TW) mejoran el aprendizaje autosupervisado de representaciones de nubes de puntos 3D.

## Variantes a Evaluar

### Variante A: GeoPoint-JEPA Puro (Baseline Fase 3)
- **Input**: `[x, y, z, intensity, r, g, b, nir]` (8 canales).
- **Arquitectura**: `PointMLPEncoder` + `JepaPredictor`.
- **Loss**: $L = L_{\text{JEPA}} + \lambda L_{\text{SIGReg}}$
- **Propósito**: Línea base para medir mejoras.

### Variante B: GeoPoint-JEPA + TW Input
- **Input**: `[x, y, z, intensity, r, g, b, nir, tw_{1..12}]` (20 canales).
- **Arquitectura**: Igual que Variante A, pero el `PointMLPEncoder` recibe 20 canales en su capa inicial.
- **Loss**: $L = L_{\text{JEPA}} + \lambda L_{\text{SIGReg}}$
- **Propósito**: Comprobar si inyectar características de curvatura directamente en la entrada ayuda a aprender un mejor embedding global del bloque.

### Variante C: TW-JEPA (Pérdida Auxiliar)
- **Input Principal**: `[x, y, z, intensity, r, g, b, nir]` (8 canales).
- **Target Auxiliar**: Características TW de la región target.
- **Arquitectura**: 
  - `encoder`: Igual que Variante A (8 canales).
  - `predictor`: Predice `pred_emb` (embedding del target).
  - `tw_head`: Un pequeño proyector (MLP) acoplado a `pred_emb` que pronostica las propiedades geométricas del target.
- **Loss**: $L = L_{\text{JEPA}} + \lambda L_{\text{SIGReg}} + \gamma L_{\text{TW}}$
- **Formulación del Target**: 
  - Dado que `pred_emb` es un vector global `[B, D]` para el target, `tw_head` proyectará `[B, D] \rightarrow [B, 12]`.
  - El "true_tw_target" será el **promedio** (o max-pooling) de las características TW de los puntos válidos del vecindario target. Esto enseña a la red a predecir la "curvatura media / rugosidad global" de la zona que no está viendo.
- **Propósito**: Obligar al codificador a abstraer la geometría topológica 3D dentro de sus embeddings sin tener que calcular TW en tiempo de inferencia.

## Refactorización de Código
- **Dataset (`jepa_dataset.py`)**: 
  - Añadir soporte condicional para retornar `x_context` con 20 canales si `use_tw_input=True`.
  - Retornar también `tw_target_global` (el promedio de las TW features del target validado) para la Variante C.
- **Modelos (`geopoint_jepa.py` / `tw_jepa.py`)**:
  - `GeoPointJEPA` soportará `in_channels` variable.
  - Crearemos `TW_JEPA` que hereda/envuelve a `GeoPointJEPA` agregándole la `tw_head`.
- **Pérdidas (`tw_aux_loss.py`)**:
  - Una clase `TWAuxLoss` que compute el MSE entre `predicted_tw` y `true_tw_global`.

## Pipeline de Evaluación
1. **Pre-training**: Entrenar las 3 variantes en `train_smoke` por épocas cortas (e.g., 20).
2. **Linear Probing**: Congelar los encoders de las 3 variantes y acoplar un Linear Head de segmentación.
3. **Comparación**: Medir el Macro-F1 y F1 por clase (especialmente vegetaciones) en `val_smoke`. Registrar los MSE y las varianzas en un CSV de ablación.
