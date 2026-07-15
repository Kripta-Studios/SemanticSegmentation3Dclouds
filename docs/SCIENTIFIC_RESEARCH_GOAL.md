# Scientific Research Goal

Este documento resume el objetivo completo del proyecto para garantizar que la continuidad de la investigación no dependa de contextos efímeros o attachments temporales.

## Objetivo Principal
El objetivo consiste en realizar una evaluación científica rigurosa y desarrollar un pipeline de segmentación semántica 3D sobre nubes de puntos geográficas, orientado a mejorar o validar el estado del arte actual. 

En concreto, se busca:
1. **Recuperar científicamente VL3D**, construyendo un entorno auditable, reproducible y robusto.
2. **Reproducir de la forma más fiel posible el artículo:** *"Deep Learning for Ultra-Large-Scale Semantic Segmentation of Geographic 3D Point Clouds With Missing Labels"* (IEEE document: 11311458).
3. **Establecer un baseline válido y comparable** frente al mencionado estudio.
4. **Implementar un modelo Point-JEPA real** adaptado a datos ALS (Airborne Laser Scanning), basado en parches 3D.
5. **Integrar DINOv3 (denso y co-registrado)** para utilizar características visuales avanzadas.
6. **Comparar sistemáticamente** Point-JEPA, DINOv3 y su fusión (mediante concatenación, gating o cross-attention).
7. **Evaluar la generalización geográfica** dentro del dominio base (Galicia), estableciendo un particionado label-blind riguroso.
8. **Evaluar el cambio de dominio (OOD)** empleando nubes de puntos de diferentes comunidades autónomas (CCAA), previo estudio de características (sensores, campañas, CRS, canales).
9. **Determinar de forma honesta si el sistema supera al artículo original**, requiriendo baselines justos, varias seeds (e.g., 7, 13, 21), holdouts geográficos estrictos, intervalos de confianza y demostración de ausencia de leakage.
10. **Producir un paper científico final** en formato `.tex` y `.pdf` documentando todo el proceso, hallazgos, ablations y resultados (incluyendo resultados negativos).

## Restricciones y Protocolos
- **Protección del Test:** No se evaluará el conjunto de test (ej. `GAL-E-2016`) repetidamente. Se mantendrá bloqueado hasta congelar la arquitectura, hiperparámetros, seeds y ablaciones.
- **Leakage y Criterios Geográficos:** El split de datos (`train`, `val`, `test`) se hará de forma estrictamente geográfica y label-blind, garantizando un margen espacial (buffer) entre conjuntos para evitar fugas de información.
- **Métricas:** Evaluación con 6 logits + ignorado explícito (ignore-aware metrics). Reporte extenso de mIoU, F1, accuracy, entropía, calibración (ECE), eficiencia de etiquetas, tiempos y consumos (RAM/VRAM/Parámetros).
- **Baselines:** Si no se dispone de baselines reproducidos válidos o no hay consistencia en semillas y holdout, **no se declarará** que el artículo ha sido superado.
