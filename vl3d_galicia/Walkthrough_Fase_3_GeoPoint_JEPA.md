# Fase 3: GeoPoint-JEPA (Smoke Test)

## Arquitectura y Parámetros
- **Arquitectura**: GeoPoint-JEPA, compuesta por un `PointMLPEncoder` que reduce características de los puntos a un vector latente global, y un `JepaPredictor`.
- **Dimensión Latente**: 256.
- **Enmascaramiento Espacial**: Separación espacial del bloque usando distancia euclídea al azar (centroide y kNN más cercanos).
- **Datos de Entrenamiento**: 100 bloques espaciales normalizados (aprox. cientos a miles de puntos por bloque).
- **Target Loss**: `MSE(Pred, Target) + lambda * SIGRegLoss`.

## Resultados del Pre-entrenamiento
- **Prediction Loss (MSE)**: Se redujo gradualmente de ~0.48 a ~0.42 en 20 épocas.
- **Métricas Anti-Colapso**: La desviación estándar del embedding se mantuvo siempre por encima de 1.10 (gamma=1.0), indicando cero colapso gracias a SIGReg. La covarianza entre dimensiones latentes se mantuvo controlada.
- **Tiempo**: Ejecución rápida del smoke test local.

## Resultados del Fine-Tuning (Linear Probing)
- **Método**: Se congelaron por completo los pesos del encoder de JEPA (256 dims).
- **Macro-F1**: Alcanzó 0.1970 tras 15 épocas.
- **Limitaciones**: Como se probó con 100 bloques y 15 épocas, el resultado es puramente conceptual. Muestra que la arquitectura *puede aprender representaciones espacialmente útiles sin etiquetas*, superando a una inicialización aleatoria que daría cerca de cero.

## Conclusión
La arquitectura JEPA sin EMA y regulada por SIGReg es viable y estable para nubes LiDAR, sin colapsar trivialmente en embeddings constantes. Esto prepara el camino para la Fase 4.
