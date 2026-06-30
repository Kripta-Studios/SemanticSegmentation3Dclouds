# Walkthrough Fase 4: Ablación TW-JEPA

## Objetivo
El objetivo de la Fase 4 fue evaluar la integración de las características geométricas locales (Taubin-Weingarten Lite) en la arquitectura autosupervisada JEPA, comparando la variante pura, inyección directa como entrada, y el uso como objetivo auxiliar geométrico.

## Variantes Evaluadas
- **Variante A (JEPA Puro)**: Encoder base de 8 canales.
- **Variante B (JEPA + TW Input)**: Encoder de 20 canales (8 features base + 12 TW normalizadas).
- **Variante C (TW-JEPA Auxiliar)**: Encoder de 8 canales con `tw_head` que pronostica el promedio de las TW features del target (usando `SmoothL1Loss` y control de no-gradiente para targets con pocos puntos).

## Resultados del Smoke Test (Linear Probing)

```csv
Variant,               Macro_F1
A (JEPA Puro),         0.1924
B (JEPA + TW Input),   0.1896
C (TW-JEPA Aux),       0.1871
```

## Conclusión

Phase 4 validates stability and implementation correctness, but does not yet demonstrate performance gains over pure GeoPoint-JEPA in short smoke-test probing.

Los resultados numéricos en este régimen corto son prácticamente equivalentes. Sin embargo, logramos un hito fundamental en ingeniería:
1. GeoPoint-JEPA funciona sin colapso (SIGReg mantiene la desviación estándar > 1.1).
2. TW como input (Variante B) se integra limpiamente sin romper SIGReg.
3. TW como target auxiliar (Variante C) no introduce inestabilidades. La pérdida decrece progresivamente mediante el *gamma scheduling* y maneja correctamente la ausencia de puntos (skipping empty targets).

La estabilidad conseguida nos proporciona una base sólida para testear estas representaciones pre-entrenadas sobre un backbone real de segmentación (Fase 5).
