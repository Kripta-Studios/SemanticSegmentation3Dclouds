# Cleanup Inventory

Este documento clasifica los archivos y artefactos presentes en el repositorio antes de realizar cualquier borrado, para garantizar que no se pierda información valiosa.

## 1. Datasets y Datos Crudos (Generados / No versionables)
- **`data/raw/pnoa_galicia/`**: **Necesarios (Raw Data)**. Archivos LAZ descargados del PNOA. No deben borrarse bajo ningún concepto.
- **`data/raw/pnoa_varias_ccaa/`**: **Necesarios (Raw Data)**. Archivos de evaluación OOD. No borrar.
- **`data/processed/galicia_blocks/`**: **Históricos / Generados**. Contiene los 449.120 bloques existentes. Se someterán a purga selectiva (solo se borrarán los incompatibles con el nuevo split geográfico).
- **`test_data/shadow_batches/`** y **`test_data/*.laz`**: **Necesarios (Tests)**. Útiles para pruebas unitarias de loader y modelos.

## 2. Scripts y Código (Necesarios vs Históricos)
- **`scripts/00_*` a `scripts/28_*`**: **Necesarios**. Forman el pipeline principal del proyecto, desde la descarga hasta el split geográfico, entrenamiento y evaluación.
- **`scripts/11_run_geo_jepa_pilot.py`**: **Histórico/Obsoleto**. Posible candidato a borrado si el nuevo Point-JEPA lo reemplaza.
- **`scripts/cesga_*.sh` y `scripts/sql_insert_*.py`**: **Ajenos/Históricos**. Parecen scripts de integración específicos del clúster CESGA o bases de datos de una versión anterior. Se documentarán y se moverán a una carpeta `legacy/` o se eliminarán tras verificación.
- **`src/`**: **Necesario**. Código fuente principal (modelos, datasets, características).
- **`tests/`**: **Necesario**. Suite de pruebas.
- **`vl3d.py` y `vl3d_*.yml`**: **Necesario / Histórico**. Entrada CLI antigua y entornos Conda.

## 3. Artefactos y Logs (Generados)
- **`artifacts/logs/`**: **Históricos / Generados**. Logs de ejecuciones pasadas (ej. `scientific_candidate_v1_training...`).
- **`reports/`**: **Generados**. Salidas JSON de preparaciones anteriores y métricas obsoletas.

## Justificación
No se borrará nada hasta completar la auditoría de los bloques y generar el nuevo split. Los scripts ajenos (CESGA) se retendrán hasta que se confirme que no contienen configuraciones críticas de despliegue. Los bloques generados incompatibles serán eliminados selectivamente para ahorrar espacio, respaldados por la matriz de decisión de compatibilidad.
