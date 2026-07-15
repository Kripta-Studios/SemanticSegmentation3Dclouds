# Estado de recuperación

Fecha: 2026-07-13.

| Área | Estado | Evidencia |
|---|---|---|
| Correcciones PNOA/DINO/mIoU | Implementadas | Tests nominales y de hash |
| Holdout oficial | Gate listo, no materializado | Default `label_blind_hash`; test de invariancia |
| Entorno local | Listo para tests | Python 3.14.2, Rasterio 1.5.0 |
| Suite completa | Verde con un skip de datos | 53 passed, 1 skipped tras gates finales |
| Raw Galicia | Disponible | 594 ficheros, 17.159.019.393 bytes |
| Raw multirregión | Disponible | 763 ficheros, 39.493.520.843 bytes |
| Bloques procesados | Bloqueado/ausente | `data/processed/galicia_blocks` no existe |
| Cachés DINO antiguas | No encontradas | No se borró nada |
| Pesos DINO reales | No encontrados localmente | Regeneración DINO bloqueada |
| Entrenamientos 3 seeds | Ejecutándose | Implementado Early Stopping (patience=3) y fix numérico (LR=1e-4, fp16) |

El único skip es `tests/test_prepare_tiles.py:12`: no hay `.pt` procesados y el
test indica ejecutar `01_prepare_tiles.py`. Es un skip heredado condicionado a
datos, no una incompatibilidad de código ni de Rasterio.

La suite no prueba componentes opcionales que no se importan en estos tests.
`tensorflow==2.12.0` y `pdal==2.4.2` del requirements histórico no son una
combinación soportada en Python 3.14; Open3D tampoco se instaló en `.venv`.
Esto debe resolverse con un entorno legacy separado si se ejecutan esas ramas,
sin degradar ni ocultar la suite PyTorch/Rasterio ya verde.
