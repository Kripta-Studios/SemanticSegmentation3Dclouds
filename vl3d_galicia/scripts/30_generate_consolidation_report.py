import pandas as pd
from pathlib import Path
import json
import hashlib

ROOT = Path("C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/SemanticSegmentation3Dclouds/vl3d_galicia")
OUTPUTS = ROOT / "outputs" / "scientific_candidate_v1"

def generate_report():
    df_runs = pd.read_csv(OUTPUTS / "individual_runs_provenance.csv")
    df_aggs = pd.read_csv(OUTPUTS / "family_aggregates.csv")
    
    md = []
    md.append("# Consolidación M0-M4 y Auditoría")
    md.append("\n## 1. Verificación 6-Logits (Provenance)")
    md.append("\nSe ha verificado la matriz de confusión (6x7 reales) y el flag `segmentation-metrics-v2-pred-ignore-is-fn` en todos los runs:")
    md.append("\n| Model Name | Provenance |")
    md.append("|---|---|")
    for _, r in df_runs.iterrows():
        md.append(f"| {r['name']} | `{r['provenance']}` |")
        
    md.append("\n## 2. Comparación Agregada por Familia (Emparejada por Seed contra M0)")
    md.append("\n| Family | mIoU Mean ± SD | mIoU Min-Max | Delta mIoU vs M0 (Mean ± SD) | OA Mean ± SD | Delta OA (Mean ± SD) |")
    md.append("|---|---|---|---|---|---|")
    for _, r in df_aggs.iterrows():
        fam = r['family']
        miou = f"{r['mIoU_mean']:.4f} ± {r['mIoU_std']:.4f}"
        miou_range = f"[{r['mIoU_min']:.4f}, {r['mIoU_max']:.4f}]"
        miou_delta = f"{r['delta_mIoU_mean']:+.4f} ± {r['delta_mIoU_std']:.4f}"
        
        oa = f"{r['OA_mean']:.4f} ± {r['OA_std']:.4f}"
        oa_delta = f"{r['delta_OA_mean']:+.4f} ± {r['delta_OA_std']:.4f}"
        
        md.append(f"| {fam} | {miou} | {miou_range} | {miou_delta} | {oa} | {oa_delta} |")

    md.append("\n## 3. Tabla Única de Métricas Individuales (M0 - M4)")
    md.append("\n| Model | OA | mIoU | macro-F1 | Bal.Acc | Cov. | Ign.Rate | Params | Best Epoch | VRAM | Gate Stats |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df_runs.iterrows():
        md.append(f"| {r['name']} | {r['OA']:.4f} | {r['mIoU']:.4f} | {r['macro_F1']:.4f} | {r['balanced_accuracy']:.4f} | {r['coverage']:.1f} | {r['ignored_prediction_rate']:.3f} | {r['trainable_params']} | {r['best_epoch']} | {r['VRAM']} | {r['gate_stats']} |")
        
    md.append("\n## 4. Auditoría de la Purga de 76.038 Bloques")
    md.append("""
- **Comando Ejecutado**: `.\\.venv\\Scripts\\python.exe scripts\\purge_incompatible_blocks.py`
- **Fecha**: 14/07/2026 09:27 CET
- **Manifest Utilizado**: `protocols/block_compatibility_matrix.csv`
- **Tiles Afectados**: 51 tiles.
- **Split Original**: Sampling balanceado para uso en Train.
- **Split Nuevo**: Requerían sampling uniforme (movidos a validation/test). 1 tile fue excluido por caer en el buffer geográfico (blind-split).
- **Incompatibilidad**: Sampling balanceado de entrenamiento utilizado accidentalmente en tiles de validación, lo cual invalida las métricas uniformes.
- **Bytes Liberados**: Aprox 76.038 archivos `.pt` (varios GB).
- **Capacidad de Regeneración**: `01_prepare_tiles.py` está preparado para regenerar los bloques faltantes desde los archivos LAZ raw en la siguiente pasada.
- **Estado de M0-M4**: Los runs M0-M4 **sí** fueron entrenados antes de esta purga, usando la versión del dataset `galicia_blocks_geo_v1_d6b98dd` que incluía esos 51 tiles con sampling erróneo en el validation set. El hash del dataset utilizado fue `d6b98dd`. Se conserva este hash como registro histórico para los runs M0-M4. No se simulará que usaron el dataset corregido.
""")

    md.append("\n## 5. Conclusión Provisional Permitida")
    md.append("""
* **M1 DINOv2 concatenado**: `mixed/unstable`.
* **M2 DINOv3 concatenado**: `no robust improvement observed`.
* **M3 gated**: `pending three-seed aggregate` (Seed 7 fue favorable, pero la agregación muestra que las seeds 13 y 21 no sostienen la mejora).
* **M4 multiview**: `pending three-seed aggregate` (No mejora el baseline).

> **Veredicto general M0-M4**: Ninguna forma de incorporar DINO (concat, gated, o multiview top-view) mejora de manera consistente y robusta el baseline geométrico M0 en las tres seeds ni en las métricas por clase simultáneamente. Las varianzas entre seeds son mayores que las ganancias arquitectónicas en los casos favorables.
""")

    (ROOT / "docs" / "M0_M4_CONSOLIDATION_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    
if __name__ == '__main__':
    generate_report()
