import json
import numpy as np
import pandas as pd
from pathlib import Path
import re
import collections

ROOT = Path("C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/SemanticSegmentation3Dclouds/vl3d_galicia")
OUTPUTS = ROOT / "outputs" / "scientific_candidate_v1"

def check_six_logit(exp_dir: Path, metrics: dict, config: dict) -> str:
    # 1. Check config for num_classes=6
    if config.get("num_classes") == 7:
        return "legacy_seven_logit_invalid"
    
    # 2. Check confusion matrix shape (should be 6x7, 6 rows (true), 7 cols (pred))
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        if len(cm) != 6:
            return "legacy_seven_logit_invalid"
        if len(cm[0]) != 7:
            return "legacy_seven_logit_invalid"
            
    # 3. Check protocol version
    protocol = metrics.get("metric_protocol_version", "")
    if protocol != "segmentation-metrics-v2-pred-ignore-is-fn":
        return "legacy_seven_logit_invalid"
        
    return "verified_six_logit"

def load_all_runs():
    runs = []
    for exp_dir in OUTPUTS.iterdir():
        if not exp_dir.is_dir(): continue
        
        metrics_path = exp_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue
            
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        config_path = exp_dir / "run_config.json"
        if not config_path.exists():
            config_path = exp_dir / "config.json"
            
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            
        val_path = exp_dir / "val_metrics.json"
        val_metrics = {}
        if val_path.exists():
            val_metrics = json.loads(val_path.read_text(encoding="utf-8"))
            
        name = exp_dir.name
        match = re.search(r'seed(\d+)$', name)
        seed = int(match.group(1)) if match else None
        
        family = name.replace(f"_seed{seed}", "") if seed is not None else name
        
        provenance = check_six_logit(exp_dir, metrics, config)
        
        # Read gating stats if M3
        gate_mean = config.get("gate_mean", None) # If they saved it in config, or we need to extract from logs?
        # The prompt says: "Si no se guardaron esos valores, no inventes una interpretacion... registra la carencia".
        
        runs.append({
            "dir": exp_dir,
            "name": name,
            "family": family,
            "seed": seed,
            "provenance": provenance,
            "metrics": metrics,
            "config": config,
            "val_metrics": val_metrics
        })
    return runs

def compute_aggregates(runs):
    # Filter only verified six logit
    valid_runs = [r for r in runs if r["provenance"] == "verified_six_logit"]
    
    # Group by family
    families = collections.defaultdict(list)
    for r in valid_runs:
        families[r["family"]].append(r)
        
    # Baseline (M0) by seed
    m0_runs = {r["seed"]: r for r in families.get("M0_geometric", [])}
    
    results = []
    
    for family, fruns in families.items():
        metrics_to_agg = ["OA", "macro_f1", "mIoU", "balanced_accuracy"]
        family_agg = {"family": family}
        
        # Means and stds
        for m in metrics_to_agg:
            vals = [r["metrics"].get(m, 0.0) for r in fruns]
            family_agg[f"{m}_mean"] = np.mean(vals) if vals else 0.0
            family_agg[f"{m}_std"] = np.std(vals) if len(vals)>1 else 0.0
            family_agg[f"{m}_min"] = np.min(vals) if vals else 0.0
            family_agg[f"{m}_max"] = np.max(vals) if vals else 0.0
            
        # Paired differences vs M0
        deltas = {"OA": [], "macro_f1": [], "mIoU": []}
        for r in fruns:
            seed = r["seed"]
            if seed in m0_runs and family != "M0_geometric":
                m0 = m0_runs[seed]
                deltas["OA"].append(r["metrics"].get("OA", 0) - m0["metrics"].get("OA", 0))
                deltas["macro_f1"].append(r["metrics"].get("macro_f1", 0) - m0["metrics"].get("macro_f1", 0))
                deltas["mIoU"].append(r["metrics"].get("mIoU", 0) - m0["metrics"].get("mIoU", 0))
        
        for m in deltas:
            if deltas[m]:
                family_agg[f"delta_{m}_mean"] = np.mean(deltas[m])
                family_agg[f"delta_{m}_std"] = np.std(deltas[m]) if len(deltas[m])>1 else 0.0
            else:
                family_agg[f"delta_{m}_mean"] = 0.0
                family_agg[f"delta_{m}_std"] = 0.0
                
        results.append(family_agg)
        
    return results, runs

def main():
    runs = load_all_runs()
    aggs, all_runs = compute_aggregates(runs)
    
    # Save individual runs table
    rows = []
    for r in all_runs:
        m = r["metrics"]
        c = r["config"]
        vm = r["val_metrics"]
        
        row = {
            "name": r["name"],
            "family": r["family"],
            "seed": r["seed"],
            "provenance": r["provenance"],
            "OA": m.get("OA", 0),
            "macro_F1": m.get("macro_f1", 0),
            "mIoU": m.get("mIoU", 0),
            "balanced_accuracy": m.get("balanced_accuracy", 0),
            "coverage": m.get("coverage", 0),
            "ignored_prediction_rate": m.get("ignored_prediction_rate", 0),
            "evaluated_points": m.get("evaluated_points", 0),
            "tiles_evaluated": m.get("tiles_evaluated", 0),
            "best_epoch": vm.get("epoch", 0), # if we have it
            "val_score": vm.get("macro_f1", 0),
            "trainable_params": c.get("trainable_params", 0),
            "duration": c.get("duration", "unknown"),
            "VRAM": c.get("vram", "unknown"),
            "gate_stats": "Not recorded" if "gated" in r["family"] else "N/A"
        }
        # Class IoUs
        ciou = m.get("class_iou", {})
        for cid, val in ciou.items():
            row[f"class_{cid}_iou"] = val
            
        rows.append(row)
        
    pd.DataFrame(rows).to_csv(OUTPUTS / "individual_runs_provenance.csv", index=False)
    
    # Save aggs table
    pd.DataFrame(aggs).to_csv(OUTPUTS / "family_aggregates.csv", index=False)
    
    print(f"Processed {len(runs)} runs.")

if __name__ == '__main__':
    main()
