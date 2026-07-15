import argparse
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
from scipy import stats

def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def read_config(exp_dir: Path) -> dict:
    path = exp_dir / "run_config.json"
    if not path.exists():
        path = exp_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def extract_experiments(root: Path):
    runs = []
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_metrics(metrics_path)
        cfg = read_config(exp_dir)
        seed = int(cfg.get("seed", 42))
        
        # Heuristics for family
        name = cfg.get("model_name", exp_dir.name)
        if "baseline" in name.lower() or "m0" in name.lower():
            family = "M0"
        elif "jepa" in name.lower() and "dino" in name.lower():
            family = "M4" if "v3" in name.lower() else "M3"
        elif "jepa" in name.lower():
            family = "M5" # Or PointJEPA only
        elif "dino" in name.lower():
            family = "M2" if "v3" in name.lower() else "M1"
        else:
            family = "Other"
            
        runs.append({
            "name": name,
            "family": family,
            "seed": seed,
            "metrics": metrics,
            "path": exp_dir
        })
    return runs

def compute_stats(values):
    if not values:
        return 0, 0, 0, 0
    arr = np.array(values)
    return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)

def compute_paired_ci(differences, confidence=0.95):
    if len(differences) < 2:
        return 0, 0
    arr = np.array(differences)
    mean = np.mean(arr)
    sem = stats.sem(arr)
    # Using t-distribution
    margin = sem * stats.t.ppf((1 + confidence) / 2., len(arr)-1)
    return mean - margin, mean + margin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=str, default="outputs")
    parser.add_argument("--out-md", type=str, default="outputs/multiseed_comparison.md")
    args = parser.parse_args()

    root = Path(args.experiments_root)
    runs = extract_experiments(root)
    
    # Group by family and seed
    grouped = defaultdict(dict)
    for run in runs:
        grouped[run["family"]][run["seed"]] = run
        
    m0_runs = grouped.get("M0", {})
    
    families = sorted(grouped.keys())
    
    md_lines = ["# Multiseed Paired Comparison against M0", ""]
    
    for family in families:
        md_lines.append(f"## Family: {family}")
        if not grouped[family]:
            continue
            
        mious = []
        macro_f1s = []
        d_mious = []
        d_f1s = []
        
        # Per class metrics
        class_ious = defaultdict(list)
        class_f1s = defaultdict(list)
        
        d_class_ious = defaultdict(list)
        d_class_f1s = defaultdict(list)
        
        for seed, run in grouped[family].items():
            metrics = run["metrics"]
            miou = metrics.get("macro_iou", 0.0)
            f1 = metrics.get("macro_f1", 0.0)
            mious.append(miou)
            macro_f1s.append(f1)
            
            c_iou = metrics.get("class_iou", {})
            c_f1 = metrics.get("class_f1", {})
            for c, val in c_iou.items():
                class_ious[c].append(val)
            for c, val in c_f1.items():
                class_f1s[c].append(val)
            
            # Paired difference vs M0
            if seed in m0_runs:
                m0_metrics = m0_runs[seed]["metrics"]
                m0_miou = m0_metrics.get("macro_iou", 0.0)
                m0_f1 = m0_metrics.get("macro_f1", 0.0)
                d_mious.append(miou - m0_miou)
                d_f1s.append(f1 - m0_f1)
                
                m0_c_iou = m0_metrics.get("class_iou", {})
                m0_c_f1 = m0_metrics.get("class_f1", {})
                for c, val in c_iou.items():
                    if c in m0_c_iou:
                        d_class_ious[c].append(val - m0_c_iou[c])
                for c, val in c_f1.items():
                    if c in m0_c_f1:
                        d_class_f1s[c].append(val - m0_c_f1[c])
                
        # Compute stats
        miou_mean, miou_std, miou_min, miou_max = compute_stats(mious)
        f1_mean, f1_std, f1_min, f1_max = compute_stats(macro_f1s)
        d_miou_mean, d_miou_std, _, _ = compute_stats(d_mious)
        d_f1_mean, d_f1_std, _, _ = compute_stats(d_f1s)
        
        md_lines.append(f"- **Seeds run:** {list(grouped[family].keys())}")
        md_lines.append(f"- **mIoU:** {miou_mean:.4f} ± {miou_std:.4f} (min: {miou_min:.4f}, max: {miou_max:.4f})")
        md_lines.append(f"- **macro-F1:** {f1_mean:.4f} ± {f1_std:.4f} (min: {f1_min:.4f}, max: {f1_max:.4f})")
        if d_mious:
            ci_low, ci_high = compute_paired_ci(d_mious)
            md_lines.append(f"- **Paired Diff vs M0 mIoU:** {d_miou_mean:+.4f} ± {d_miou_std:.4f} (95% CI: [{ci_low:+.4f}, {ci_high:+.4f}])")
            ci_low, ci_high = compute_paired_ci(d_f1s)
            md_lines.append(f"- **Paired Diff vs M0 macro-F1:** {d_f1_mean:+.4f} ± {d_f1_std:.4f} (95% CI: [{ci_low:+.4f}, {ci_high:+.4f}])")
        md_lines.append("")
        
        # Per class results
        md_lines.append("### Metrics by Class")
        md_lines.append("| Class | mIoU | mIoU Diff vs M0 | F1 | F1 Diff vs M0 |")
        md_lines.append("|-------|------|-----------------|----|---------------|")
        
        for c in sorted(class_ious.keys(), key=lambda x: int(x)):
            c_iou_m, c_iou_s, _, _ = compute_stats(class_ious[c])
            c_f1_m, c_f1_s, _, _ = compute_stats(class_f1s[c])
            
            diff_iou_str = "-"
            diff_f1_str = "-"
            if c in d_class_ious and d_class_ious[c]:
                cd_iou_m, cd_iou_s, _, _ = compute_stats(d_class_ious[c])
                diff_iou_str = f"{cd_iou_m:+.4f} ± {cd_iou_s:.4f}"
            if c in d_class_f1s and d_class_f1s[c]:
                cd_f1_m, cd_f1_s, _, _ = compute_stats(d_class_f1s[c])
                diff_f1_str = f"{cd_f1_m:+.4f} ± {cd_f1_s:.4f}"
                
            md_lines.append(f"| {c} | {c_iou_m:.4f} ± {c_iou_s:.4f} | {diff_iou_str} | {c_f1_m:.4f} ± {c_f1_s:.4f} | {diff_f1_str} |")
        
        md_lines.append("")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved multiseed comparison to {args.out_md}")

if __name__ == "__main__":
    main()
