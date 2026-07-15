import json
import csv
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]
    prepare_path = repo_root / "data" / "processed" / "galicia_blocks" / "_prepare_complete.json"
    manifest_path = repo_root / "protocols" / "galicia_geographic_split_v1.json"
    
    if not prepare_path.exists():
        print("No prepare_complete found.")
        return
    
    with prepare_path.open(encoding="utf-8") as f:
        prepare_data = json.load(f)
        
    with manifest_path.open(encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    old_stats = {t["tile_id"]: t for t in prepare_data.get("tile_stats", [])}
    new_tiles = {t["tile_id"]: t for t in manifest_data.get("tiles", [])}
    
    matrix = []
    
    # Check all existing tiles
    for tile_id, old_stat in old_stats.items():
        old_split = old_stat.get("split", "unknown")
        # In the previous generation, train was balanced, val/test uniform.
        # Actually, let's look at split_mode or assumptions. 
        # The prompt says: "train generado con sampling balanceado; validation/test generados con sampling uniforme y label-blind."
        old_policy = "balanced" if old_split == "train" else "uniform"
        
        new_tile = new_tiles.get(tile_id)
        if not new_tile:
            # Maybe it wasn't processed in new manifest
            new_split = "unknown"
            new_policy = "unknown"
        else:
            new_split = new_tile.get("split")
            new_policy = "balanced" if new_split == "train" else "uniform"
            
        block_count = old_stat.get("written_blocks", 0)
        
        # Decide action
        if new_split in ["excluded_source_error"]:
            action = "exclude_source_error"
            compat = "no"
        elif new_split not in ["train", "val", "test"]:
            action = "exclude_buffer" if new_split != "unknown" else "invalid_unknown_provenance"
            compat = "no"
        elif old_split == new_split and old_policy == new_policy:
            action = "reuse_verified"
            compat = "yes"
        elif old_policy == new_policy:
            action = "relink_verified"
            compat = "yes"
        else:
            action = f"regenerate_{new_policy}"
            compat = "no"
            
        matrix.append({
            "tile_id": tile_id,
            "old_split": old_split,
            "old_policy": old_policy,
            "new_split": new_split,
            "new_policy": new_policy,
            "block_count": block_count,
            "compatibility": compat,
            "action": action
        })
        
    out_csv = repo_root / "protocols" / "block_compatibility_matrix.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=matrix[0].keys())
        writer.writeheader()
        writer.writerows(matrix)
        
    print(f"Matrix written to {out_csv}. Total tiles: {len(matrix)}")
    
    actions_count = {}
    for r in matrix:
        actions_count[r["action"]] = actions_count.get(r["action"], 0) + 1
        
    print(json.dumps(actions_count, indent=2))

if __name__ == "__main__":
    main()
