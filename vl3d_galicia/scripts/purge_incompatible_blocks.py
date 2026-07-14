import csv
from pathlib import Path
import glob
import os

def main():
    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = repo_root / "protocols" / "block_compatibility_matrix.csv"
    blocks_dir = repo_root / "data" / "processed" / "galicia_blocks"
    
    with matrix_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        matrix = list(reader)
        
    tiles_to_purge = [row["tile_id"] for row in matrix if row["compatibility"] != "yes"]
    
    print(f"Purging {len(tiles_to_purge)} incompatible tiles...")
    
    purged_blocks = 0
    for tile_id in tiles_to_purge:
        # Remove done marker
        marker = blocks_dir / "_tile_done" / f"{tile_id}.json"
        if marker.exists():
            marker.unlink()
            
        # Remove blocks
        for split in ["train", "val", "test"]:
            pattern = str(blocks_dir / split / f"{tile_id}_block_*.pt")
            for block_path in glob.glob(pattern):
                os.remove(block_path)
                purged_blocks += 1
                
    print(f"Purged {purged_blocks} incompatible block files.")
    
if __name__ == "__main__":
    main()
