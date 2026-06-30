from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch

from src.data.jepa_dataset import block_features
from src.models.segmentation.heads import PointSegmentationNet


CLASS_COLORS = {
    0: (139, 118, 85),
    1: (144, 238, 144),
    2: (34, 139, 34),
    3: (0, 80, 0),
    4: (190, 190, 190),
    5: (40, 120, 220),
    6: (0, 0, 0),
}


def read_run_config(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "run_config.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_segmentation_model(
    checkpoint: str | Path,
    data_root: str | Path,
    split: str = "test",
    run_config: str | Path | None = None,
    device: str | None = None,
):
    checkpoint = Path(checkpoint)
    config = read_run_config(run_config or checkpoint.parent)
    use_tw = bool(config.get("use_tw_input", False))
    probe_type = config.get("probe_type", "mlp")
    split_dir = Path(data_root) / split
    if config.get("in_channels"):
        in_channels = int(config["in_channels"])
    else:
        first = next(iter(sorted(split_dir.glob("*.pt"))), None)
        if first is None:
            raise FileNotFoundError(f"No .pt blocks in {split_dir}")
        sample = torch.load(first, weights_only=False, map_location="cpu")
        in_channels = int(block_features(sample, use_tw_input=use_tw).shape[1])
    model = PointSegmentationNet(in_channels=in_channels, probe_type=probe_type)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    model.eval()
    return model, {"device": dev, "use_tw_input": use_tw, "probe_type": probe_type, "in_channels": in_channels, **config}


@torch.no_grad()
def predict_block(model, data: dict, use_tw_input: bool, device) -> np.ndarray:
    x = block_features(data, use_tw_input=use_tw_input).float().unsqueeze(0).to(device)
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
    logits = model(x, mask)
    return logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(np.int64)


def block_files(data_root: str | Path, split: str = "test", max_blocks: int = 0) -> list[Path]:
    files = [Path(path) for path in sorted(glob.glob(str(Path(data_root) / split / "*.pt")))]
    return files[:max_blocks] if max_blocks and max_blocks > 0 else files


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    rgb = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[labels == cls] = color
    return rgb


def error_rgb(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = target != 6
    err = valid & (pred != target)
    rgb = np.zeros((target.shape[0], 3), dtype=np.uint8)
    rgb[valid & ~err] = (210, 210, 210)
    rgb[err] = (220, 30, 30)
    return rgb
