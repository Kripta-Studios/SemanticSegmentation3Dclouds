from __future__ import annotations

import importlib.util
import math
from pathlib import Path


def load_orchestrator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "28_run_scientific_candidate_v1.py"
    spec = importlib.util.spec_from_file_location("candidate_orchestrator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_seed_summary_uses_sample_sd_and_student_t_interval():
    module = load_orchestrator()
    summary = module.summarize([0.4, 0.5, 0.6])
    assert summary["n"] == 3
    assert math.isclose(summary["mean"], 0.5)
    assert math.isclose(summary["std"], 0.1)
    expected_half_width = module.T_CRITICAL_95[3] * 0.1 / math.sqrt(3)
    assert math.isclose(summary["ci95_low"], 0.5 - expected_half_width)
    assert math.isclose(summary["ci95_high"], 0.5 + expected_half_width)
