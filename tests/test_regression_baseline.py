"""Guard: the S24 model must stay bit-identical while new features are added."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from matblast.config import load_config
from matblast.models import build_model

FIXTURE = Path(__file__).with_name("fixtures") / "s24_baseline_regression.json"


def test_s24_model_is_unchanged_by_new_features():
    fixture = json.loads(FIXTURE.read_text())
    cfg = load_config(Path(__file__).resolve().parents[1] / fixture["config"])
    cfg["model"]["base_channels"] = fixture["base_channels_override"]
    torch.manual_seed(fixture["model_seed"])
    model = build_model(cfg).eval()
    keys = sorted(model.state_dict().keys())
    assert len(keys) == fixture["n_state_dict_keys"]
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == fixture["state_dict_keys_sha256"]
    generator = torch.Generator().manual_seed(fixture["input_seed"])
    image = torch.rand(*fixture["input_shape"], generator=generator)
    with torch.no_grad():
        out = model(image)
    assert sorted(out.keys()) == fixture["output_keys"]
    logits = out["logits"].double()
    assert abs(float(logits.sum()) - fixture["logits_sum"]) < 1e-3
    assert abs(float(logits.abs().sum()) - fixture["logits_abs_sum"]) < 1e-3
    assert torch.allclose(
        logits.flatten()[:5], torch.tensor(fixture["logits_first5"], dtype=torch.float64), atol=1e-6
    )
