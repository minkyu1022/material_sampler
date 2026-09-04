import runpy
from pathlib import Path

import torch


def test_loads_migrated_checkpoint_model():
    checkpoint = Path("outputs/cuni_corrected_diff002_tref750/checkpoint_tref750_recentered.pt")
    if not checkpoint.exists():
        return
    load_checkpoint = runpy.run_path("scripts/evaluate_cuni.py")["load_checkpoint"]
    model, config = load_checkpoint(checkpoint, torch.device("cpu"))
    assert config.n_atoms == 108
    assert config.diffusion_temperature_ref == 750.0
    assert not model.training
