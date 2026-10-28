"""Every embedded crop must match the same self-contained source field."""

import importlib.util
from pathlib import Path

import numpy as np

from sparse_idcs.embedded import decode_binary


ROOT = Path(__file__).resolve().parents[1]


def load_experiment(name):
    path = ROOT / "experiments" / name / "run.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_embedded_inputs_match_full_source():
    geom = load_experiment("reference_geometry")
    full = decode_binary(geom.FULL_PAYLOAD, (2500, 2500), geom.FULL_SHA256)
    assert full.shape == (2500, 2500)
    crops = {
        "ablation_50x100": full[1700:1800, 2194:2244],
        "penetration_sweep": full[1700:1800, 2194:2394],
        "large_seed_stability": full[1700:2000, 2194:2344],
        "forward_data": full[1700:1800, 2194:2244],
        "likelihood_cost": full[1700:1800, 2194:2244],
    }
    for name, reference in crops.items():
        module = load_experiment(name)
        ti = decode_binary(module.TI_PAYLOAD, (500, 500), module.TI_SHA256)
        truth = decode_binary(module.TRUTH_PAYLOAD, reference.shape, module.TRUTH_SHA256)
        assert np.array_equal(ti, full[:500, :500])
        assert np.array_equal(truth, reference)


def test_bitpacked_decoder_rejects_changed_checksum():
    module = load_experiment("ablation_50x100")
    try:
        decode_binary(module.TI_PAYLOAD, (500, 500), "0" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed checksum was accepted")
