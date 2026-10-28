"""Shared execution helpers for the manuscript's synthetic experiments."""

from __future__ import annotations

import json
import math
import sys
import time
import types
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt

from . import binary
from . import incremental_solver as sparse_solver


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "idcs_original"


def configure_sparse(shape, seed, sample_mode="argmax", candidates=100, refresh_fraction=0.02):
    sparse_solver.NY, sparse_solver.NX = shape
    sparse_solver.SPACING_M = 0.1
    sparse_solver.SIGMA_D = 1.0
    sparse_solver.SNR_SPACING = 4
    sparse_solver.LIMIT_ANGLE_DEG = 50.0
    sparse_solver.RANDOM_SEED = seed
    sparse_solver.CANDIDATE_MODE = "qs_fft"
    sparse_solver.QS_CANDIDATES = candidates
    sparse_solver.QS_NEIGHBORS = 25
    sparse_solver.QS_TEMPLATE_SIZE = 31
    sparse_solver.QS_TEMPLATE_ALPHA = 0.0
    sparse_solver.SAMPLE_MODE = sample_mode
    sparse_solver.DATA_WEIGHT = 1.0
    sparse_solver.MPS_WEIGHT = 1.0
    sparse_solver.INITIAL_MODE = "prior"
    sparse_solver.PATH_MODE = "coverage_random"
    sparse_solver.HARD_MODE = "none"
    sparse_solver.LIKELIHOOD_BASELINE_MODE = "prior_mean"
    sparse_solver.MEAN_UPDATE_MODE = "point"
    sparse_solver.RESIDUAL_CORRECT_EVERY_STEPS = 0
    sparse_solver.RESIDUAL_CORRECT_EVERY_FRACTION = refresh_fraction
    sparse_solver.REFINE_PASSES = 4
    sparse_solver.REFINE_BOUNDARY_ONLY = True
    sparse_solver.REFINE_DATA_WEIGHT = 1.0
    sparse_solver.REFINE_MPS_WEIGHT = 1.0
    sparse_solver.REFINE_SMOOTH_WEIGHT = 0.0
    sparse_solver.REFINE_MIN_GAIN = 0.0


def software_versions():
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import gstools
    except ImportError:
        return versions
    versions["gstools"] = gstools.__version__
    return versions


def sparse_operator(truth, noise_mode="filtered"):
    if noise_mode not in ("filtered", "pre_filter"):
        raise ValueError("noise_mode must be filtered or pre_filter")
    a, csc, observed, clean, records, rejected, total = sparse_solver.build_sparse_linear_operator(
        binary.slowness_from_binary(truth)
    )
    if noise_mode == "pre_filter":
        full_noise = np.random.default_rng(0).normal(0.0, sparse_solver.SIGMA_D, total)
        observed = clean + full_noise[[int(rec[0]) for rec in records]]
    return a, csc, observed, clean, records, rejected, total


def run_sparse(ti, truth, seed=20260829, sample_mode="argmax", use_map=True,
               candidates=100, refresh_fraction=0.02, noise_mode="filtered"):
    if ti.shape != (500, 500):
        raise ValueError("Expected a 500 x 500 training image")
    configure_sparse(truth.shape, seed, sample_mode, candidates, refresh_fraction)
    t0 = time.perf_counter()
    a, csc, observed, clean, records, rejected, total = sparse_operator(truth, noise_mode)
    operator_seconds = time.perf_counter() - t0
    hard = np.zeros_like(truth, dtype=np.uint8)
    t0 = time.perf_counter()
    field, wrmse_trace, loglike_trace, pattern_trace, cache_size, drift = sparse_solver.simulate_sparse(
        ti, a, csc, observed, hard, hard
    )
    simulation_seconds = time.perf_counter() - t0
    pre_map = field.copy()
    map_trace = []
    map_seconds = 0.0
    if use_map:
        t0 = time.perf_counter()
        field, map_trace = sparse_solver.local_map_refine(a, csc, observed, field, ti, hard)
        map_seconds = time.perf_counter() - t0
    metrics = sparse_solver.binary_metrics(field, truth)
    metrics["wrmse"] = sparse_solver.final_wrmse(a, observed, field)
    metrics["max_residual_drift"] = float(max((row["max_abs"] for row in drift), default=0.0))
    summary = {
        "method": "Sparse-IDCS",
        "software": software_versions(),
        "parameters": {
            "shape_yx": list(truth.shape), "spacing_m": 0.1, "velocity_channel_m_ns": 0.06,
            "velocity_background_m_ns": 0.08, "sigma_d_ns": 1.0,
            "antenna_spacing_cells": 4, "angle_limit_deg": 50.0,
            "candidate_mode": "qs_fft", "candidate_count": candidates, "neighbors": 25,
            "template_size": 31, "sample_mode": sample_mode, "seed": seed,
            "data_weight": 1.0, "mps_weight": 1.0, "path_mode": "coverage_random",
            "initial_mode": "prior", "likelihood_baseline_mode": "prior_mean",
            "mean_update_mode": "point", "hard_mode": "none",
            "residual_refresh_fraction": refresh_fraction, "map_enabled": use_map,
            "map_passes": 4 if use_map else 0, "map_boundary_only": True,
            "map_data_weight": 1.0, "map_mps_weight": 1.0,
            "map_smooth_weight": 0.0, "noise_mode": noise_mode,
        },
        "operator": {"retained_rays": int(a.shape[0]), "all_rays": total,
                     "rejected_rays": rejected, "nnz": int(a.nnz)},
        "metrics": metrics,
        "timing_seconds": {
            "operator": operator_seconds, "simulation": simulation_seconds,
            "map": map_seconds, "total": operator_seconds + simulation_seconds + map_seconds,
        },
        "map_accepted_flips": int(sum(row["accepted_flips"] for row in map_trace)),
        "pattern_cache_size": cache_size,
    }
    return {
        "summary": summary, "truth": truth, "training_image": ti, "result": field,
        "pre_map": pre_map, "operator": a, "observed": observed, "clean": clean,
        "rays": records, "wrmse_trace": wrmse_trace, "loglike_trace": loglike_trace,
        "pattern_trace": pattern_trace, "drift": drift,
    }


def _original_imports():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    if "int" not in np.__dict__:
        np.int = int
    shim = types.ModuleType("set_fw")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Only the upstream linear straight-ray path is used")

    shim.set_fw = shim.pygimli_fw = shim.set_J = shim.update_J = unavailable
    sys.modules["set_fw"] = shim
    from Model_obj import Model_obj
    from fit_vario import fit_vario
    from sim import qsCat
    return Model_obj, fit_vario, qsCat


_VARIOGRAM = None


def _get_variogram(ti_slowness, fit_vario):
    global _VARIOGRAM
    if _VARIOGRAM is None:
        _VARIOGRAM = fit_vario(ti_slowness, seed=0, case="channels")
    return _VARIOGRAM


def original_forward_data(truth, noise_mode="pre_filter"):
    """Build the dense upstream geometry and its selected noise realization."""
    if noise_mode not in ("pre_filter", "filtered"):
        raise ValueError("noise_mode must be pre_filter or filtered")
    ny, nx = truth.shape
    true_slowness = binary.slowness_from_binary(truth)
    z = np.arange(2, ny, 4) * 0.1
    edges_x = np.arange(nx + 1, dtype=float) * 0.1
    edges_z = np.arange(ny + 1, dtype=float) * 0.1
    rows = np.asarray([[0.0, sz + 0.05, nx * 0.1, rz + 0.05]
                       for sz in z for rz in z], dtype=float)
    all_a = sparse_solver.tomokernel_straight_2D(rows, edges_x, edges_z).toarray()
    angles = np.asarray([math.degrees(math.atan(abs(row[3] - row[1]) / (nx * 0.1)))
                         for row in rows])
    keep = angles <= 50.0
    rng_noise = np.random.default_rng(0)
    if noise_mode == "pre_filter":
        noise = rng_noise.normal(0.0, 1.0, all_a.shape[0])[keep]
    else:
        noise = rng_noise.normal(0.0, 1.0, int(keep.sum()))
    clean = (all_a @ true_slowness.reshape(-1))[keep]
    observed = clean + noise
    return all_a, keep, observed, clean


def run_original(ti, truth, seed=20260829, candidates=100, noise_mode="pre_filter"):
    """Run the locally reproduced upstream QS-IDCS chain with a dense operator."""
    Model_obj, fit_vario, qsCat = _original_imports()
    ny, nx = truth.shape
    ti_slowness = binary.slowness_from_binary(ti)
    true_slowness = binary.slowness_from_binary(truth)
    t0 = time.perf_counter()
    all_a, keep, observed, clean = original_forward_data(truth, noise_mode)
    operator_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    est_a0, est_alpha, est_var = _get_variogram(ti_slowness, fit_vario)
    variogram_seconds = time.perf_counter() - t0
    model = Model_obj(
        data_cond=1, LikeProb=2, sampProp=1, x=nx, y=ny, sigma_d=1.0,
        sigma_m=float(np.std(ti_slowness)), mu_m=float(np.mean(ti_slowness)),
        A=all_a, d_obs=observed[:, None],
    )
    model.fw = "linear"
    model.index = np.flatnonzero(~keep)
    model.est_a0 = est_a0
    model.est_alpha = est_alpha
    model.est_var = est_var
    model.rand = 0
    model.seed = np.array([seed], dtype=int)
    model.update = 0
    dst = np.full((ny, nx), np.nan)
    path = np.random.default_rng(seed).permutation(dst.size)
    kernel = np.ones((31, 31), dtype=float)
    kernel[15, 15] = 0.0
    kernel = np.exp(-0.0 * distance_transform_edt(kernel))
    t0 = time.perf_counter()
    result_slowness, wrmse_trace, _ = qsCat(
        ti_slowness, dst, path, kernel, 25, candidates, true_model=true_slowness,
        modelObj=model, thresh=100, idr=0,
    )
    simulation_seconds = time.perf_counter() - t0
    result = binary.binary_from_slowness(result_slowness)
    metrics = sparse_solver.binary_metrics(result, truth)
    predicted = (all_a @ np.asarray(result_slowness).reshape(-1))[keep]
    metrics["wrmse"] = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    summary = {
        "method": "Original QS-IDCS local reproduction",
        "software": software_versions(),
        "parameters": {
            "shape_yx": [ny, nx], "spacing_m": 0.1, "sigma_d_ns": 1.0,
            "antenna_spacing_cells": 4, "angle_limit_deg": 50.0,
            "candidate_count": candidates, "neighbors": 25, "template_size": 31,
            "sample_mode": "proportional", "seed": seed, "noise_mode": noise_mode,
        },
        "operator": {"retained_rays": int(keep.sum()), "all_rays": int(len(keep)),
                     "rejected_rays": int((~keep).sum()), "dense_bytes": int(all_a.nbytes)},
        "metrics": metrics,
        "timing_seconds": {
            "operator": operator_seconds, "variogram": variogram_seconds,
            "simulation": simulation_seconds,
            "total": operator_seconds + variogram_seconds + simulation_seconds,
        },
    }
    return {
        "summary": summary, "truth": truth, "training_image": ti, "result": result,
        "operator": all_a[keep], "observed": observed, "clean": clean,
        "wrmse_trace": np.asarray(wrmse_trace),
    }


def save_run(output, run):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(run["summary"], indent=2), encoding="utf-8")
    np.save(output / "reconstruction.npy", run["result"].astype(np.uint8))
    np.save(output / "observed_traveltime_ns.npy", np.asarray(run["observed"]).reshape(-1))
    if "pre_map" in run and run["summary"]["parameters"]["map_enabled"]:
        np.save(output / "before_map.npy", run["pre_map"].astype(np.uint8))
    return output
