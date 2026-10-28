import math
import os
import numpy as np
from scipy.ndimage import distance_transform_edt, label, uniform_filter

from . import binary as encoding
from .straight_ray import tomokernel_straight_2D

NX = int(os.environ.get("MF_IDCS_NX", "50"))
NY = int(os.environ.get("MF_IDCS_NY", "100"))

RANDOM_SEED = int(os.environ.get("MF_IDCS_SEED", "20260817"))
SPACING_M = float(os.environ.get("MF_IDCS_SPACING_M", "0.1"))
SNR_SPACING = int(os.environ.get("MF_IDCS_SNR_SPACING", "4"))
SIGMA_D = float(os.environ.get("MF_IDCS_SIGMA_D", "1.0"))
LIMIT_ANGLE_DEG = float(os.environ.get("MF_IDCS_LIMIT_ANGLE_DEG", "50.0"))

DATA_WEIGHT = float(os.environ.get("MF_IDCS_DATA_WEIGHT", "1.0"))
MPS_WEIGHT = float(os.environ.get("MF_IDCS_MPS_WEIGHT", "1.0"))
PRIOR_STRENGTH = float(os.environ.get("MF_IDCS_PRIOR_STRENGTH", "2.0"))
MIN_PATTERN_COUNT = int(os.environ.get("MF_IDCS_MIN_PATTERN_COUNT", "12"))
MAX_COND_NEIGHBORS = int(os.environ.get("MF_IDCS_MAX_COND_NEIGHBORS", "12"))
SAMPLE_MODE = os.environ.get("MF_IDCS_SAMPLE_MODE", "sample")  # sample or argmax
INITIAL_MODE = os.environ.get("MF_IDCS_INITIAL_MODE", "prior")  # prior, nonkarst, random_crop
PATH_MODE = os.environ.get("MF_IDCS_PATH_MODE", "coverage_random")  # random, coverage, coverage_random
HARD_MODE = os.environ.get("MF_IDCS_HARD_MODE", "none")  # none, top_bottom
RESIDUAL_CORRECT_EVERY_STEPS = int(os.environ.get("MF_IDCS_RESIDUAL_CORRECT_EVERY_STEPS", "0"))
RESIDUAL_CORRECT_EVERY_FRACTION = float(os.environ.get("MF_IDCS_RESIDUAL_CORRECT_EVERY_FRACTION", "0.02"))
CANDIDATE_MODE = os.environ.get("MF_IDCS_CANDIDATE_MODE", "pattern_prior")  # pattern_prior, qs_fft, qs_fft_candidates
LIKELIHOOD_BASELINE_MODE = os.environ.get("MF_IDCS_LIKELIHOOD_BASELINE", "prior_mean")  # prior_mean, initial_field
MEAN_UPDATE_MODE = os.environ.get("MF_IDCS_MEAN_UPDATE_MODE", "point")  # point, local_cov, local_cov_score
LOCAL_COV_RADIUS = int(os.environ.get("MF_IDCS_LOCAL_COV_RADIUS", "10"))
LOCAL_COV_RANGE = float(os.environ.get("MF_IDCS_LOCAL_COV_RANGE", "6.0"))
LOCAL_COV_ALPHA = float(os.environ.get("MF_IDCS_LOCAL_COV_ALPHA", "1.0"))
LOCAL_COV_RELAXATION = float(os.environ.get("MF_IDCS_LOCAL_COV_RELAXATION", "1.0"))
REFINE_ENABLE = os.environ.get("MF_IDCS_REFINE_ENABLE", "0").strip().lower() in ("1", "true", "yes", "on")
REFINE_PASSES = int(os.environ.get("MF_IDCS_REFINE_PASSES", "4"))
REFINE_BOUNDARY_ONLY = os.environ.get("MF_IDCS_REFINE_BOUNDARY_ONLY", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
REFINE_DATA_WEIGHT = float(os.environ.get("MF_IDCS_REFINE_DATA_WEIGHT", "1.0"))
REFINE_MPS_WEIGHT = float(os.environ.get("MF_IDCS_REFINE_MPS_WEIGHT", "1.0"))
REFINE_SMOOTH_WEIGHT = float(os.environ.get("MF_IDCS_REFINE_SMOOTH_WEIGHT", "0.0"))
REFINE_MIN_GAIN = float(os.environ.get("MF_IDCS_REFINE_MIN_GAIN", "0.0"))
QS_CANDIDATES = int(os.environ.get("MF_IDCS_QS_K", "100"))
QS_NEIGHBORS = int(os.environ.get("MF_IDCS_QS_N", "25"))
QS_TEMPLATE_SIZE = int(os.environ.get("MF_IDCS_QS_TEMPLATE_SIZE", "31"))
QS_TEMPLATE_ALPHA = float(os.environ.get("MF_IDCS_QS_TEMPLATE_ALPHA", "0.0"))

SSIM_WINDOW = 7
SSIM_C1 = 0.01
SSIM_C2 = 0.03
EPS = 1.0e-9
LOCAL_COV_CACHE = {}


# Deliberately small enough to keep the exact pattern table fast on a 500x500 TI.
# The order is near-to-far; far offsets are dropped first when a pattern is rare.
TEMPLATE_OFFSETS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (0, -2),
    (0, 2),
    (-2, 0),
    (2, 0),
    (0, -3),
    (0, 3),
    (-3, 0),
    (3, 0),
]


def slowness_from_binary(binary):
    return encoding.slowness_from_binary(encoding.ensure_binary(binary)).astype(float)


def binary_from_slowness(slowness):
    return encoding.binary_from_slowness(slowness)


def ssim_paper(a, b):
    a = encoding.ensure_binary(a).astype(float)
    b = encoding.ensure_binary(b).astype(float)
    ux = uniform_filter(a, size=SSIM_WINDOW, mode="reflect")
    uy = uniform_filter(b, size=SSIM_WINDOW, mode="reflect")
    vx = uniform_filter(a * a, size=SSIM_WINDOW, mode="reflect") - ux * ux
    vy = uniform_filter(b * b, size=SSIM_WINDOW, mode="reflect") - uy * uy
    vxy = uniform_filter(a * b, size=SSIM_WINDOW, mode="reflect") - ux * uy
    ssim_map = ((2 * ux * uy + SSIM_C1) * (2 * vxy + SSIM_C2)) / (
        (ux * ux + uy * uy + SSIM_C1) * (vx + vy + SSIM_C2)
    )
    return float(np.mean(ssim_map))


def structure_metrics(binary):
    binary = encoding.ensure_binary(binary)
    labels, n_components = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if n_components:
        sizes = np.bincount(labels.ravel())[1:]
        largest = int(sizes.max())
    else:
        largest = 0
    return {
        "karst_ratio": float(binary.mean()),
        "karst_cells": int(binary.sum()),
        "component_count_8conn": int(n_components),
        "largest_component_area": largest,
        "largest_component_ratio_in_karst": float(largest / max(int(binary.sum()), 1)),
        "horizontal_spanning": int(any(np.any((labels == lab)[:, 0]) and np.any((labels == lab)[:, -1]) for lab in range(1, n_components + 1))),
        "vertical_spanning": int(any(np.any((labels == lab)[0, :]) and np.any((labels == lab)[-1, :]) for lab in range(1, n_components + 1))),
    }


def binary_metrics(pred, true):
    out = encoding.binary_metrics(pred, true)
    out.update(structure_metrics(pred))
    out["ssim_paper"] = ssim_paper(true, pred)
    return out


def build_hard_data(true_binary):
    hard_mask = np.zeros_like(true_binary, dtype=np.uint8)
    hard_values = np.zeros_like(true_binary, dtype=np.uint8)
    if HARD_MODE == "top_bottom":
        hard_mask[0, :] = 1
        hard_mask[-1, :] = 1
        hard_values[hard_mask > 0] = true_binary[hard_mask > 0]
    elif HARD_MODE != "none":
        raise ValueError(f"Unsupported MF_IDCS_HARD_MODE={HARD_MODE}")
    return hard_mask, hard_values


def build_sparse_linear_operator(true_slowness):
    """Build a sparse straight-ray operator and keep it sparse.

    This follows the geometry of the original IDCS reproduction but does not
    call .todense(). The returned CSR/CSC matrices are used only through sparse
    products and sparse column updates.
    """
    ny, nx = true_slowness.shape
    start = 2
    sourcex = 0.0
    receiverx = nx * SPACING_M
    sourcez = np.arange(start, ny, SNR_SPACING) * SPACING_M
    receiverz = np.arange(start, ny, SNR_SPACING) * SPACING_M
    x_edges = np.arange(0, nx + 1, dtype=float) * SPACING_M
    z_edges = np.arange(0, ny + 1, dtype=float) * SPACING_M

    rows = []
    ray_records = []
    for jj, sz in enumerate(sourcez):
        for ii, rz in enumerate(receiverz):
            rows.append([sourcex, sz + SPACING_M / 2.0, receiverx, rz + SPACING_M / 2.0])
            angle = math.degrees(math.atan(abs(rz - sz) / max(receiverx - sourcex, EPS)))
            ray_records.append((len(rows) - 1, 0.0, float(sz / SPACING_M), float(nx - 1), float(rz / SPACING_M), angle))
    data = np.asarray(rows, dtype=float)
    A_full = tomokernel_straight_2D(data, x_edges, z_edges).tocsr()
    keep = np.ones(A_full.shape[0], dtype=bool)
    if LIMIT_ANGLE_DEG > 0:
        keep = np.asarray([rec[-1] <= LIMIT_ANGLE_DEG for rec in ray_records], dtype=bool)
    A = A_full[keep].tocsr()
    kept_records = [rec for rec, ok in zip(ray_records, keep) if ok]

    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, SIGMA_D, size=A.shape[0])
    d_clean = A @ true_slowness.reshape(-1)
    d_obs = d_clean + noise
    return A, A.tocsc(), d_obs, d_clean, kept_records, int((~keep).sum()), len(ray_records)


class PatternPrior:
    def __init__(self, ti_binary, offsets, max_cond_neighbors=12, min_count=12, prior_strength=2.0):
        self.ti = encoding.ensure_binary(ti_binary)
        self.offsets = list(offsets)
        self.max_cond_neighbors = int(max_cond_neighbors)
        self.min_count = int(min_count)
        self.prior_strength = float(prior_strength)
        self.prior = float(self.ti.mean())
        self.cache = {}
        self._build_pattern_table()

    def _build_pattern_table(self):
        dys = [dy for dy, _ in self.offsets]
        dxs = [dx for _, dx in self.offsets]
        margin_top = max(0, -min(dys))
        margin_bottom = max(0, max(dys))
        margin_left = max(0, -min(dxs))
        margin_right = max(0, max(dxs))
        ys = np.arange(margin_top, self.ti.shape[0] - margin_bottom)
        xs = np.arange(margin_left, self.ti.shape[1] - margin_right)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        self.center_values = self.ti[yy, xx].reshape(-1).astype(np.uint8)
        pattern_columns = []
        for dy, dx in self.offsets:
            pattern_columns.append(self.ti[yy + dy, xx + dx].reshape(-1).astype(np.uint8))
        self.pattern_values = np.stack(pattern_columns, axis=1)

    def probability(self, known_indices, known_values):
        if len(known_indices) == 0:
            return self.prior, 0, 0
        idx = list(known_indices[: self.max_cond_neighbors])
        vals = list(known_values[: self.max_cond_neighbors])
        key = tuple(zip(idx, vals))
        if key in self.cache:
            return self.cache[key]

        idx_arr = np.asarray(idx, dtype=int)
        vals_arr = np.asarray(vals, dtype=np.uint8)
        while idx_arr.size:
            match = np.all(self.pattern_values[:, idx_arr] == vals_arr[None, :], axis=1)
            count = int(match.sum())
            if count >= self.min_count or idx_arr.size == 1:
                count1 = int(self.center_values[match].sum()) if count else 0
                prob = (count1 + self.prior_strength * self.prior) / (count + self.prior_strength)
                result = (float(np.clip(prob, 1.0e-4, 1.0 - 1.0e-4)), count, int(idx_arr.size))
                self.cache[key] = result
                return result
            idx_arr = idx_arr[:-1]
            vals_arr = vals_arr[:-1]

        result = (self.prior, 0, 0)
        self.cache[key] = result
        return result


class QSFftCandidatePrior:
    """Original-QS-style FFT candidate proposal, reduced to a center-value prior."""

    def __init__(self, ti_binary, k=100, n=25, template_size=31, alpha=0.0, prior_strength=2.0):
        self.ti = encoding.ensure_binary(ti_binary).astype(np.uint8)
        self.k = int(k)
        self.n = int(n)
        self.prior_strength = float(prior_strength)
        self.prior = float(self.ti.mean())
        size = int(template_size)
        if size % 2 == 0:
            size += 1
        self.template = np.ones((size, size), dtype=float)
        center = size // 2
        self.template[center, center] = 0.0
        self.template = np.exp(-float(alpha) * distance_transform_edt(self.template))
        self.dist = np.zeros_like(self.template, dtype=float)
        self.dist[center, center] = 1.0
        self.dist = distance_transform_edt(1.0 - self.dist)
        self.values = np.unique(self.ti.reshape(-1))
        self.fftim = np.stack([np.fft.fft2(-1 * (self.ti == value)) for value in self.values], axis=2)
        self.cache = {}

    def probability_at(self, field, assigned, flat_index):
        key = self._cache_key(field, assigned, flat_index)
        if key in self.cache:
            candidate_values = self.cache[key]
        else:
            candidate_values = self._candidate_center_values(field, assigned, flat_index)
            self.cache[key] = candidate_values
        if candidate_values.size == 0:
            result = (float(np.clip(self.prior, 1.0e-4, 1.0 - 1.0e-4)), 0, 0)
        else:
            count1 = int(np.sum(candidate_values > 0))
            prob = (count1 + self.prior_strength * self.prior) / (candidate_values.size + self.prior_strength)
            result = (float(np.clip(prob, 1.0e-4, 1.0 - 1.0e-4)), int(candidate_values.size), self.n)
        return result

    def candidates_at(self, field, assigned, flat_index):
        key = self._cache_key(field, assigned, flat_index)
        if key not in self.cache:
            self.cache[key] = self._candidate_center_values(field, assigned, flat_index)
        return self.cache[key]

    def _cache_key(self, field, assigned, flat_index):
        y, x = divmod(int(flat_index), field.shape[1])
        half_y = self.template.shape[0] // 2
        half_x = self.template.shape[1] // 2
        vals = []
        for yy in range(max(0, y - half_y), min(field.shape[0], y + half_y + 1)):
            for xx in range(max(0, x - half_x), min(field.shape[1], x + half_x + 1)):
                if assigned[yy, xx]:
                    vals.append((yy - y, xx - x, int(field[yy, xx])))
        return tuple(vals)

    def _candidate_center_values(self, field, assigned, flat_index):
        y, x = divmod(int(flat_index), field.shape[1])
        half_y = self.template.shape[0] // 2
        half_x = self.template.shape[1] // 2
        source = np.full(self.template.shape, np.nan, dtype=float)
        y0 = max(0, y - half_y)
        y1 = min(field.shape[0], y + half_y + 1)
        x0 = max(0, x - half_x)
        x1 = min(field.shape[1], x + half_x + 1)
        sy0 = y0 - (y - half_y)
        sx0 = x0 - (x - half_x)
        block = field[y0:y1, x0:x1]
        block_assigned = assigned[y0:y1, x0:x1]
        local = source[sy0 : sy0 + block.shape[0], sx0 : sx0 + block.shape[1]]
        local[block_assigned] = block[block_assigned]

        for radius in range(1, max(self.template.shape) // 2):
            if np.isfinite(source[self.dist <= radius]).sum() >= self.n:
                break
        source[self.dist > radius] = np.nan

        pad_y = self.ti.shape[0] - source.shape[0]
        pad_x = self.ti.shape[1] - source.shape[1]
        if pad_y < 0 or pad_x < 0:
            raise ValueError("QS FFT template cannot be larger than training image.")
        extend_source = np.pad(source, ((0, pad_y), (0, pad_x)), "constant", constant_values=np.nan)
        extend_template = np.pad(self.template, ((0, pad_y), (0, pad_x)), "constant", constant_values=0)
        terms = [
            self.fftim[:, :, value_id] * np.conj(np.fft.fft2(extend_template * np.nan_to_num(extend_source == value)))
            for value_id, value in enumerate(self.values)
        ]
        mismatch = np.real(np.fft.ifft2(np.sum(np.stack(terms, axis=2), axis=2)))
        mismatch[-self.template.shape[0] + 1 :, :] = np.nan
        mismatch[:, -self.template.shape[1] + 1 :] = np.nan
        rolled = np.roll(mismatch, tuple(side // 2 for side in self.template.shape), axis=(0, 1)).reshape(-1)
        finite = np.isfinite(rolled)
        if not np.any(finite):
            return np.asarray([], dtype=np.uint8)
        finite_idx = np.flatnonzero(finite)
        k = min(self.k, finite_idx.size)
        chosen = finite_idx[np.argpartition(rolled[finite_idx], k - 1)[:k]]
        return self.ti.reshape(-1)[chosen]


def initialize_field(shape, ti_binary, hard_mask, hard_values, rng):
    if INITIAL_MODE == "nonkarst":
        field = np.zeros(shape, dtype=np.uint8)
    elif INITIAL_MODE == "random_crop":
        max_y = ti_binary.shape[0] - shape[0]
        max_x = ti_binary.shape[1] - shape[1]
        y0 = int(rng.integers(0, max_y + 1))
        x0 = int(rng.integers(0, max_x + 1))
        field = ti_binary[y0 : y0 + shape[0], x0 : x0 + shape[1]].copy()
    elif INITIAL_MODE == "prior":
        prior = float(ti_binary.mean())
        field = (rng.random(shape) < prior).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported MF_IDCS_INITIAL_MODE={INITIAL_MODE}")
    field[hard_mask > 0] = hard_values[hard_mask > 0]
    return field


def initialize_likelihood_slowness(field, ti_binary, hard_mask, hard_values):
    if LIKELIHOOD_BASELINE_MODE == "initial_field":
        slowness = slowness_from_binary(field)
    elif LIKELIHOOD_BASELINE_MODE == "prior_mean":
        mu_s = float(np.mean(slowness_from_binary(ti_binary)))
        slowness = np.full(field.shape, mu_s, dtype=float)
        if np.any(hard_mask > 0):
            hard_slowness = slowness_from_binary(hard_values)
            slowness[hard_mask > 0] = hard_slowness[hard_mask > 0]
    else:
        raise ValueError(f"Unsupported MF_IDCS_LIKELIHOOD_BASELINE={LIKELIHOOD_BASELINE_MODE}")
    return slowness


def make_path(hard_mask, csc, rng):
    all_idx = np.arange(hard_mask.size, dtype=np.int64)
    hard_flat = hard_mask.reshape(-1).astype(bool)
    candidates = all_idx[~hard_flat]
    if PATH_MODE == "random":
        return rng.permutation(candidates)
    coverage = np.diff(csc.indptr).astype(float)
    if PATH_MODE == "coverage":
        return candidates[np.argsort(-coverage[candidates])]
    if PATH_MODE == "coverage_random":
        # Bin by ray coverage, randomize within bins, then simulate high-coverage
        # cells first. This is a cheap entropy/path proxy and avoids deterministic
        # scan artifacts.
        cov = coverage[candidates]
        jitter = rng.random(candidates.size) * 0.01
        order = np.lexsort((jitter, -cov))
        return candidates[order]
    raise ValueError(f"Unsupported MF_IDCS_PATH_MODE={PATH_MODE}")


def local_known_pattern(field, assigned, y, x):
    known_idx = []
    known_vals = []
    ny, nx = field.shape
    for offset_id, (dy, dx) in enumerate(TEMPLATE_OFFSETS):
        yy = y + dy
        xx = x + dx
        if 0 <= yy < ny and 0 <= xx < nx and assigned[yy, xx]:
            known_idx.append(offset_id)
            known_vals.append(int(field[yy, xx]))
            if len(known_idx) >= MAX_COND_NEIGHBORS:
                break
    return known_idx, known_vals


def sparse_column(csc, flat_index):
    start = csc.indptr[flat_index]
    end = csc.indptr[flat_index + 1]
    return csc.indices[start:end], csc.data[start:end]


def candidate_data_loglik(residual, ray_ids, ray_weights, delta_s):
    if ray_ids.size == 0 or abs(delta_s) < EPS:
        return 0.0
    r = residual[ray_ids]
    a = ray_weights
    return float((delta_s * np.dot(a, r) - 0.5 * delta_s * delta_s * np.dot(a, a)) / (SIGMA_D * SIGMA_D))


def local_cov_support(shape, flat_index):
    key = (shape, int(flat_index), LOCAL_COV_RADIUS, LOCAL_COV_RANGE, LOCAL_COV_ALPHA, LOCAL_COV_RELAXATION)
    if key in LOCAL_COV_CACHE:
        return LOCAL_COV_CACHE[key]
    y, x = divmod(int(flat_index), shape[1])
    y0 = max(0, y - LOCAL_COV_RADIUS)
    y1 = min(shape[0], y + LOCAL_COV_RADIUS + 1)
    x0 = max(0, x - LOCAL_COV_RADIUS)
    x1 = min(shape[1], x + LOCAL_COV_RADIUS + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
    mask = dist <= LOCAL_COV_RADIUS
    cols = (yy[mask] * shape[1] + xx[mask]).astype(np.int64)
    weights = np.exp(-np.power(dist[mask] / max(LOCAL_COV_RANGE, EPS), LOCAL_COV_ALPHA))
    weights = (LOCAL_COV_RELAXATION * weights).astype(float)
    center = cols == int(flat_index)
    if np.any(center):
        weights[center] = 1.0
    LOCAL_COV_CACHE[key] = (cols, weights)
    return cols, weights


def candidate_residual_effect(A_csc, residual, assigned, likelihood_slowness, flat_index, candidate_s, purpose="score"):
    current_s = float(likelihood_slowness.reshape(-1)[int(flat_index)])
    mode = MEAN_UPDATE_MODE
    if mode == "local_cov_score" and purpose == "update":
        mode = "point"
    if mode == "point":
        delta = float(candidate_s - current_s)
        if abs(delta) <= EPS:
            return 0.0, np.asarray([], dtype=np.int64), np.asarray([], dtype=float), np.zeros_like(residual)
        ray_ids, ray_weights = sparse_column(A_csc, int(flat_index))
        cols = np.asarray([int(flat_index)], dtype=np.int64)
        deltas = np.asarray([delta], dtype=float)
        effect = np.zeros_like(residual, dtype=float)
        if ray_ids.size:
            effect[ray_ids] = ray_weights * delta
    elif mode in ("local_cov", "local_cov_score"):
        cols, weights = local_cov_support(likelihood_slowness.shape, int(flat_index))
        assigned_flat = assigned.reshape(-1)
        like_flat = likelihood_slowness.reshape(-1)
        active = (~assigned_flat[cols]) | (cols == int(flat_index))
        cols = cols[active]
        weights = weights[active]
        if cols.size == 0:
            return 0.0, np.asarray([], dtype=np.int64), np.asarray([], dtype=float), np.zeros_like(residual)
        old_values = like_flat[cols]
        proposed = old_values + weights * float(candidate_s - current_s)
        proposed[cols == int(flat_index)] = candidate_s
        lo = min(encoding.SLOWNESS_NONKARST, encoding.SLOWNESS_KARST)
        hi = max(encoding.SLOWNESS_NONKARST, encoding.SLOWNESS_KARST)
        proposed = np.clip(proposed, lo, hi)
        deltas = proposed - old_values
        keep = np.abs(deltas) > EPS
        cols = cols[keep]
        deltas = deltas[keep]
        if cols.size == 0:
            return 0.0, cols, deltas, np.zeros_like(residual)
        effect = np.asarray(A_csc[:, cols] @ deltas).reshape(-1)
    else:
        raise ValueError(f"Unsupported MF_IDCS_MEAN_UPDATE_MODE={MEAN_UPDATE_MODE}")

    loglik = float((np.dot(effect, residual) - 0.5 * np.dot(effect, effect)) / (SIGMA_D * SIGMA_D))
    return loglik, cols, deltas, effect


def choose_from_logits(values, logits, rng):
    values = np.asarray(values, dtype=np.uint8)
    logits = np.asarray(logits, dtype=float)
    if values.size == 0:
        return 0, 0.0
    if SAMPLE_MODE == "argmax":
        idx = int(np.argmax(logits))
    elif SAMPLE_MODE == "sample":
        centered = logits - float(np.max(logits))
        weights = np.exp(centered)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0:
            idx = int(rng.integers(0, values.size))
        else:
            probs = weights / total
            idx = int(rng.choice(values.size, p=probs))
    else:
        raise ValueError(f"Unsupported MF_IDCS_SAMPLE_MODE={SAMPLE_MODE}")
    return int(values[idx]), float(logits[idx])


def residual_correction_interval(path_size):
    if RESIDUAL_CORRECT_EVERY_STEPS > 0:
        return RESIDUAL_CORRECT_EVERY_STEPS
    if RESIDUAL_CORRECT_EVERY_FRACTION > 0:
        return max(1, int(round(path_size * RESIDUAL_CORRECT_EVERY_FRACTION)))
    return 0


def recompute_residual(A_csr, d_obs, likelihood_slowness):
    return d_obs - (A_csr @ likelihood_slowness.reshape(-1))


def local_refine_prior_log(prior, field, y, x, value):
    known_idx = []
    known_vals = []
    ny, nx = field.shape
    for offset_id, (dy, dx) in enumerate(TEMPLATE_OFFSETS):
        yy = y + dy
        xx = x + dx
        if 0 <= yy < ny and 0 <= xx < nx:
            known_idx.append(offset_id)
            known_vals.append(int(field[yy, xx]))
            if len(known_idx) >= MAX_COND_NEIGHBORS:
                break
    p_mps, _, _ = prior.probability(known_idx, known_vals)
    if value:
        return math.log(max(p_mps, EPS))
    return math.log(max(1.0 - p_mps, EPS))


def local_refine_boundary_and_smooth(field, y, x, old_value, new_value):
    has_opposite_neighbor = False
    same_old = 0
    same_new = 0
    ny, nx = field.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy = y + dy
            xx = x + dx
            if not (0 <= yy < ny and 0 <= xx < nx):
                continue
            neighbor = int(field[yy, xx])
            has_opposite_neighbor = has_opposite_neighbor or neighbor != old_value
            same_old += int(neighbor == old_value)
            same_new += int(neighbor == new_value)
    return has_opposite_neighbor, same_new - same_old


def local_map_refine(A_csr, A_csc, d_obs, field, ti_binary, hard_mask):
    """Boundary-only local MAP polish using sparse residual updates.

    The sequential simulation remains the main IDCS step. This optional pass
    repairs local boundary decisions by accepting flips that improve a combined
    sparse-data residual and local TI-pattern prior objective.
    """
    refined = encoding.ensure_binary(field).astype(np.uint8).copy()
    hard_flat = hard_mask.reshape(-1).astype(bool)
    prior = PatternPrior(
        ti_binary,
        TEMPLATE_OFFSETS,
        max_cond_neighbors=MAX_COND_NEIGHBORS,
        min_count=MIN_PATTERN_COUNT,
        prior_strength=PRIOR_STRENGTH,
    )
    residual = d_obs - (A_csr @ slowness_from_binary(refined).reshape(-1))
    coverage = np.diff(A_csc.indptr).astype(float)
    order = np.argsort(-coverage)
    s0 = encoding.SLOWNESS_NONKARST
    s1 = encoding.SLOWNESS_KARST
    trace = []

    for pass_id in range(1, REFINE_PASSES + 1):
        accepted = 0
        for flat in order:
            flat = int(flat)
            if hard_flat[flat]:
                continue
            y, x = divmod(flat, refined.shape[1])
            old_value = int(refined[y, x])
            new_value = 1 - old_value
            is_boundary, smooth_gain = local_refine_boundary_and_smooth(refined, y, x, old_value, new_value)
            if REFINE_BOUNDARY_ONLY and not is_boundary:
                continue

            delta_s = (s1 if new_value else s0) - (s1 if old_value else s0)
            ray_ids, ray_weights = sparse_column(A_csc, flat)
            if ray_ids.size:
                data_gain = (
                    delta_s * np.dot(ray_weights, residual[ray_ids])
                    - 0.5 * delta_s * delta_s * np.dot(ray_weights, ray_weights)
                ) / (SIGMA_D * SIGMA_D)
            else:
                data_gain = 0.0
            mps_gain = local_refine_prior_log(prior, refined, y, x, new_value) - local_refine_prior_log(
                prior, refined, y, x, old_value
            )
            gain = (
                REFINE_DATA_WEIGHT * float(data_gain)
                + REFINE_MPS_WEIGHT * float(mps_gain)
                + REFINE_SMOOTH_WEIGHT * float(smooth_gain)
            )
            if gain <= REFINE_MIN_GAIN:
                continue

            refined[y, x] = new_value
            if ray_ids.size:
                residual[ray_ids] -= ray_weights * delta_s
            accepted += 1

        trace.append(
            {
                "pass": int(pass_id),
                "accepted_flips": int(accepted),
                "wrmse": float(np.sqrt(np.mean((residual / SIGMA_D) ** 2))),
                "karst_ratio": float(refined.mean()),
            }
        )
        if accepted == 0:
            break

    return refined, trace


def simulate_sparse(ti_binary, A_csr, A_csc, d_obs, hard_mask, hard_values):
    """Run sparse sequential conditioning without access to a synthetic truth field.

    The truth field is used outside this solver only to generate synthetic observations
    and to calculate post-run evaluation metrics.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    if CANDIDATE_MODE == "pattern_prior":
        prior = PatternPrior(
            ti_binary,
            TEMPLATE_OFFSETS,
            max_cond_neighbors=MAX_COND_NEIGHBORS,
            min_count=MIN_PATTERN_COUNT,
            prior_strength=PRIOR_STRENGTH,
        )
    elif CANDIDATE_MODE in ("qs_fft", "qs_fft_candidates"):
        prior = QSFftCandidatePrior(
            ti_binary,
            k=QS_CANDIDATES,
            n=QS_NEIGHBORS,
            template_size=QS_TEMPLATE_SIZE,
            alpha=QS_TEMPLATE_ALPHA,
            prior_strength=PRIOR_STRENGTH,
        )
    else:
        raise ValueError(f"Unsupported MF_IDCS_CANDIDATE_MODE={CANDIDATE_MODE}")
    field = initialize_field(hard_mask.shape, ti_binary, hard_mask, hard_values, rng)
    likelihood_slowness = initialize_likelihood_slowness(field, ti_binary, hard_mask, hard_values)
    residual = recompute_residual(A_csr, d_obs, likelihood_slowness)
    path = make_path(hard_mask, A_csc, rng)
    assigned = hard_mask.astype(bool).copy()
    wrmse_trace = []
    data_loglik_trace = []
    pattern_count_trace = []
    residual_drift_trace = []
    correction_interval = residual_correction_interval(path.size)

    s0 = encoding.SLOWNESS_NONKARST
    s1 = encoding.SLOWNESS_KARST

    for step, flat in enumerate(path, start=1):
        y, x = divmod(int(flat), hard_mask.shape[1])

        if CANDIDATE_MODE == "qs_fft_candidates":
            candidate_values = prior.candidates_at(field, assigned, int(flat))
            if candidate_values.size == 0:
                candidate_values = np.asarray([0, 1], dtype=np.uint8)
            candidate_logits = []
            for candidate in candidate_values:
                candidate_s = s1 if int(candidate) else s0
                loglik, _, _, _ = candidate_residual_effect(
                    A_csc, residual, assigned, likelihood_slowness, int(flat), candidate_s, purpose="score"
                )
                candidate_logits.append(DATA_WEIGHT * loglik)
            new_value, selected_loglik = choose_from_logits(candidate_values, candidate_logits, rng)
            pattern_count = int(candidate_values.size)
            ll = candidate_logits
        else:
            if CANDIDATE_MODE == "pattern_prior":
                known_idx, known_vals = local_known_pattern(field, assigned, y, x)
                p_mps, pattern_count, used_count = prior.probability(known_idx, known_vals)
            else:
                p_mps, pattern_count, used_count = prior.probability_at(field, assigned, int(flat))
            ll = []
            for candidate, candidate_s in ((0, s0), (1, s1)):
                loglik, _, _, _ = candidate_residual_effect(
                    A_csc, residual, assigned, likelihood_slowness, int(flat), candidate_s, purpose="score"
                )
                ll.append(loglik)

            logp0 = MPS_WEIGHT * math.log(max(1.0 - p_mps, EPS)) + DATA_WEIGHT * ll[0]
            logp1 = MPS_WEIGHT * math.log(max(p_mps, EPS)) + DATA_WEIGHT * ll[1]
            new_value, selected_loglik = choose_from_logits([0, 1], [logp0, logp1], rng)

        new_s = s1 if new_value else s0
        _, update_cols, update_deltas, update_effect = candidate_residual_effect(
            A_csc, residual, assigned, likelihood_slowness, int(flat), new_s, purpose="update"
        )
        if update_cols.size:
            residual -= update_effect
            likelihood_slowness.reshape(-1)[update_cols] += update_deltas
        field[y, x] = new_value
        assigned[y, x] = True

        if correction_interval > 0 and (step % correction_interval == 0 or step == path.size):
            exact_residual = recompute_residual(A_csr, d_obs, likelihood_slowness)
            drift = residual - exact_residual
            residual_drift_trace.append(
                {
                    "step": int(step),
                    "max_abs": float(np.max(np.abs(drift))) if drift.size else 0.0,
                    "rmse": float(np.sqrt(np.mean(drift * drift))) if drift.size else 0.0,
                }
            )
            residual = exact_residual

        if step == 1 or step % max(1, path.size // 100) == 0 or step == path.size:
            wrmse_trace.append(float(np.sqrt(np.mean((residual / SIGMA_D) ** 2))))
            data_loglik_trace.append(float(selected_loglik))
            pattern_count_trace.append(int(pattern_count))

    return (
        field,
        np.asarray(wrmse_trace),
        np.asarray(data_loglik_trace),
        np.asarray(pattern_count_trace),
        int(len(prior.cache)),
        residual_drift_trace,
    )


def final_wrmse(A_csr, d_obs, binary):
    pred = A_csr @ slowness_from_binary(binary).reshape(-1)
    return float(np.sqrt(np.mean(((d_obs - pred) / SIGMA_D) ** 2)))


