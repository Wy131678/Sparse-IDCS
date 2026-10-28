import numpy as np

from sparse_idcs import incremental_solver as solver
from sparse_idcs.research import configure_sparse, original_forward_data, sparse_operator


def test_pre_filter_noise_is_original_full_ray_draw():
    truth = np.zeros((30, 20), dtype=np.uint8)
    configure_sparse(truth.shape, 20260829)
    a, _, observed, clean, records, _, total = sparse_operator(truth, "pre_filter")
    expected_noise = np.random.default_rng(0).normal(0, 1, total)[[int(rec[0]) for rec in records]]
    assert a.shape[0] == observed.size
    assert np.allclose(observed - clean, expected_noise, atol=1e-14, rtol=0)


def test_historical_sparse_noise_order_differs():
    truth = np.zeros((30, 20), dtype=np.uint8)
    configure_sparse(truth.shape, 20260829)
    _, _, historical, _, _, _, _ = sparse_operator(truth, "filtered")
    _, _, shared, _, _, _, _ = sparse_operator(truth, "pre_filter")
    assert not np.array_equal(historical, shared)


def test_paired_dense_and_sparse_observations_agree():
    truth = np.zeros((30, 20), dtype=np.uint8)
    configure_sparse(truth.shape, 20260829)
    sparse_a, _, sparse_obs, _, _, _, _ = sparse_operator(truth, "pre_filter")
    dense_a, keep, dense_obs, _ = original_forward_data(truth, "pre_filter")
    model = solver.slowness_from_binary(truth).reshape(-1)
    assert np.allclose(dense_a[keep] @ model, sparse_a @ model, atol=1e-10)
    assert np.allclose(dense_obs, sparse_obs, atol=1e-10)
