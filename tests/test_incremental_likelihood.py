import numpy as np
from scipy import sparse

from sparse_idcs import incremental_solver as solver


def test_csc_column_loglik_equals_full_recomputation():
    operator = sparse.csr_matrix(np.array([[1.0, 0.5], [0.0, 2.0], [1.5, 0.0]])).tocsc()
    residual = np.array([2.0, -1.0, 0.5])
    delta = -0.25
    ray_ids, ray_weights = solver.sparse_column(operator, 1)
    incremental = solver.candidate_data_loglik(residual, ray_ids, ray_weights, delta)
    effect = np.zeros_like(residual)
    effect[ray_ids] = ray_weights * delta
    full = np.dot(effect, residual) - 0.5 * np.dot(effect, effect)
    assert np.isclose(incremental, full, atol=1e-12)
