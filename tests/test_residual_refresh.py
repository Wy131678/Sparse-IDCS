import numpy as np
from scipy import sparse

from sparse_idcs import incremental_solver as solver


def test_incremental_residual_equals_full_sparse_residual():
    operator = sparse.csr_matrix(np.array([[1.0, 0.2], [0.0, 2.0], [1.5, 0.0]]))
    d_obs = np.array([3.0, 1.0, 2.0])
    model = np.array([[1.0, 1.0]])
    residual = solver.recompute_residual(operator, d_obs, model)
    update = np.array([0.3, -0.4])
    residual -= operator @ update
    model.reshape(-1)[:] += update
    assert np.allclose(residual, solver.recompute_residual(operator, d_obs, model), atol=1e-12)
