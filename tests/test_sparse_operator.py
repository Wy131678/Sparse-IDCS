import numpy as np

from sparse_idcs.straight_ray import tomokernel_straight_2D


def test_sparse_ray_operator_matches_dense_product():
    x_edges = np.linspace(0.0, 2.0, 3)
    z_edges = np.linspace(0.0, 2.0, 3)
    rays = np.array([[0.0, 0.5, 2.0, 1.5], [0.0, 1.5, 2.0, 0.5]])
    operator = tomokernel_straight_2D(rays, x_edges, z_edges).tocsr()
    model = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(operator @ model, operator.toarray() @ model, atol=1e-12)
    assert operator.nnz > 0
