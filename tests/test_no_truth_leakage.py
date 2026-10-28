import inspect

from sparse_idcs import incremental_solver as solver


def test_solver_does_not_accept_synthetic_truth_field():
    parameters = inspect.signature(solver.simulate_sparse).parameters
    assert "true_binary" not in parameters
    assert tuple(parameters) == ("ti_binary", "A_csr", "A_csc", "d_obs", "hard_mask", "hard_values")
