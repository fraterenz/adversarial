import logging
import math
import torch
import numpy as np
from hypothesis import given, strategies as st

from adversarial import TorchImageProcessed
from adversarial.attack import Perturbation, ProjGrad, ProjGradL2, ProjGradLInf

TOL = 1e-5
log = logging.getLogger(__name__)
MAX_H, MAX_W = 264, 264
# RBG chanel
C = 3


def is_v_within_eps_ball(
    projecting_strategy: ProjGrad, v: torch.Tensor, epsilon: float
) -> bool:
    v_norm = projecting_strategy.norm(v).squeeze(0).item()
    return round(v_norm, 5) <= round(epsilon, 5)


@given(
    st.integers(min_value=1, max_value=MAX_H),
    st.integers(min_value=1, max_value=MAX_W),
    st.floats(0.01, 1),
    st.floats(0.1, 1),
    st.booleans(),
    st.integers(min_value=-10, max_value=30000),
)
def test_proj_grad_ones_initial(H, W, lr, eps, is_l2, seed):
    torch.manual_seed(seed)
    strategy = ProjGradL2(lr, eps) if is_l2 else ProjGradLInf(lr, eps)
    x_data = torch.ones((1, C, H, W), requires_grad=True)

    x_bf_proj = Perturbation(x_data.clone())
    x_bf_proj.pert.grad = torch.empty_like(x_bf_proj.pert)

    x_after_proj = Perturbation(x_data.clone())
    x_after_proj.pert.grad = torch.randn_like(x_bf_proj.pert)

    strategy.perturb_(x_after_proj)

    assert is_v_within_eps_ball(strategy, x_after_proj.pert, eps)
    assert torch.all((x_after_proj.pert - x_data).abs() >= TOL)


@given(
    st.integers(min_value=1, max_value=MAX_H),
    st.integers(min_value=1, max_value=MAX_W),
    st.floats(0.01, 1),
    st.floats(0.1, 1),
    st.booleans(),
    st.integers(-10, 12012),
)
def test_projection_random_with_norm_smaller_than_epsilon(H, W, lr, eps, is_l2, seed):
    torch.manual_seed(seed)
    grad_method = ProjGradL2(lr, eps) if is_l2 else ProjGradLInf(lr, eps)
    x = torch.randn((1, C, H, W))
    current_norm = grad_method.norm(x)
    # Scale to have norm < eps (use factor slightly less than e)
    scale_factor = (eps * 0.99) / current_norm
    x.mul_(scale_factor)
    v = x.clone()

    assert is_v_within_eps_ball(grad_method, v, eps)
    assert torch.all(v.abs() <= (eps * torch.ones_like(v)))

    grad_method.projection_(v)
    torch.testing.assert_close(x, v)


@given(
    st.integers(min_value=1, max_value=MAX_H),
    st.integers(min_value=1, max_value=MAX_W),
    st.floats(0.01, 1),
    st.floats(0.01, 1),
    st.booleans(),
)
def test_projection_all_entries_gr_than_epsilon(H, W, lr, eps, is_l2):
    grad_method = ProjGradL2(lr, eps) if is_l2 else ProjGradLInf(lr, eps)
    v = 2 * torch.ones((1, C, H, W), dtype=torch.float32)

    grad_method.projection_(v)
    v_norm = grad_method.norm(v).squeeze(0).item()

    assert math.isclose(v_norm, eps, rel_tol=0.05)
    if is_l2:
        np.testing.assert_allclose(v, eps * v / v_norm, atol=TOL)
    else:
        torch.all(v.abs() <= eps)


@given(
    st.integers(min_value=1, max_value=MAX_H),
    st.integers(min_value=1, max_value=MAX_W),
    st.floats(0.01, 1),
    st.floats(0, 1),
    st.booleans(),
)
def test_initialisation_perturbation(H, W, lr, eps, is_l2):
    grad_method = ProjGradL2(lr, eps) if is_l2 else ProjGradLInf(lr, eps)
    v = grad_method.initialise_perturbation(
        TorchImageProcessed(torch.empty((1, C, W, H)))
    )
    assert is_v_within_eps_ball(grad_method, v.pert, eps)
