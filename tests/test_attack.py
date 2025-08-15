import logging
import torch
import numpy as np
from hypothesis import given, strategies as st

from adversarial import TorchImageProcessed
from adversarial.attack import Perturbation, ProjGradL2, ProjGradLInf

TOL = 5e-4
log = logging.getLogger(__name__)
MAX_H, MAX_W, MAX_BATCH = 264, 264, 10
# RBG chanel
C = 3


@given(
    st.integers(min_value=1, max_value=MAX_BATCH),
    st.integers(min_value=1, max_value=MAX_H),
    st.integers(min_value=1, max_value=MAX_W),
    st.booleans(),
    st.integers(min_value=0, max_value=30000),
)
def test_proj_grad_ones_initial(N, H, W, is_l2, seed):
    torch.manual_seed(seed)
    pert_data = torch.ones((N, C, H, W), requires_grad=True)
    pert = Perturbation(pert_data.clone())
    # log.debug("%s", pert)
    eps, lr = 1.5, 0.1
    # fake gradient
    pert.pert.grad = torch.randn_like(pert.pert)

    new_pert = Perturbation(pert_data.clone())
    new_pert.pert.grad = torch.randn_like(pert.pert)
    ProjGradL2(lr, eps).perturb_(new_pert)

    with torch.no_grad():
        assert torch.all((new_pert.pert - pert_data).abs() >= TOL)
        # log.debug("%s", pert_data)
        assert torch.all(
            (new_pert.pert.mean(dim=(1, 2, 3)) - pert_data.mean(dim=(1, 2, 3))).abs()
            >= TOL
        )
        # inside L2 ball per image
        norms = torch.linalg.vector_norm(new_pert.pert, 2, dim=(1, 2, 3))
        assert torch.all(norms <= eps + TOL)


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
    scaling = (eps * 0.9) / grad_method.norm(
        eps * torch.ones((1, C, H, W), dtype=torch.float16) / (C * H * W)
    ).item()
    min_, max_ = -scaling, scaling
    rand_smaller_than_eps = torch.testing.make_tensor(
        (1, C, H, W), dtype=torch.float16, device="cpu", low=min_, high=max_
    )
    v = rand_smaller_than_eps.clone()
    assert torch.all(
        rand_smaller_than_eps.abs() <= (eps * torch.ones_like(rand_smaller_than_eps))
    )
    assert grad_method.norm(v).item() < eps
    grad_method.projection_(v)
    torch.testing.assert_close(rand_smaller_than_eps, v)


@given(
    st.integers(min_value=1, max_value=MAX_H),
    st.integers(min_value=1, max_value=MAX_W),
    st.floats(0.01, 1),
    st.floats(0, 1),
    st.booleans(),
)
def test_projection_all_epsilon(H, W, lr, eps, is_l2):
    grad_method = ProjGradL2(lr, eps) if is_l2 else ProjGradLInf(lr, eps)
    v = 2 * torch.ones((1, C, H, W))
    grad_method.projection_(v)
    np.testing.assert_allclose(grad_method.norm(v).item(), eps, atol=TOL)


@given(
    st.integers(min_value=1, max_value=MAX_BATCH),
    st.integers(min_value=1, max_value=MAX_H),
    st.integers(min_value=1, max_value=MAX_W),
    st.floats(0.01, 1),
    st.floats(0, 1),
    st.booleans(),
)
def test_initialisation_perturbation(N, H, W, lr, eps, is_l2):
    grad_method = ProjGradL2(lr, eps) if is_l2 else ProjGradLInf(lr, eps)
    v = grad_method.initialise_perturbation(
        TorchImageProcessed(torch.empty((N, C, W, H)))
    )
    v_norm = grad_method.norm(v.pert)
    if torch.any(v_norm > eps):
        log.info("%s", eps)
        log.info("%s", v_norm)
    assert torch.all((v_norm - eps).abs() <= TOL * 10)
