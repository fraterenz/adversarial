import logging
import torch
from hypothesis import given, strategies as st

from adversarial.attack import Perturbation, ProjGradL2

TOL = 1e-4
log = logging.getLogger(__name__)
MAX_H, MAX_W, MAX_BATCH = 264, 264, 10


@given(
    st.integers(min_value=1, max_value=MAX_BATCH),
    st.integers(min_value=1, max_value=MAX_H),
    st.integers(min_value=1, max_value=MAX_W),
    st.booleans(),
    st.integers(min_value=0, max_value=30000),
)
def test_proj_grad_ones_initial(N, H, W, is_l2, seed):
    torch.manual_seed(seed)
    C = 3
    pert_data = torch.ones((N, C, H, W), requires_grad=True)
    pert = Perturbation(pert_data.clone())
    # log.debug("%s", pert)
    eps, lr = 1.5, 0.1
    # fake gradient
    pert.pert.grad = torch.randn_like(pert.pert)

    new_pert = ProjGradL2(lr, eps).perturb(pert)

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
