import logging
import torch
from hypothesis import given, strategies as st

from adversarial import TorchImage
from adversarial.model import ResNet50

MAX_H, MAX_W = 264, 264
C = 3
MY_MDL = ResNet50()
CROP_SIZ = MY_MDL.crop_size

log = logging.getLogger(__name__)


@given(
    st.integers(min_value=CROP_SIZ + 1, max_value=MAX_H),
    st.integers(min_value=CROP_SIZ + 1, max_value=MAX_W),
    st.integers(min_value=-10, max_value=30000),
)
def test_resnet50_preprocess(H, W, seed):
    torch.manual_seed(seed)
    x = torch.empty((C, H, W)).uniform_(-2, 2)
    mean_c = torch.mean(x, (1, 2))
    assert mean_c.shape == (C,)
    std_c = torch.std(x, (1, 2))
    assert std_c.shape == (C,)

    x_processed = MY_MDL.preprocess(TorchImage(x))
    assert x_processed.shape == (1, C, CROP_SIZ, CROP_SIZ)
    x_processed.squeeze_(0)
    assert x_processed.shape == (C, CROP_SIZ, CROP_SIZ)
    mean_proc_c = torch.mean(x_processed, (1, 2))
    assert mean_proc_c.shape == (C,)
    std_proc_c = torch.std(x_processed, (1, 2))
    assert std_proc_c.shape == (C,)
    assert MY_MDL.std.shape == MY_MDL.mean.shape
    assert MY_MDL.mean.ndim == x_processed.ndim


@given(
    st.integers(min_value=CROP_SIZ + 1, max_value=MAX_H),
    st.integers(min_value=CROP_SIZ + 1, max_value=MAX_W),
    st.integers(min_value=-10, max_value=30000),
)
def test_normalise(H, W, seed):
    x = torch.empty((C, H, W)).uniform_(-2, 2)
    x_processed = MY_MDL.preprocess(TorchImage(x))
    x_processed1 = x_processed.clone()
    x_processed2 = x_processed.clone()
    MY_MDL.unnormalise_(x_processed1)
    MY_MDL.unnormalise_(x_processed2)
    torch.testing.assert_close(x_processed1, x_processed2)

    MY_MDL.normalise_(x_processed1)
    torch.testing.assert_close(x_processed1, x_processed)


def test_resnet50_init():
    torch.testing.assert_close(
        MY_MDL.mean, torch.tensor([0.485, 0.456, 0.406]).view_as(MY_MDL.mean)
    )
    torch.testing.assert_close(
        MY_MDL.std, torch.tensor([0.229, 0.224, 0.225]).view_as(MY_MDL.std)
    )
    assert len(MY_MDL.meta["categories"]) == 1000
