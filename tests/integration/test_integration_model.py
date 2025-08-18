from pathlib import Path
import numpy as np

import torch
from torchvision.models import ResNet50_Weights

from adversarial.model import ResNet50
from adversarial.utils import load_image

BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")
NB_CLASSES = len(ResNet50_Weights.DEFAULT.meta["categories"])
EXPECTED_SIZE = torch.Size([1, NB_CLASSES])
MY_MDL = ResNet50()
PANDA = MY_MDL.preprocess(load_image(BASEPATH / "panda.jpeg"))
CAT = MY_MDL.preprocess(load_image(BASEPATH / "tabby_cat.jpeg"))


def test_resnet50_predict_panda():
    expected = "giant panda"
    label, score = MY_MDL.predict_label(PANDA)
    assert label == expected
    assert score > 0.5

    prediction = MY_MDL.predict(PANDA)
    assert prediction.shape == EXPECTED_SIZE
    prediction.squeeze_(0)
    np.testing.assert_almost_equal(prediction.sum(0).item(), 1.0)
    assert (prediction < 1).all().item()


def test_resnet50_predict_tabby_cat():
    label, score = MY_MDL.predict_label(CAT)
    assert label == "tabby"
    assert score > 0.3
