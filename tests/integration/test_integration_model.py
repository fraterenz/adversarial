from pathlib import Path
import numpy as np

import torch
from torchvision.models import ResNet50_Weights

from adversarial.model import ResNet50
from adversarial.utils import load_image

BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")
NB_CLASSES = len(ResNet50_Weights.DEFAULT.meta["categories"])
EXPECTED_SIZE = torch.Size([1, NB_CLASSES])


def test_resnet50_predict_panda():
    expected = "giant panda"
    image = load_image(BASEPATH / "panda.jpeg")
    model = ResNet50()
    label, score = model.predict_label(image)
    assert label == expected
    assert score > 0.5

    processed_img = model.preprocess(image)
    prediction = model.predict(processed_img)
    assert prediction.shape == EXPECTED_SIZE
    prediction.squeeze_(0)
    np.testing.assert_almost_equal(prediction.sum(0).item(), 1.0)
    assert (prediction < 1).all().item()


def test_resnet50_predict_tabby_cat():
    image = load_image(BASEPATH / "tabby_cat.jpeg")
    model = ResNet50()
    label, score = model.predict_label(image)
    assert label == "tabby"
    assert score > 0.3
