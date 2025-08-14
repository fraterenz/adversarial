from pathlib import Path
import pytest

from adversarial.model import ResNet50
from adversarial.utils import load_image

BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")


def test_resnet50_predict_panda():
    image = load_image(BASEPATH / "panda.jpeg")
    model = ResNet50()
    label, score = model.predict_label_with_score(image)
    assert label == "giant panda"
    assert score > 0.5


def test_resnet50_predict_tabby_cat():
    image = load_image(BASEPATH / "tabby_cat.jpeg")
    model = ResNet50()
    label, score = model.predict_label_with_score(image)
    assert label == "tabby"
    assert score > 0.3
