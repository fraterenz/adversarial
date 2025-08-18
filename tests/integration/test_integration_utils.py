from pathlib import Path
from adversarial import TorchImageProcessed
from adversarial.model import ResNet50
from adversarial.utils import load_image, plot_image
import torch
import matplotlib.pyplot as plt

BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")
EXPECTED_SIZE = torch.Size([3, 224, 224])


def test_load_image_tabby_cat():
    assert load_image(BASEPATH / "tabby_cat.jpeg").shape == EXPECTED_SIZE


def test_load_image_panda():
    assert load_image(BASEPATH / "panda.jpeg").shape == EXPECTED_SIZE


def test_plot_panda():
    image = load_image(BASEPATH / "panda.jpeg")
    model = ResNet50()
    image_proc = model.preprocess(image).squeeze(0)
    figure, ax = plt.subplots(1, 1, squeeze=True, figsize=(4, 3))
    plot_image(TorchImageProcessed(image_proc), ax)
    plt.savefig(BASEPATH / "test_plot_panda.png", dpi=800)
    plt.tight_layout()
    plt.close(figure)
