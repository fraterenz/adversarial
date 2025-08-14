from pathlib import Path
from adversarial.utils import load_image
import torch

BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")
EXPECTED_SIZE = torch.Size([3, 224, 224])

def test_load_image_tabby_cat():
    assert load_image(BASEPATH / "tabby_cat.jpeg").shape == EXPECTED_SIZE

def test_load_image_panda():
    assert load_image(BASEPATH / "panda.jpeg").shape == EXPECTED_SIZE

