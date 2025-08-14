from pathlib import Path
import pytest
from hypothesis import given, strategies as st

from adversarial.utils import load_image

PATH = Path(".").absolute()


@pytest.mark.parametrize(
    "wrong_suffix",
    [".pdf", ".txt"],
)
def test_wrong_path2file_suffix(wrong_suffix: str):
    path2file = PATH.with_suffix(str(wrong_suffix))
    with pytest.raises(ValueError):
        load_image(path2file)


@given(st.booleans())
def test_wrong_path2file_file_not_found_png(is_png: bool):
    path2file = PATH.with_suffix(".png" if is_png else ".jpeg")
    with pytest.raises(ValueError):
        load_image(Path(path2file))
