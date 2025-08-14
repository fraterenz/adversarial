from pathlib import Path
from torchvision.io import decode_image

from adversarial import TorchImage


def load_image(path2image: Path) -> TorchImage:
    """Load image from `path2image` using pytorch `decode_image` utility.
    Raise `ValueError` if the `path2image` is not a file and if it doesn't end
    with `.jpeg` or `.png`.

    # TODO: create custom errors instead of ValueErrors.

    Returns:
        An image wrapped into a `TorchImage`
    """
    suffix = path2image.suffix
    if suffix not in {".png", ".jpeg"}:
        raise ValueError(
            f"File type {path2image} must be '.png' or '.jpeg', found {suffix} instead"
        )
    if not path2image.is_file():
        raise ValueError(f"Cannot find image {path2image}")
    return TorchImage(decode_image(str(path2image)))
