import logging
import numpy as np
from pathlib import Path
from torchvision.io import decode_image
import torchvision.transforms.functional as F

from adversarial import TorchImage

log = logging.getLogger(__name__)


def load_image(path2image: Path) -> TorchImage:
    """Load image from `path2image` using pytorch `decode_image` utility.
    Raise `ValueError` if the `path2image` is not a file and if it doesn't end
    with `.jpeg` or `.png`.

    # TODO: create custom errors instead of ValueErrors.

    Returns:
        An image wrapped into a `TorchImage`
    """
    log.info("Loading image from %s", path2image)
    suffix = path2image.suffix
    if suffix not in {".png", ".jpeg"}:
        raise ValueError(
            f"File type {path2image} must be '.png' or '.jpeg', found {suffix} instead"
        )
    if not path2image.is_file():
        raise ValueError(f"Cannot find image {path2image}")
    img = TorchImage(decode_image(str(path2image)))
    log.info("Image loaded from %s", path2image)
    return img


def plot_image(img: TorchImage, ax, **imshow_kwargs):
    img = F.to_pil_image(img.to("cpu"))
    ax.imshow(np.asarray(img), **imshow_kwargs)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return ax
