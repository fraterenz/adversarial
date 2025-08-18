"""This library manipulates images by adding adversarial noise, designed to
trick an image classification model into misclassifying the altered image as a
specified target class, regardless of the original content.

The main function to have a look at is `adversarial.attack.adversarial_attack`.
"""

from typing import NewType
import logging
from torch import Tensor


TorchImageProcessed = NewType("TorchImageProcessed", Tensor)
"""A preprocessed image (pixels are within 0 and 1) for which we can compute a prediction."""
TorchImage = NewType("TorchImage", Tensor)
"""An image loaded with `torchvision.io.decode_image` pixels are usually within 0 and 255."""
Probabilities = NewType("Probabilities", Tensor)
"""The output of a prediction for a processed image, indicating the probabilty for the image to belong to each category."""
Score = NewType("Score", float)
"""The score for an image, i.e. the probability that the image belongs to a category."""
Category = NewType("Category", str)
"""The category for an Image."""


logging.getLogger(__name__).addHandler(logging.NullHandler())
