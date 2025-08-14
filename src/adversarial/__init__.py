from typing import NewType
import logging
from torch import Tensor


TorchImage = NewType("TorchImage", Tensor)
Label = NewType("Label", str)


logging.getLogger(__name__).addHandler(logging.NullHandler())
