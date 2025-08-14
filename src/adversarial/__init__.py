from typing import NewType

from torch import Tensor


TorchImage = NewType("TorchImage", Tensor)


def hello() -> str:
    return "Hello from adversarial!"
