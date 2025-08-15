import abc
from dataclasses import dataclass
from functools import partial
import logging
from typing import Callable

from torch import Tensor, linalg
import torch.nn as nn
import torch

from adversarial import TorchImageProcessed

log = logging.getLogger(__name__)


class Perturbation:
    """A Tensor of the same dimensions as a `TorchImageProcessed` storing the perturbation to apply to the image to get a noisy image."""

    def __init__(self, data: Tensor):
        log.debug("Creating a noisy perturbation")
        self.pert = nn.Parameter(data.detach(), requires_grad=True)
        if self.pert.grad is not None:
            self.pert.grad.zero_()
            log.debug("Setting grad to zero")
        log.debug("Noisy perturbation created")


def noisy_image(
    img: TorchImageProcessed, perturbation: Perturbation
) -> TorchImageProcessed:
    """Add additive noise to an processed image (pixels within 0 and 1)."""
    log.info("Creating a noisy image")
    # need to clamp to keep the pixels within 0 and 1 as we are dealing with a processed image
    return TorchImageProcessed(torch.clamp(img + perturbation.pert.detach(), 0, 1))


@dataclass
class AdvAttack(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "perturb")
            and callable(subclass.perturb)
            or NotImplemented
        )

    @abc.abstractmethod
    def perturb(self, perturbation: Perturbation) -> Perturbation:
        """Compute a perturbation that can be later added to an image."""
        raise NotImplementedError


@dataclass
class ProjGrad(AdvAttack):
    """Relative gradient ascent method projected into the Lp norm, a ball of radius `epsilon`.

    Reference: Adversarial Robustness - Theory and Practice, [Chap3](https://adversarial-ml-tutorial.org/adversarial_examples/).
    """

    lr: float
    """Learning rate."""
    epsilon: float
    """The max norm value of the perturbation."""
    norm: Callable[[Tensor], Tensor]
    """A function that returns a norm."""
    projection: Callable[[Tensor], Tensor]
    """A function that projects the input tensor into the Lp volume."""

    def perturb(self, perturbation: Perturbation) -> Perturbation:
        log.debug(
            "Stepping into the loss landscape according to a projected gradient ascent"
        )
        with torch.no_grad():
            pert = perturbation.pert
            grad = pert.grad
            if grad is not None:
                grad = grad.data
            else:
                raise RuntimeError("Cannot access the gradient of the perturbation")
            log.debug("Starting from %s", pert)
            # clamp to avoid numerical instabilities i.e. division by 0
            step = self.lr * grad / self.norm(grad).clamp_min(1e-12)
            # + because we are ascending not descending
            pert += step
            log.debug("Taking a step forward %s landing on %s", step, pert)

            # project onto lp: if ||pert||>ε scale<1 else 1
            scale = self.epsilon / self.norm(pert).clamp_min(1e-12)
            pert *= self.projection(scale)
            log.debug("Rescaling the gradient by %s and projecting it %s", scale, pert)

        log.debug("One step completed")
        return Perturbation(pert)


class ProjGradLInf(ProjGrad):
    """Relative gradient ascent method projected into the Linfinite norm, a volume of size `epsilon`^D.

    Reference: Adversarial Robustness - Theory and Practice, [Chap3](https://adversarial-ml-tutorial.org/adversarial_examples/).
    """

    lr: float
    """Learning rate."""
    epsilon: float
    """The max norm value of the perturbation."""

    def __init__(self, lr: float, epsilon: float):
        super().__init__(
            lr,
            epsilon,
            partial(linalg.vector_norm, ord=2, dim=(1, 2, 3), keepdim=True),
            partial(torch.clamp, min=-self.epsilon, max=self.epsilon),
        )


class ProjGradL2(ProjGrad):
    """Relative gradient ascent method projected into the L2 norm, a ball of radius `epsilon`.

    Reference: Adversarial Robustness - Theory and Practice, [Chap3](https://adversarial-ml-tutorial.org/adversarial_examples/).
    """

    lr: float
    """Learning rate."""
    epsilon: float
    """The max norm2 value of the perturbation."""

    def __init__(self, lr: float, epsilon: float):
        super().__init__(
            lr,
            epsilon,
            partial(linalg.vector_norm, ord=2, dim=(1, 2, 3), keepdim=True),
            partial(torch.clamp, max=1),
        )
