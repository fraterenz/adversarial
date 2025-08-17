import abc
from dataclasses import dataclass
from functools import partial
import logging
from adversarial.utils import plot_image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable

from torch import Tensor, linalg
import torch.nn as nn
import torch

from adversarial import Category, TorchImage, TorchImageProcessed
from adversarial.model import Model

log = logging.getLogger(__name__)


class Perturbation:
    """A Tensor of the same dimensions as a `TorchImageProcessed` storing the perturbation to apply to the image to get a noisy image."""

    def __init__(self, data: Tensor):
        log.debug("Creating a noisy perturbation")
        self.pert = nn.Parameter(data.detach(), requires_grad=True)
        if self.pert.grad is not None:
            self.pert.grad = None
            log.debug("Setting grad to zero")
        log.debug("Noisy perturbation created")


def noisy_image(
    img: TorchImageProcessed, perturbation: Perturbation, mean: Tensor, std: Tensor
) -> TorchImageProcessed:
    """Add additive noise to an processed image.

    To return a `TorchImageProcessed` we need to keep the pixels with 0 mean and std of 1,
    hence with need to know `mean` and `std`, and both should have the same shape
    as the `img`.
    """
    # in resnet50, values are first rescaled to [0.0, 1.0] and then normalized
    # using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
    # To keep them in this range, we need to clip values.
    # The lower and upper bounds are determined by the rescaling followed by
    # normalisation.
    # Althoug this is specific to this model, I think we can have other models
    # preprocessing the images in this way, and thus this function might be used.
    low = (0.0 - mean) / std
    high = (1.0 - mean) / std
    log.debug("Creating a noisy image")
    return TorchImageProcessed((img + perturbation.pert).clamp(low, high))


@dataclass
class AdvAttack(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "initialise_perturbation")
            and callable(subclass.initialise_perturbation)
            and hasattr(subclass, "perturb_")
            and callable(subclass.perturb_)
            or NotImplemented
        )

    @abc.abstractmethod
    def initialise_perturbation(self, like: TorchImageProcessed) -> Perturbation:
        """Create the first perturbation of same shape as `like` from random data and projecting them onto the norm ball."""
        raise NotImplementedError

    @abc.abstractmethod
    def perturb_(self, perturbation: Perturbation):
        """Compute a perturbation in place that can be later added to an image."""
        raise NotImplementedError


@dataclass
class ProjGrad(AdvAttack):
    """Relative gradient descent method projected into the Lp norm, a ball of radius `epsilon`.

    Reference: Adversarial Robustness - Theory and Practice, [Chap3](https://adversarial-ml-tutorial.org/adversarial_examples/).
    """

    lr: float
    """Learning rate."""
    epsilon: float
    """The max norm value of the perturbation."""
    norm: Callable[[Tensor], Tensor]
    """A function that returns a norm."""

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "projection_")
            and callable(subclass.projection_)
            or NotImplemented
        )

    @abc.abstractmethod
    def projection_(self, v: Tensor):
        """Projects the input tensor into the Lp volume by modyfing the tensor inplace."""
        raise NotImplementedError

    def initialise_perturbation(self, like: TorchImageProcessed) -> Perturbation:
        rand = torch.empty_like(like.data).uniform_(
            -self.epsilon / 10, self.epsilon / 10
        )
        self.projection_(rand)
        perturbation = Perturbation(rand)
        log.info(
            "Initialised a perturbation with random noise with shape %s with norm 2 %s < %s",
            perturbation.pert.shape,
            linalg.vector_norm(
                perturbation.pert,
                ord=torch.inf,
                keepdim=False,
            ).item(),
            self.epsilon,
        )
        log.debug("%s", perturbation.pert)
        return perturbation

    @torch.no_grad
    def perturb_(self, perturbation: Perturbation):
        log.debug(
            "Stepping into the loss landscape according to a projected gradient descent"
        )
        pert = perturbation.pert
        grad = pert.grad
        if grad is None:
            raise RuntimeError("Cannot access the gradient of the perturbation")
        grad = grad.detach()
        log.debug("Starting from a perturbation with shape %s", pert.shape)
        log.debug("gradient shape: %s, ndim %s", grad.shape, grad.ndim)
        step = self.lr * grad / self.norm(grad).clamp_min(1e-12)
        pert.sub_(step)
        log.debug(
            "Taking a step forward of shape %s landing on a tensor of shape %s",
            step.shape,
            pert.shape,
        )

        # project onto lp
        self.projection_(pert)
        log.debug(
            "Projecting the updated perturbation with shape %s onto the Lp norm ball",
            pert.shape,
        )
        pert.grad = None

        log.debug("One step completed")


class ProjGradLInf(ProjGrad):
    """Relative gradient descent method projected into the Linfinite norm, a ball of radius epsilon.

    Reference: Adversarial Robustness - Theory and Practice, [Chap3](https://adversarial-ml-tutorial.org/adversarial_examples/).
    """

    lr: float
    """Learning rate."""
    epsilon: float
    """The max norm value of the perturbation aka the radius of the ball of norm Linf."""

    def __init__(self, lr: float, epsilon: float):
        super().__init__(
            lr,
            epsilon,
            partial(linalg.vector_norm, ord=torch.inf, dim=(1, 2, 3), keepdim=True),
        )

    def projection_(self, v: Tensor):
        """Project `v` into a Linf ball."""
        v.clamp_(-self.epsilon, self.epsilon)


class ProjGradL2(ProjGrad):
    """Relative gradient descent method projected into the L2 norm, a ball of radius `epsilon`.

    Reference: Adversarial Robustness - Theory and Practice, [Chap3](https://adversarial-ml-tutorial.org/adversarial_examples/).
    """

    lr: float
    """Learning rate."""
    epsilon: float
    """The max norm value of the perturbation aka the radius of the ball of norm L2."""

    def __init__(self, lr: float, epsilon: float):
        super().__init__(
            lr,
            epsilon,
            partial(linalg.vector_norm, ord=2, dim=(1, 2, 3), keepdim=True),
        )

    def projection_(self, v: Tensor):
        """Project `v` into a ball with radius epsilon."""
        # project onto l2: if ||v||>eps scale<1 else 1
        # first clamp to avoid numerical instabilities i.e. division by 0
        scale = (self.epsilon / self.norm(v).clamp_min(1e-12)).clamp(max=1)
        v.mul_(scale)


@dataclass
class AdvResult:
    adv_img: TorchImage
    adv_target: Category
    adv_prediction: Category
    adv_prob: float
    original_img: TorchImage
    original_category: Category
    original_prob: float

    def plot(self, path2save: Path):
        log.info("Saving the results as a plot in %s", path2save)
        figure, axes = plt.subplots(
            nrows=1, ncols=2, squeeze=False, figsize=(8, 6), sharex=True
        )
        axes = axes.squeeze()
        log.debug("%s", axes)
        for ax, img in zip(axes, [self.original_img, self.adv_img]):
            plot_image(img, ax)
        axes[0].set_title(f"{self.original_category}:{self.original_prob:.2f}")
        axes[1].set_title(f"{self.adv_prediction}:{self.adv_prob:.2f}")
        figure.suptitle(
            f"Original (left) adversarial (right) with target class {self.adv_target}\nthe model after an adv attack predicts this image as {self.adv_prediction}\non top of the images the <category>:<probability> are shown"
        )
        plt.savefig(path2save, dpi=800)
        plt.tight_layout()
        plt.close(figure)
        assert path2save.is_file()
        log.info("Figure closed and saved")


def adversarial_attack(
    img: TorchImage,
    target: Category,
    model: Model,
    strategy: AdvAttack,
    steps: int = 20,
) -> AdvResult:
    """Add adversial noise to an image."""

    def save_results():
        log.info("End of the attack after %d iter, storing result", step)
        adv_img = noisy_image(processed_img, perturbation, model.mean, model.std)
        adv_img = model.unpreprocess(adv_img)
        log.debug("the adversarial image has size %s", adv_img.shape)
        adv_prediction, adv_prob = model.predict_label(adv_img)
        return AdvResult(
            adv_img,
            target,
            adv_prediction,
            adv_prob,
            img,
            original_category,
            original_prob,
        )

    log.info(
        "Running the adversial attack on an image with shape %s to target a %s",
        img.shape,
        target,
    )
    # batch the target to get consistent size with the loss
    target_int = torch.tensor(
        model.category_to_int(target), dtype=torch.long, requires_grad=False
    ).unsqueeze(0)
    log.debug(
        "found integer %d corresponding to the target category %s of the adversial attack",
        target_int,
        target,
    )

    log.info("Using the cross entropy loss")
    criterion = torch.nn.CrossEntropyLoss()
    log.debug("preprocessing the image with shape %s", img.shape)
    processed_img = model.preprocess(img)
    log.debug("after preprocessing the image with shape %s", processed_img.shape)
    log.debug("initialisation of the first perturbation to random data")
    perturbation = strategy.initialise_perturbation(processed_img)
    log.debug("perturbation shape %s", perturbation.pert.shape)
    original_category, original_prob = model.predict_label(img)
    log.info(
        "The model predicts original category %s with score %.2f",
        original_category,
        original_prob,
    )

    log.info("Starting the attack with %s", strategy)
    for step in range(steps):
        adv_img = noisy_image(processed_img, perturbation, model.mean, model.std)
        log.debug("noisy image has shape %s", adv_img.shape)
        logits = model.predict(adv_img)
        log.debug("logits: shape %s", logits.shape)
        predicted_int = torch.argmax(logits, dim=-1)
        log.debug("target_int %s with shape %s", target_int, target_int.shape)
        loss = criterion(logits, target_int)
        log.debug("%s", loss)
        loss.backward()

        strategy.perturb_(perturbation)
        log.debug("%d %s", target_int, predicted_int)
        if target_int == predicted_int:
            log.info("Predicted the target category: %s", target)
            return save_results()
        log.debug("%d %s", step, step)
    log.info("Max iter reached")
    return save_results()
