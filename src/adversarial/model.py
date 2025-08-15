import abc
import logging
from typing import Tuple
import torch
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
from adversarial import Category, Probabilities, Score, TorchImage, TorchImageProcessed

log = logging.getLogger(__name__)


class Model(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "mean")
            and callable(subclass.mean)
            and hasattr(subclass, "std")
            and callable(subclass.std)
            and hasattr(subclass, "category_to_int")
            and callable(subclass.category_to_int)
            and hasattr(subclass, "unpreprocess")
            and callable(subclass.unpreprocess)
            and hasattr(subclass, "preprocess")
            and callable(subclass.preprocess)
            and hasattr(subclass, "predict_label")
            and callable(subclass.predict_label)
            and hasattr(subclass, "predict")
            and callable(subclass.predict)
            or NotImplemented
        )

    @abc.abstractmethod
    def category_to_int(self, category: Category) -> int:
        """Return an integer corresponding to the `category` in the set of all possible categories known by the model.
        Raises ValueError if the category is not known by the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unpreprocess(self, img: TorchImageProcessed) -> TorchImage:
        """Undo the preprocessing applied to an image to get back the original."""
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self, img: TorchImage) -> TorchImageProcessed:
        """Preprocess an image."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mean(self) -> torch.Tensor:
        """The mean of the wights of the model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def std(self) -> torch.Tensor:
        """The std of the wights of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, img: TorchImageProcessed) -> Probabilities:
        """Predict the probabilities for the processed image as a one dimensional Tensor."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_label(
        self, img: TorchImage, category: Category | None = None
    ) -> Tuple[Category, Score]:
        """Predict the label for the image.

        Returs:
            If `category` is `None`, then predict the most likely category for the image.
            Otherwise, returns the score for `category`.
        """

        raise NotImplementedError


class ResNet50(Model):
    def __init__(self, *, device=None) -> None:
        super().__init__()
        log.info("Initialisation of a ResNet50 pretrained model")
        if not device:
            device = torch.device("cpu")
        log.debug("Setting device to: %s", device)
        # from https://docs.pytorch.org/vision/stable/models.html#classification
        weights = ResNet50_Weights.DEFAULT
        self.meta = weights.meta
        self._preprocess = weights.transforms()
        # TODO cant get the weights from my machine using pytorch API, so need to download them manually
        # model = resnet50(weights=weights) # cant run this :(
        path2load = Path(".").expanduser() / "resnet50-11ad3fa6.pth"
        log.info("Loading weights from: %s", path2load)
        sd = torch.load(path2load, map_location="cpu", weights_only=True)
        self.model = resnet50()
        self.model.load_state_dict(sd, strict=True)
        self.model.to(device)
        self.model.eval()

        # freeze weights: we dont want to take the grad of w but of x the image
        log.debug("Freezing the weights of the model")
        for param in self.model.parameters():
            param.requires_grad = False
        log.info("Initialisation of resnet50 completed")

    @property
    def mean(self) -> torch.Tensor:
        return torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

    @property
    def std(self) -> torch.Tensor:
        return torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def predict(self, img: TorchImageProcessed) -> Probabilities:
        # torch expects batches
        return self.model(img).softmax(-1)

    def _prediction(self, img: TorchImage) -> Probabilities:
        return self.predict(self.preprocess(img))

    def preprocess(self, img: TorchImage) -> TorchImageProcessed:
        return self._preprocess(img).unsqueeze(0)

    @torch.no_grad
    def predict_label(
        self, img: TorchImage, category: Category | None = None
    ) -> Tuple[Category, Score]:
        prediction = self._prediction(img)
        if category:
            log.info("Predicting the score of category %s for this image", category)
            score = prediction[self.meta["categories"].index(category)].item()
            return category, Score(score)
        log.info("Predicting most likely category for this image")
        class_id = int(prediction.argmax().item())
        log.debug("Found %s", class_id)
        log.debug("prediction shape %s", prediction.shape)
        score = prediction[:, class_id].item()
        category = Category(self.meta["categories"][class_id])
        log.info(
            "Returning a score of %.2f for the category %s for this image",
            score,
            category,
        )
        return category, Score(score)

    def unpreprocess(self, img: TorchImageProcessed) -> TorchImage:
        """As indicated [here](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights),
        the processing is lossy due to croping and resizing.
        """
        return TorchImage((img * self.std + self.mean).squeeze(0))

    def category_to_int(self, category: Category) -> int:
        return self.meta["categories"].index(category)
