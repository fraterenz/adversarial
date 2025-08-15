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
            hasattr(subclass, "preprocess")
            and callable(subclass.preprocess)
            and hasattr(subclass, "predict_label")
            and callable(subclass.predict_label)
            and hasattr(subclass, "predict")
            and callable(subclass.predict)
            or NotImplemented
        )

    @abc.abstractmethod
    def preprocess(self, img: TorchImage) -> TorchImageProcessed:
        """ "Preprocess an image."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, img: TorchImageProcessed) -> Probabilities:
        """Predict the probabilities for the processed image."""
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
        # TODO deal with device?
        log.info("Initialisation of a ResNet50 pretrained model")
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def predict(self, img: TorchImageProcessed) -> Probabilities:
        with torch.no_grad():
            # torch expects batches
            batch = img.unsqueeze(0)
            return self.model(batch).squeeze(0).softmax(0)

    def _prediction(self, img: TorchImage) -> Probabilities:
        return self.predict(self.preprocess(img))

    def preprocess(self, img: TorchImage) -> TorchImageProcessed:
        return self._preprocess(img)

    def predict_label(
        self, img: TorchImage, category: Category | None = None
    ) -> Tuple[Category, Score]:
        with torch.no_grad():
            prediction = self._prediction(img)
            if category:
                log.info("Predicting the score of category %s for this image", category)
                score = prediction[self.meta["categories"].index(category)].item()
                return category, Score(score)
            log.info("Predicting most likely category for this image")
            class_id = int(prediction.argmax().item())
            score = prediction[class_id].item()
            category = Category(self.meta["categories"][class_id])
            log.info(
                "Returning a score of %.2f for the category %s for this image",
                score,
                category,
            )
            return category, Score(score)
