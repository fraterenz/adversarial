import abc
import logging
from typing import Tuple
import torch
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
from adversarial import Label, TorchImage

log = logging.getLogger(__name__)


class Model(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "") and callable(subclass.load_dataset) or NotImplemented
        )

    @abc.abstractmethod
    def predict_label(self, img: TorchImage) -> Label:
        """Predict the label for the image"""

        raise NotImplementedError


class ResNet50(Model):
    def __init__(self, *, device=None) -> None:
        super().__init__()
        # TODO deal with device?
        log.info("Initialisation of a ResNet50 pretrained model")
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.debug("Setting device to: ", device)
        # from https://docs.pytorch.org/vision/stable/models.html#classification
        weights = ResNet50_Weights.DEFAULT
        self.meta = weights.meta
        self.preprocess = weights.transforms()
        # TODO cant get the weights from my machine using pytorch API, so need to download them manually
        # model = resnet50(weights=weights) # cant run this :(
        path2load = Path(".").expanduser() / "resnet50-11ad3fa6.pth"
        log.info("Loading weights from: ", path2load)
        sd = torch.load(path2load, map_location="cpu", weights_only=True)
        self.model = resnet50()
        self.model.load_state_dict(sd, strict=True)
        self.model.to(device)
        self.model.eval()

        # freeze weights: we dont want to take the grad of w but of x the image
        log.debug("Freezing the weights of the model")
        for param in self.model.parameters():
            param.requires_grad = False
        log.info("end initialisation of resnet50")


    def _prediction(self, img: TorchImage):
        batch = self.preprocess(img).unsqueeze(0)
        prediction = self.model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        return prediction, class_id

    def predict_label(self, img: TorchImage) -> Label:
        prediction, class_id = self._prediction(img)
        return self.meta["categories"][class_id]

    def predict_label_with_score(self, img: TorchImage) -> Tuple[Label, float]:
        prediction, class_id = self._prediction(img)
        score = prediction[class_id].item()
        return self.meta["categories"][class_id], score
