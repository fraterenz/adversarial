import torch

from adversarial.model import ResNet50


def test_resnet50_init():
    my_model = ResNet50()
    torch.testing.assert_close(
        my_model.mean, torch.tensor([0.485, 0.456, 0.406]).view_as(my_model.mean)
    )
    torch.testing.assert_close(
        my_model.std, torch.tensor([0.229, 0.224, 0.225]).view_as(my_model.std)
    )
    assert len(my_model.meta["categories"]) == 1000
