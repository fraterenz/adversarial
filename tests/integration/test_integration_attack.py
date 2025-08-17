import pytest
from pathlib import Path


from adversarial import Category, TorchImageProcessed
from adversarial.attack import (
    AdvResult,
    ProjGradL2,
    ProjGradLInf,
    adversarial_attack,
    noisy_image,
)
from adversarial.model import ResNet50
from adversarial.utils import load_image

BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")
EPSILON = 0.1
LR = 0.2


def test_adversarial_attack_wrong():
    image = load_image(BASEPATH / "panda.jpeg")
    model = ResNet50()
    category, score = model.predict_label(image)
    with pytest.raises(ValueError):
        adversarial_attack(
            image, Category("yo"), model, ProjGradLInf(lr=0.025, epsilon=EPSILON)
        )


def test_adversarial_attack():
    image = load_image(BASEPATH / "panda.jpeg")
    model = ResNet50()
    category, score = model.predict_label(image)
    for norm_type in ["l2", "lInf"]:
        # assert result.adv_target == result.adv_prediction
        result = adversarial_attack(
            image,
            Category("tabby"),
            model,
            ProjGradL2(lr=LR, epsilon=EPSILON)
            if norm_type == "l2"
            else ProjGradLInf(lr=LR, epsilon=EPSILON),
        )
        result.plot(BASEPATH / f"panda_attack_{norm_type}_into_tabby.png")


def test_adversarial_attack_no_attack():
    image = load_image(BASEPATH / "panda.jpeg")
    model = ResNet50()
    category, score = model.predict_label(image)
    for norm_type in ["l2", "lInf"]:
        result = adversarial_attack(
            image,
            category,
            model,
            ProjGradL2(lr=LR, epsilon=EPSILON)
            if norm_type == "l2"
            else ProjGradLInf(lr=LR, epsilon=EPSILON),
        )
        assert result.adv_target == result.adv_prediction == result.original_category
        result.plot(BASEPATH / f"panda_no_attack_{norm_type}.png")


def test_result_plot_noisy_image_add_rm_noise():
    image = load_image(BASEPATH / "panda.jpeg")
    model = ResNet50()
    for norm_type in ["l2", "lInf"]:
        strategy = (
            ProjGradL2(lr=0.025, epsilon=0.01)
            if norm_type == "l2"
            else ProjGradLInf(lr=0.025, epsilon=0.01)
        )
        img_processed = model.preprocess(image)
        pert = strategy.initialise_perturbation(img_processed)
        img_noisy = noisy_image(img_processed, pert, model.mean, model.std)
        img_noisy = TorchImageProcessed(img_noisy - pert.pert)
        img_unprocessed = model.unpreprocess(img_noisy)
        result = AdvResult(
            img_unprocessed,
            Category("giant panda"),
            Category("small panda"),
            0.4,
            image,
            Category("giant panda"),
            0.6,
        )
        result.plot(BASEPATH / f"panda_test_noisy_image_denoised_{norm_type}.png")


def test_result_plot_noisy_image():
    image = load_image(BASEPATH / "panda.jpeg")
    model = ResNet50()
    for norm_type in ["l2", "lInf"]:
        strategy = (
            ProjGradL2(lr=0.025, epsilon=0.01)
            if norm_type == "l2"
            else ProjGradLInf(lr=0.025, epsilon=0.01)
        )
        img_processed = model.preprocess(image)
        pert = strategy.initialise_perturbation(img_processed)
        img_noisy = noisy_image(img_processed, pert, model.mean, model.std)
        img_unprocessed = model.unpreprocess(img_noisy)
        result = AdvResult(
            img_unprocessed,
            Category("giant panda"),
            Category("small panda"),
            0.4,
            image,
            Category("giant panda"),
            0.6,
        )
        result.plot(BASEPATH / f"panda_test_noisy_image_{norm_type}.png")


def test_result_plot_preprocessing():
    image = load_image(BASEPATH / "panda.jpeg")
    model = ResNet50()
    img_processed = model.preprocess(image)
    img_unprocessed = model.unpreprocess(img_processed)
    result = AdvResult(
        img_unprocessed,
        Category("giant panda"),
        Category("small panda"),
        0.4,
        image,
        Category("giant panda"),
        0.6,
    )
    result.plot(BASEPATH / "panda_test_preprocessing.png")


def test_result_plot_prediction():
    image = load_image(BASEPATH / "panda.jpeg")
    model = ResNet50()
    category, score = model.predict_label(image)
    result = AdvResult(
        image,
        Category("giant panda"),
        category,
        score,
        image,
        category,
        score,
    )
    result.plot(BASEPATH / "panda_test_predict_label.png")


def test_result_plot():
    image = load_image(BASEPATH / "panda.jpeg")
    result = AdvResult(
        image,
        Category("giant panda"),
        Category("small panda"),
        0.4,
        image,
        Category("giant panda"),
        0.6,
    )
    result.plot(BASEPATH / "panda_plot_test.png")
