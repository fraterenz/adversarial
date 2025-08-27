import math
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
EPSILON = 0.05
LR = 0.08


def prepare_data(panda: bool = True):
    image = (
        load_image(BASEPATH / "panda.jpeg")
        if panda
        else load_image(BASEPATH / "tabby_cat.jpeg")
    )
    model = ResNet50()
    image_proc = model.preprocess(image)
    return model, image, image_proc


def test_adversarial_attack_wrong():
    model, image, image_proc = prepare_data()
    category, score = model.predict_label(image_proc)
    with pytest.raises(ValueError):
        adversarial_attack(
            image, Category("yo"), model, ProjGradLInf(lr=LR, epsilon=EPSILON)
        )


def test_adversarial_attack():
    for category, target in [
        #("tabby", "giant panda"),
        ("tabby", "gibbon"),
        ("giant panda", "tabby"),
        ("giant panda", "gibbon"),
    ]:
        # the eps for need for l2 norm perturbations is larger than what you need
        # for Linf perturbations, because the volume of the L2 ball is proportional
        # to sqrt(D) times the volume of the Linf ball, D is dimension
        if category == "tabby":
            model, image, image_proc = prepare_data(False)
        else:
            model, image, image_proc = prepare_data(True)
        D = image.numel()
        category_predicted, score = model.predict_label(image_proc)
        assert category_predicted == category
        for norm_type, eps in [("l2", EPSILON * math.sqrt(D)), ("lInf", EPSILON)]:
            result = adversarial_attack(
                image,
                Category(target),
                model,
                ProjGradL2(lr=LR, epsilon=eps)
                if norm_type == "l2"
                else ProjGradLInf(lr=LR, epsilon=eps),
                steps=150,
            )
            assert result.adv_target == result.adv_prediction
            result.plot(BASEPATH / f"{category}_attack_{norm_type}_into_{target}.png")


def test_adversarial_attack_no_attack():
    for norm_type in ["l2", "lInf"]:
        model, image, image_proc = prepare_data()
        category, score = model.predict_label(image_proc)
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
    for norm_type in ["l2", "lInf"]:
        model, image, image_proc = prepare_data()
        strategy = (
            ProjGradL2(lr=0.025, epsilon=0.01)
            if norm_type == "l2"
            else ProjGradLInf(lr=0.025, epsilon=0.01)
        )
        pert = strategy.initialise_perturbation(image_proc)
        img_noisy = noisy_image(image_proc, pert)
        img_noisy = TorchImageProcessed(img_noisy - pert.pert)
        result = AdvResult(
            img_noisy,
            Category("giant panda"),
            Category("small panda"),
            0.4,
            image_proc,
            Category("giant panda"),
            0.6,
        )
        result.plot(BASEPATH / f"panda_test_noisy_image_denoised_{norm_type}.png")


def test_result_plot_noisy_image():
    for norm_type in ["l2", "lInf"]:
        model, image, image_proc = prepare_data()
        strategy = (
            ProjGradL2(lr=0.025, epsilon=0.01)
            if norm_type == "l2"
            else ProjGradLInf(lr=0.025, epsilon=0.01)
        )
        pert = strategy.initialise_perturbation(image_proc)
        img_noisy = noisy_image(image_proc, pert)
        result = AdvResult(
            image_proc,
            Category("giant panda"),
            Category("small panda"),
            0.4,
            img_noisy,
            Category("giant panda"),
            0.6,
        )
        result.plot(BASEPATH / f"panda_test_noisy_image_{norm_type}.png")


def test_result_plot_preprocessing():
    model, image, image_proc = prepare_data()
    result = AdvResult(
        image_proc,
        Category("giant panda"),
        Category("small panda"),
        0.4,
        image_proc,
        Category("giant panda"),
        0.6,
    )
    result.plot(BASEPATH / "panda_test_preprocessing.png")


def test_result_plot_prediction():
    model, image, image_proc = prepare_data()
    category, score = model.predict_label(image_proc)
    result = AdvResult(
        image_proc,
        Category("giant panda"),
        category,
        score,
        image_proc,
        category,
        score,
    )
    result.plot(BASEPATH / "panda_test_predict_label.png")


def test_result_plot():
    model, image, image_proc = prepare_data()
    result = AdvResult(
        image_proc,
        Category("giant panda"),
        Category("small panda"),
        0.4,
        image_proc,
        Category("giant panda"),
        0.6,
    )
    result.plot(BASEPATH / "panda_plot_test.png")
