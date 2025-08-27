# Adversarial
[![Tests](https://github.com/fraterenz/adversarial/actions/workflows/app.yml/badge.svg)](https://github.com/fraterenz/adversarial/actions/workflows/app.yml)

Generate adversarial attacks by adding noise to the image such that the model missclassifies it as the desired target class, without making the noise perceptible to a casual human viewer.

## Features
My idea was to be generic over the method to perform the targeted attacks (but only `torchvision` models).
Key features:
- 🧰 being generic such that it can be expanded in the future by implementing new models and new attacks (they are independent):
    - 🤖 models can be implemented by subclassing `adversarial.model.Model` (here I used `ResNet50`)
    - :ninja: attacks can be defined by subclassing `adversarial.attack.AdvAttack` (here I have implemented projected gradient descent with L2 and Linf norms)
- 📝 informative logging
- 🧪 accurate unit testing and integration tests
- 📚 [docs](https://fraterenz.github.io/adversarial/)

This code is generic also over the strategy to perform adverserial attack.
However, as I've only implemented projected gradient descent, I will briefly discuss this method here.

### Projected gradient descent
The idea is to find the adversarial noise by taking many small steps (learning rate $\alpha$) that makes the model favor the target class.
The norm of the noise is constrained by $\epsilon$ such that the noise is small and not perceptible to a casual human viewer.

This is achieved in three main steps:
1. Take a step towards the minimum of the loss $\mathcal{L}$:
```math
    \delta = \delta_0 - \alpha \frac{\nabla_x \mathcal{L}}{\left\lVert \nabla_x \mathcal{L} \right\rVert_p}
```
2. Project $\delta$ to obtain a small noise $\left\lVert \delta \right\rVert_p \leq \epsilon$:
    - $p=\infty$, `ProjGradLInf` clamps each pixel within the range $\[-\epsilon, +\epsilon\]$
    - $p=2$, `ProjGradL2` rescales as $\delta = \epsilon  \delta / \left\lVert \delta \right\rVert_2$.
3. Keep valid pixel values (within 0 and 1 here).

We repeat this procedure until either the model switches from the original prediction to the target class or a maximal number of `steps` have been taken (set to default in `adversarial_attack()` to 100).

## Usage
There are two main ways to use/test this, one is installing the code as a library and the other is to download the source code and run the integration tests.
The main function to use is `adversarial.attack.adversarial_attack`.

### Option 1
Install the library `adversarial` into a project or an environment with python 3.13 and run a `main.py`.
For example:
1. create a new app with `uv`: `uv init --app test-adv --python 3.13`
2. `cd` into `test-adv`: `cd test-adv`
2. install this library into the project `uv add git+https://github.com/fraterenz/adversarial --tag v1.0.2`
3. download the pretrained model manually: `curl -O -L "https://www.dropbox.com/scl/fi/3cfjlzp4ls8n5imtfe51d/resnet50-11ad3fa6.pth?rlkey=zxaaj95mzlsd4tv7vjos0kwc5&st=om7rfwgo&dl=0"`
4. use the library: create a `main.py` and copy-paste the code below replacing `/path/to/image` with the folder where the image is stored.
5. run main `uv run main.py log_cli=true --log-cli-level=INFO`

An example of a `main.py` performing the attack that can be run with `uv run main.py log_cli=true --log-cli-level=INFO`:
```python
from pathlib import Path
from adversarial import Category, model
from adversarial.attack import (
    ProjGradLInf,
    adversarial_attack,
)
from adversarial.utils import load_image
import logging

logging.basicConfig(level=logging.INFO)


def main():
    logging.info("Running adversarial attack!")
    path2img = Path("/path/to/image")
    image = load_image(path2img / "panda.jpeg")
    # projected gradient with norm 2
    adv_attack = ProjGradLInf(lr=0.05, epsilon=0.01)
    result = adversarial_attack(
        image,
        Category("tabby"),
        model.ResNet50(),
        adv_attack,
    )
    result.plot(path2img / "panda_attack_into_tabby.png")
    logging.info("End adversarial attack!")


if __name__ == "__main__":
    main()
```

### Option 2
Download source code, the pretrained model weights and run tests with `pytest`
1. `git clone git@github.com:fraterenz/adversarial.git`
3. download the pretrained model manually: `curl -O -L "https://www.dropbox.com/scl/fi/3cfjlzp4ls8n5imtfe51d/resnet50-11ad3fa6.pth?rlkey=zxaaj95mzlsd4tv7vjos0kwc5&st=om7rfwgo&dl=0"`
3. `uv run pytest`

Cute pictures of giant pandas will be generated in the folder `/tests/integration/fixtures/`.
Have a look in particular at the test `test_adversarial_attack` in `test/integration/test_integration_attack.py` and the output it generates:
  - `panda_attack_l2_into_gibbon.png`
  - `panda_attack_l2_into_tabby.png`
  - `panda_attack_lInf_into_gibbon.png`
  - `panda_attack_lInf_into_tabby.png`


## Limitations and future steps
Main limitations:
1. works for CPU only
2. work for one image at the time.
3. need to trail-and-error manually the learning rate `lr` and the norm noise `epsilon`. Ideally, find a way to automatically tune that.
4. use a nicer optimiser that is already provided (ADAM?), not manual implementation of SGD.
5. being completely generic over the model, i.e. not only torch vision models, should be easy to implement. We would need to refactor `adversarial_attack` to take also an update strategy. This update stragegy would do its things, such as computing the backward pass for torch.
