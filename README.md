## Adversarial
[![Tests](https://github.com/fraterenz/adversarial/actions/workflows/app.yml/badge.svg)](https://github.com/fraterenz/adversarial/actions/workflows/app.yml)

Generate adversarial attacks by adding noise to the image such that the model missclassifies it as the desired target class, without making the noise perceptible to a casual human viewer.

## Features
My idea was to be generic over the method to perform the targeted attacks.
Key features:
- being generic such that it can be expanded in the future by implementing new models and new attacks (they are independent):
    - models can be implemented by subclassing `adversarial.model.Model` (here I used `ResNet50`)
    - attacks can be defined by subclassing `adversarial.attack.AdvAttack` (here I have implemented projected gradient descent with L2 and Linf norms)
- informative logging
- accurate unit testing and integration tests

## Usage
There are two main ways to use/test this, one is installing the code as a library and the other is to download the source code and run the integration tests.
The main function to use is `adversarial.attack.adversarial_attack`.

### Option 1
Install the library `adversarial` into a project or an environment with python 3.13 and run a `main.py`.
For example:
1. create a new app with `uv`: `uv init --app test-adv --python 3.13`
2. `cd` into `test-adv`: `cd test-adv`
2. install this library into the project `uv add git+https://github.com/fraterenz/adversarial`
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
    adv_attack = ProjGradLInf(lr=0.1, epsilon=0.05)
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
Cute pictures of pandas will be generated in the folder `/tests/integration/fixtures/`


## Limitations and future steps
Main limitations:
1. works for CPU only
2. work for one image at the time.

What are the future steps? Maybe addressing the limitations and something else?
