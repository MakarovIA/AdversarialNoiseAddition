# Adversarial Noise Addition

The repository contains a Python library designed to manipulate images by adding adversarial noise. The goal is to trick an image classification model, specifically a `ResNet-50`, into misclassifying the altered image as a specified target class, regardless of its original content.

* Input: path to image, target class
* Output: `PIL.Image` that 

## Installation

```
git clone https://github.com/MakarovIA/AdversarialNoiseAddition.git
cd AdversarialNoiseAddition/
pip install -r requirements.txt
```

## Usage

The script below add noise to the given image file and shows the result:

```
from adversarial_image_library import add_noise

image_path = 'examples/data/panda.jpg'
target_class = 355

adversarial_image = add_noise(image_path, target_class)
adversarial_image.show()
```

## Simple example

* The [example file](./examples/example_addition.py) applies noise addition to the specified image file, saves and shows resulting image:
```
cd examples/
python example_addition.py
```

## Tests
* The [test file](./examples/example_addition.py) applies noise addition to the images in [tests/data directory](./tests/data) and checks that the model predicts predefined class for them:
```
cd tests/
python test_noise_addition.py
```

