from .attack import fgsm_attack_step
from .image_utils import create_image_from_tensor, normalization_transforms, \
    preprocess_image_without_normalization
from PIL import Image


def add_noise(image_path, target_class: int, epsilon=0.05, n_steps=5):
    """Add noise to the image to be classified as target_class"""
    image = Image.open(image_path).convert('RGB')
    image = preprocess_image_without_normalization(image)
    image = image.unsqueeze(0)
    for _ in range(n_steps):
        image = normalization_transforms(image)
        image = fgsm_attack_step(image, target_class, epsilon)
    return create_image_from_tensor(image.squeeze(0).detach())
