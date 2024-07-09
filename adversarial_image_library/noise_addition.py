from .attack import fgsm_attack_step
from .image_utils import create_image_from_tensor, normalization_transforms, \
    preprocess_image_without_normalization
from PIL import Image


def add_noise(image_path: str, target_class: int, epsilon: float=0.05, n_steps: int=5):
    """
    Main function that add noise to the image to be classified as target_class

    Args:
        image_path (str): path to file with image to add noise for
        epsilon (float): parameter of FGSM attack that impacts on the degree of image modification
        n_steps (int): number of steps to apply modifications

    Returns:
        PIL.Image: resulting image after adding noise to the original one
    """
    image = Image.open(image_path).convert('RGB')
    image = preprocess_image_without_normalization(image)
    image = image.unsqueeze(0)
    for _ in range(n_steps):
        image = normalization_transforms(image)
        image = fgsm_attack_step(image, target_class, epsilon)
    return create_image_from_tensor(image.squeeze(0).detach())
