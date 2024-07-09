import torch

from .model import target_model
from .image_utils import denorm_image


def fgsm_attack_step(image, target_class, epsilon=0.05):
    """Apply 1 step of FGSM attack"""
    image.requires_grad = True

    # Forward pass
    output = target_model(image)
    loss = torch.nn.functional.cross_entropy(output, torch.tensor([target_class]))

    # Backward pass
    target_model.zero_grad()
    loss.backward()

    # Generate adversarial image
    data_grad = image.grad.data
    image_denorm = denorm_image(image)
    image = fgsm_attack_transform(image_denorm, epsilon, data_grad)
    image = image.detach()
    return image


def fgsm_attack_transform(image, epsilon, data_grad):
    """Transform image according to FGSM attack"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
