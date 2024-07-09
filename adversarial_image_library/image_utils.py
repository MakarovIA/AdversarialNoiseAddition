import torch
import torchvision.transforms as transforms

from PIL import Image
from adversarial_image_library.model import preprocess_mean, preprocess_std


normalization_transforms = transforms.Normalize(mean=preprocess_mean, std=preprocess_std)

def preprocess_image_without_normalization(image):
    """Preprocess image for ResNet50 model without applying normalization"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return preprocess(image)


def preprocess_image(image):
    """Preprocess image for ResNet50 model"""
    image = preprocess_image_without_normalization(image)
    return normalization_transforms(image)


def save_image(image, file_path):
    """Saves image to """
    image.save(file_path)


def denorm_image(batch):
    """Denormalizes batch if images"""
    mean = preprocess_mean
    if isinstance(mean, list):
        mean = torch.tensor(mean)

    std = preprocess_std
    if isinstance(std, list):
        std = torch.tensor(std)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def create_image_from_tensor(tensor):
    """Create image from tensor"""
    return transforms.ToPILImage()(tensor)