"""Shared RGB preprocessing for prototype training and the desktop inference engine."""
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PREPROCESSING_VERSION = "original_rgb_bilinear_v1"

def resize_rgb(image, size):
    return transforms.functional.resize(image.convert("RGB"), [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)

def build_transform(size, norm_value, mean, std):
    return transforms.Compose([
        transforms.Lambda(lambda image: resize_rgb(image, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda tensor: tensor * (255.0 / norm_value)),
        transforms.Normalize(mean, std),
    ])
