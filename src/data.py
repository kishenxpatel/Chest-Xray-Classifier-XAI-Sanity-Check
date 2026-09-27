"""
Shared image loading and preprocessing utilities.
"""

import numpy as np
import skimage.io
import skimage.color
import torch
import torchvision
import torchxrayvision as xrv


def load_and_preprocess_image(filepath: str) -> torch.Tensor:
    """
    Load a chest X-ray image from disk and preprocess it into the format
    expected by the torchxrayvision model: normalised, center-cropped,
    resized to 224x224, with batch and channel dimensions added.

    Args:
        filepath: path to a .png image file.

    Returns:
        Preprocessed tensor of shape (1, 1, 224, 224).
    """
    img = skimage.io.imread(filepath)

    if img.ndim == 3:
        img = skimage.color.rgb2gray(img[..., :3])  # drop alpha if present, convert to grayscale

    img = xrv.datasets.normalize(img, 255)
    img = img[None, ...]  # add channel dim: (1, H, W)

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    img = transform(img)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)  # add batch dim: (1, 1, 224, 224)

    return img