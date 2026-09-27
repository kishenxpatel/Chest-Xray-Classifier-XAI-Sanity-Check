"""
Unified interface for generating saliency maps using four methods:
vanilla gradients, guided backpropagation, GradCAM, and integrated gradients.

All functions share a consistent interface:
- Input: a preprocessed image tensor of shape (1, 1, 224, 224), matching
  the output of the preprocessing pipeline in 01_setup.ipynb.
- Output: a 2D numpy array of shape (224, 224), normalised to [0, 1],
  so that outputs from different methods are directly comparable.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import Saliency, GuidedBackprop, IntegratedGradients, LayerGradCam
from captum.attr import LayerAttribution

def compute_vanilla_gradients(model, img: torch.Tensor, target_class: int) -> np.ndarray:
    """
    Compute a vanilla gradient saliency map.

    Uses captum's Saliency method to compute the gradient of the model's
    output for `target_class` with respect to each input pixel.

    Args:
        model: the loaded torchxrayvision model.
        img: preprocessed input tensor, shape (1, 1, 224, 224).
        target_class: index into the model's pathology output vector
            (e.g. via model.pathologies.index('Cardiomegaly')).

    Returns:
        2D numpy array, shape (224, 224), normalised to [0, 1].
    """
    method = Saliency(model)
    img.requires_grad_()
    attribution = method.attribute(img, target=target_class)
    result = attribution.squeeze()
    result = result.detach().numpy()
    normalised_result = (result - result.min()) / (result.max() - result.min())
    return normalised_result

def compute_guided_backprop(model, img: torch.Tensor, target_class: int) -> np.ndarray:
    """
    Compute a guided backpropagation saliency map.

    Uses captum's GuidedBackprop method, which modifies gradient flow
    through ReLU layers to only propagate positive contributions.

    Args:
        model: the loaded torchxrayvision model.
        img: preprocessed input tensor, shape (1, 1, 224, 224).
        target_class: index into the model's pathology output vector.

    Returns:
        2D numpy array, shape (224, 224), normalised to [0, 1].
    """
    method = GuidedBackprop(model)
    img.requires_grad_()
    attribution = method.attribute(img, target=target_class)
    result = attribution.squeeze()
    result = result.detach().numpy()
    normalised_result = (result - result.min()) / (result.max() - result.min())
    return normalised_result
    


def compute_gradcam(model, img: torch.Tensor, target_class: int, target_layer) -> np.ndarray:
    """
    Compute a GradCAM saliency map.

    Uses captum's LayerGradCam method, which uses gradients flowing into
    a chosen convolutional layer (not the raw input pixels) to weight
    that layer's feature maps, then upsamples the result to input resolution.

    Args:
        model: the loaded torchxrayvision model.
        img: preprocessed input tensor, shape (1, 1, 224, 224).
        target_class: index into the model's pathology output vector.
        target_layer: the specific model layer (e.g. model.features.denseblock4)
            whose activations/gradients GradCAM will use.

    Returns:
        2D numpy array, shape (224, 224), normalised to [0, 1]
        (upsampled from the target layer's spatial resolution).
    """
    method = LayerGradCam(model, model.features.denseblock4)
    img.requires_grad_()
    attribution = method.attribute(img, target=target_class)
    upsampled = LayerAttribution.interpolate(attribution, (224, 224))
    result = upsampled.squeeze()
    result = result.detach().numpy()
    normalised_result = (result - result.min()) / (result.max() - result.min())
    return normalised_result


def compute_integrated_gradients(model, img: torch.Tensor, target_class: int, baseline, 
                                 n_steps: int = 50, internal_batch_size: int = 5) -> np.ndarray:
    """
    Compute an integrated gradients saliency map.

    Uses captum's IntegratedGradients method, which averages gradients
    along a straight-line path from a baseline image (e.g. all-black or
    dataset-mean) to the actual input image.

    Args:
        model: the loaded torchxrayvision model.
        img: preprocessed input tensor, shape (1, 1, 224, 224).
        target_class: index into the model's pathology output vector.
        n_steps: number of interpolation steps along the baseline-to-input path.

    Returns:
        2D numpy array, shape (224, 224), normalised to [0, 1].
    """
    method = IntegratedGradients(model)
    img.requires_grad_()
    attribution = method.attribute(img, target=target_class, baselines=baseline, n_steps=n_steps, internal_batch_size=5)
    result = attribution.squeeze()
    result = result.detach().numpy()
    normalised_result = (result - result.min()) / (result.max() - result.min())
    return normalised_result


def overlay_saliency(img: np.ndarray, saliency_map: np.ndarray, alpha: float = 0.4):
    """
    Overlay a saliency map on top of the original image for visualisation.

    Args:
        img: the original (preprocessed or raw) 2D image, shape (224, 224).
        saliency_map: 2D array, shape (224, 224), normalised to [0, 1].
        alpha: blending weight for the saliency overlay (0 = invisible, 1 = opaque).

    Returns:
        A matplotlib-ready RGB(A) array or figure showing the image with
        the saliency map overlaid as a heatmap.
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.imshow(saliency_map, cmap='hot', alpha=alpha)
    ax.axis('off')
    return fig, ax


def compute_dataset_mean_baseline(image_filenames: list[str], data_dir: str) -> torch.Tensor:
    """
    Compute the mean image across a set of preprocessed images, for use as
    the IntegratedGradients baseline.

    Args:
        image_filenames: list of image filenames (not full paths).
        data_dir: directory containing the images.

    Returns:
        A single tensor of shape (1, 1, 224, 224) representing the mean image.
    """
    from src.data import load_and_preprocess_image

    all_images = []
    for filename in image_filenames:
        img = load_and_preprocess_image(f"{data_dir}/{filename}")
        all_images.append(img)

    stacked = torch.cat(all_images, dim=0)  # shape: (N, 1, 224, 224)
    mean_image = stacked.mean(dim=0, keepdim=True)  # shape: (1, 1, 224, 224)

    return mean_image