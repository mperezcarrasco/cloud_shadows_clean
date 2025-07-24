from typing import Tuple, List
import numpy as np
import random


def center_crop(
    data: np.ndarray, mask: np.ndarray, crop_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a center crop on hyperspectral data and corresponding mask.

    Args:
        data (np.ndarray): Hyperspectral data with shape (H, W, C).
        mask (np.ndarray): Mask corresponding to the data with shape (H, W).
        crop_size (int): The desired crop size (e.g., 448).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cropped data and mask.
    """
    h, w, _ = data.shape
    ch, cw = crop_size, crop_size

    # Ensure the crop size is smaller than the dimensions of the input
    if h < ch or w < cw:
        raise ValueError("Crop size must be smaller than data dimensions.")

    top = (h - ch) // 2
    left = (w - cw) // 2

    data_cropped = data[top : top + ch, left : left + cw, :]
    mask_cropped = mask[top : top + ch, left : left + cw]
    return data_cropped, mask_cropped


def random_crop(
    data: np.ndarray, mask: np.ndarray, crop_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a random crop on hyperspectral data and corresponding mask.

    Args:
        data (np.ndarray): Hyperspectral data with shape (H, W, C).
        mask (np.ndarray): Mask corresponding to the data with shape (H, W).
        crop_size (int): The desired crop size (e.g., 448).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cropped data and mask.
    """
    h, w, _ = data.shape
    ch, cw = crop_size, crop_size

    # Ensure the crop size is smaller than the dimensions of the input
    if h < ch or w < cw:
        raise ValueError("Crop size must be smaller than data dimensions.")

    top = random.randint(0, h - ch)
    left = random.randint(0, w - cw)

    data_cropped = data[top : top + ch, left : left + cw, :]
    mask_cropped = mask[top : top + ch, left : left + cw]
    return data_cropped, mask_cropped


def random_horizontal_flip(
    data: np.ndarray, mask: np.ndarray, p: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a random horizontal flip on hyperspectral data and mask.

    Args:
        data (np.ndarray): Hyperspectral data with shape (H, W, C).
        mask (np.ndarray): Corresponding mask with shape (H, W).
        p (float): Probability of applying the flip. Default is 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Flipped data and mask.
    """
    if random.random() < p:
        data = np.flip(data, axis=1).copy()  # Flip along the width axis
        mask = np.flip(mask, axis=1).copy()
    return data, mask


def random_vertical_flip(
    data: np.ndarray, mask: np.ndarray, p: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a random vertical flip on hyperspectral data and mask.

    Args:
        data (np.ndarray): Hyperspectral data with shape (H, W, C).
        mask (np.ndarray): Corresponding mask with shape (H, W).
        p (float): Probability of applying the flip. Default is 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Flipped data and mask.
    """
    if random.random() < p:
        data = np.flip(data, axis=0).copy()  # Flip along the height axis
        mask = np.flip(mask, axis=0).copy()
    return data, mask


def random_rotation(
    data: np.ndarray, mask: np.ndarray, angles: List[int] = [0, 90, 180, 270], p: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a random rotation on hyperspectral data and mask.

    Args:
        data (np.ndarray): Hyperspectral data with shape (H, W, C).
        mask (np.ndarray): Corresponding mask with shape (H, W).
        angles (List[int]): List of angles (in degrees) to randomly select from.
        p (float): Probability of applying the rotation. Default is 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotated data and mask.
    """
    if random.random() < p:
        angle = random.choice(angles)
        if angle == 0:
            return data, mask  # No rotation needed
        if angle == 90:
            data = np.rot90(data, k=1, axes=(0, 1)).copy()  # Rotate 90 degrees counterclockwise
            mask = np.rot90(mask, k=1, axes=(0, 1)).copy()
        elif angle == 180:
            data = np.rot90(data, k=2, axes=(0, 1)).copy()  # Rotate 180 degrees
            mask = np.rot90(mask, k=2, axes=(0, 1)).copy()
        elif angle == 270:
            data = np.rot90(data, k=3, axes=(0, 1)).copy()  # Rotate 270 degrees counterclockwise
            mask = np.rot90(mask, k=3, axes=(0, 1)).copy()
    return data, mask


def spectral_jitter(
    data: np.ndarray, mask: np.ndarray, jitter_strength: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add random noise to spectral values to simulate sensor noise.

    Args:
        data (np.ndarray): Hyperspectral data with shape (H, W, C).
        mask (np.ndarray): Corresponding mask with shape (H, W).
        jitter_strength (float): Standard deviation of the Gaussian noise. Default is 0.05.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Data with spectral jitter and unchanged mask.
    """
    noise = np.random.normal(0, jitter_strength, data.shape)
    data_jittered = data + noise
    return data_jittered, mask


def random_brightness_contrast(
    data: np.ndarray,
    mask: np.ndarray,
    brightness_factor: float = 0.2,
    contrast_factor: float = 0.2,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random brightness and contrast adjustments to hyperspectral data.

    Args:
        data (np.ndarray): Hyperspectral data with shape (H, W, C).
        mask (np.ndarray): Corresponding mask with shape (H, W).
        brightness_factor (float): Maximum brightness adjustment. Default is 0.2.
        contrast_factor (float): Maximum contrast adjustment. Default is 0.2.
        p (float): Probability of applying the adjustments. Default is 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted data and unchanged mask.
    """
    if random.random() < p:
        # Random brightness adjustment
        brightness = 1.0 + random.uniform(-brightness_factor, brightness_factor)
        # Random contrast adjustment
        contrast = 1.0 + random.uniform(-contrast_factor, contrast_factor)
        # Apply adjustments
        adjusted_data = data.copy()
        adjusted_data = adjusted_data * brightness
        mean = np.mean(adjusted_data, axis=(0, 1), keepdims=True)
        adjusted_data = (adjusted_data - mean) * contrast + mean
        return adjusted_data, mask

    return data, mask


def compute_percentiles(arr: np.ndarray, lower: int = 1, higher: int = 99) -> Tuple[float, float]:
    """
    Compute lower and higher percentiles for normalizing.

    Args:
        arr (np.ndarray): Input array to compute percentiles on.
        lower (int): Lower percentile value. Default is 1.
        higher (int): Higher percentile value. Default is 99.

    Returns:
        Tuple[float, float]: Lower and higher percentile values.
    """
    return np.percentile(arr, [lower, higher], axis=0)
