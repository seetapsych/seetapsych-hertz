# -*- coding: utf-8 -*-

import numpy


def extract_signal(image: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    """
    Extract the mean BGR signal from the masked image area.

    :param image: Image in HWC format with BGR layout.
    :param mask: Single-channel ROI mask with shape [H, W].
    :return: Mean signal of each channel in BGR order, with shape [3].
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape [H, W, 3], got {image.shape}")

    if mask.ndim != 2:
        raise ValueError(f"mask must have shape [H, W], got {mask.shape}")

    if image.shape[:2] != mask.shape:
        raise ValueError(f"image and mask size mismatch: image={image.shape[:2]}, mask={mask.shape}")

    # Keep the same threshold behavior.
    valid = mask > 10

    if not numpy.any(valid):
        return numpy.zeros(3, dtype=numpy.float64)

    return numpy.asarray(image[valid].mean(axis=0, dtype=numpy.float64))


def main():
    pass


if __name__ == "__main__":
    main()
