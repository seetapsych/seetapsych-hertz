"""Shared ROI mask and BGR extraction primitives."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


_CV2_MODULE: Any | None = None
_CV2_LOAD_ATTEMPTED = False


def _load_cv2() -> Any | None:
    global _CV2_MODULE, _CV2_LOAD_ATTEMPTED
    if _CV2_LOAD_ATTEMPTED:
        return _CV2_MODULE
    _CV2_LOAD_ATTEMPTED = True
    try:
        import cv2
    except Exception:
        _CV2_MODULE = None
    else:
        _CV2_MODULE = cv2
    return _CV2_MODULE


def initialize_facearea_roi(
    points: Any,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Port the C++ inilizeROI() point selection for 81 landmarks."""
    coords = _points_to_array(points)
    if coords.shape[0] < 81:
        return None

    left = [9999999, 9999999]
    right = [-1, -1]
    length = int(
        math.sqrt(
            (coords[9, 1] - coords[0, 1]) * (coords[9, 1] - coords[0, 1])
            + (coords[9, 0] - coords[0, 0]) * (coords[9, 0] - coords[0, 0])
        )
    )
    dlta1 = int(length / 4.5)
    dlta2 = int(length / 50)

    cheek_points: list[tuple[int, int]] = []
    mouth_points: list[tuple[int, int]] = []

    def add_cheek(x: float, y: float) -> None:
        point = (int(x), int(y))
        cheek_points.append(point)
        _update_bbox(point, left, right)

    add_cheek(coords[0, 0], coords[0, 1] + dlta1)
    for i in range(65, 73):
        add_cheek(coords[i, 0] + dlta2, coords[i, 1])
    for i in range(80, 72, -1):
        add_cheek(coords[i, 0] - dlta2, coords[i, 1])
    add_cheek(coords[9, 0], coords[9, 1] + dlta1)

    dlta = int(length / 50)
    left[0] -= dlta
    left[1] -= dlta
    right[0] += dlta
    right[1] += dlta

    mouth_indices = [46, 50, 48, 51, 47, 59, 55, 58]
    for index in mouth_indices:
        mouth_points.append((int(coords[index, 0]), int(coords[index, 1])))

    if image_shape is not None:
        height = int(image_shape[0])
        width = int(image_shape[1])
        if left[0] <= 0 or left[1] <= 0:
            return None
        if right[0] >= width or right[1] >= height:
            return None

    return (left[0], left[1]), (right[0], right[1]), cheek_points, mouth_points


def skin_detection(image_bgr: Any) -> np.ndarray:
    """Port of the C++ cvSkinSegment YCrCb ellipse rule."""
    image = np.asarray(image_bgr, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("image_bgr must have shape (H, W, 3).")

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    y = np.rint(np.float32(0.299) * r + np.float32(0.587) * g + np.float32(0.114) * b).astype(np.int32)
    cr = np.rint((r - y) * np.float32(0.713) + np.float32(128.0)).astype(np.int32)
    cb = np.rint((b - y) * np.float32(0.564) + np.float32(128.0)).astype(np.int32)

    cb = cb - 109
    cr = cr - 152
    x1 = _trunc_div(819 * cr - 614 * cb, 32) + 51
    y1 = _trunc_div(819 * cr + 614 * cb, 32) + 77
    x1 = _trunc_div(x1 * 41, 1024)
    y1 = _trunc_div(y1 * 73, 1024)
    value = x1 * x1 + y1 * y1

    return np.where(y < 100, value < 700, value < 850)


def combine_mask(face_mask: Any, skin_mask: Any) -> np.ndarray:
    active = _mask_active(face_mask) & _mask_active(skin_mask)
    return np.where(active, 255, 0).astype(np.uint8)


def get_bgr(image_bgr: Any, mask: Any) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=float)
    mask = np.asarray(mask)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("image_bgr must have shape (H, W, 3).")
    selected = mask > 10
    if not np.any(selected):
        return np.zeros(3, dtype=float)
    cv2 = _load_cv2()
    if cv2 is not None:
        cv_mask = np.where(selected, 255, 0).astype(np.uint8)
        return np.asarray(cv2.mean(image[:, :, :3], mask=cv_mask)[:3], dtype=float)
    return image[:, :, :3][selected].mean(axis=0).astype(float)


def _trunc_div(numerator: Any, denominator: int) -> np.ndarray:
    values = np.asarray(numerator, dtype=np.int32)
    return (np.sign(values) * (np.abs(values) // int(denominator))).astype(np.int32)


def _mask_active(mask: Any) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.dtype == bool:
        return mask_array
    return mask_array > 10


def _points_to_array(points: Any) -> np.ndarray:
    coords = []
    for point in points:
        if hasattr(point, "x") and hasattr(point, "y"):
            coords.append((float(point.x), float(point.y)))
        else:
            x, y = point[:2]
            coords.append((float(x), float(y)))
    return np.asarray(coords, dtype=float)


def _update_bbox(point: tuple[int, int], left: list[int], right: list[int]) -> None:
    x, y = point
    left[0] = min(left[0], x)
    left[1] = min(left[1], y)
    right[0] = max(right[0], x)
    right[1] = max(right[1], y)
