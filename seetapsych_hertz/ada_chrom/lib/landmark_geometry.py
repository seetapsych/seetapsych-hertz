"""Landmark validity and face-area geometry helpers."""

from __future__ import annotations

import numpy as np


def landmark_points_valid(points: np.ndarray, *, axis: tuple[int, ...] | None) -> np.ndarray | bool:
    points = np.asarray(points, dtype=float)
    valid = np.all(np.isfinite(points), axis=axis)
    valid &= ~np.all(np.isclose(points, 0.0), axis=axis)
    if np.ndim(valid) == 0:
        return bool(valid)
    return valid


def face_bbox_area_from_landmarks(frame_points: np.ndarray) -> float:
    points = np.asarray(frame_points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        return float("nan")
    if points.shape[0] >= 33:
        points = points[:33]
    points = points[:, :2]
    valid = np.all(np.isfinite(points), axis=1)
    valid &= ~np.all(np.isclose(points, 0.0), axis=1)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    valid_points = points[valid]
    width = float(np.max(valid_points[:, 0]) - np.min(valid_points[:, 0]))
    height = float(np.max(valid_points[:, 1]) - np.min(valid_points[:, 1]))
    area = width * height
    return area if area > 0.0 else float("nan")


def face_area_from_bbox(face_bbox: np.ndarray) -> float:
    bbox = np.asarray(face_bbox, dtype=float)
    if bbox.shape[0] < 4 or not np.all(np.isfinite(bbox[:4])):
        return float("nan")
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    area = width * height
    return area if area > 0.0 else float("nan")
