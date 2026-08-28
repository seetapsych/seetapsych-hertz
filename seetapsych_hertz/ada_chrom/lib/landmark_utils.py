"""Landmark data structures, mapping, and validity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .landmark_geometry import face_area_from_bbox, landmark_points_valid
from .roi_definitions import N_LANDMARKS, ROI_POLYGON_INDICES, SDK_FACEAREA_ROI_NAME, SKIN_ROI_NAMES


_SDK_LANDMARK_MAPPER = None


@dataclass(frozen=True)
class FrameLandmarks:
    path: Path
    frame_offset: int
    frame_index: int
    frame_time_seconds: float
    video_fps: float
    points: np.ndarray
    face_bbox: np.ndarray | None = None
    sdk_points81: np.ndarray | None = None


@dataclass
class LandmarkSeries:
    path: Path
    frame_indices: np.ndarray
    frame_time_seconds: np.ndarray
    video_fps: float
    points: np.ndarray
    face_bboxes: np.ndarray | None = None
    sdk_points81: np.ndarray | None = None


def load_landmark_mapper_280_to_81():
    global _SDK_LANDMARK_MAPPER
    if _SDK_LANDMARK_MAPPER is not None:
        return _SDK_LANDMARK_MAPPER

    from .landmark_mapper_280281 import LandmarkMapper

    _SDK_LANDMARK_MAPPER = LandmarkMapper()
    return _SDK_LANDMARK_MAPPER


def spipnet280_to_sdk81_points(frame_points: np.ndarray, *, mapper=None) -> np.ndarray:
    points = np.asarray(frame_points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=float)

    points = points[:, :2]
    if points.shape[0] >= N_LANDMARKS:
        mapper = mapper if mapper is not None else load_landmark_mapper_280_to_81()
        return np.asarray(mapper.map_280_to_81(points[:N_LANDMARKS]), dtype=float)

    return points[:81].astype(float, copy=True)


def spipnet280_series_to_sdk81_points(points: np.ndarray, *, mapper=None) -> np.ndarray:
    series = np.asarray(points, dtype=float)
    if series.ndim != 3 or series.shape[2] < 2:
        return np.empty((0, 81, 2), dtype=float)
    mapper = mapper if mapper is not None else load_landmark_mapper_280_to_81()
    mapped = [spipnet280_to_sdk81_points(frame_points, mapper=mapper) for frame_points in series]
    return np.asarray(mapped, dtype=float)


def sdk81_points_for_landmarks(
    landmarks: LandmarkSeries,
    frame_limit: int | None = None,
    *,
    mapper=None,
) -> np.ndarray:
    limit = landmarks.points.shape[0] if frame_limit is None else min(int(frame_limit), landmarks.points.shape[0])
    cached = landmarks.sdk_points81
    if cached is not None:
        cached = np.asarray(cached, dtype=float)
        if cached.ndim == 3 and cached.shape[0] >= limit and cached.shape[1] >= 81 and cached.shape[2] >= 2:
            return cached[:limit, :81, :2]

    mapped = spipnet280_series_to_sdk81_points(landmarks.points[:limit], mapper=mapper)
    landmarks.sdk_points81 = mapped
    return mapped


def frame_has_valid_face_bbox(frame_landmarks: FrameLandmarks) -> bool:
    if frame_landmarks.face_bbox is None:
        return True
    bbox = np.asarray(frame_landmarks.face_bbox, dtype=float)
    return bool(np.all(np.isfinite(bbox)) and np.isfinite(face_area_from_bbox(bbox)))


def frame_raw_landmarks_valid(points: np.ndarray) -> bool:
    return bool(landmark_points_valid(points, axis=None))


def frame_landmark_valid_masks(
    frame_landmarks: FrameLandmarks,
    *,
    roi_names: Iterable[str],
    sdk_points81: np.ndarray | None,
) -> dict[str, bool]:
    roi_names = tuple(roi_names)
    masks: dict[str, bool] = {}
    points = np.asarray(frame_landmarks.points, dtype=float)
    for name, indices in ROI_POLYGON_INDICES:
        if name not in roi_names:
            continue
        polygon = points[indices]
        masks[name] = bool(landmark_points_valid(polygon, axis=None))
    raw_valid = frame_raw_landmarks_valid(points)
    sdk_valid = False
    if sdk_points81 is not None:
        sdk_valid = bool(landmark_points_valid(sdk_points81, axis=None) and raw_valid)
    masks[SDK_FACEAREA_ROI_NAME] = sdk_valid
    skin_valid = raw_valid and sdk_valid
    for name in SKIN_ROI_NAMES:
        if name in roi_names:
            masks[name] = skin_valid
    return masks


def roi_landmark_valid_masks(
    landmarks: LandmarkSeries,
    frame_limit: int | None = None,
    *,
    skin_roi_methods: Iterable[str] | None = (),
    polygon_roi_names: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    from .roi_definitions import POLYGON_ROI_NAMES, SKIN_ROI_NAME_BY_METHOD, normalize_skin_roi_methods

    limit = landmarks.points.shape[0] if frame_limit is None else min(frame_limit, landmarks.points.shape[0])
    masks: dict[str, np.ndarray] = {}
    wanted_polygon_names = set(POLYGON_ROI_NAMES if polygon_roi_names is None else polygon_roi_names)
    for name, indices in ROI_POLYGON_INDICES:
        if name not in wanted_polygon_names:
            continue
        points = landmarks.points[:limit, indices]
        masks[name] = np.asarray(landmark_points_valid(points, axis=(1, 2)), dtype=bool)
    sdk_raw_points = landmarks.points[:limit]
    has_cached_sdk_points = landmarks.sdk_points81 is not None
    sdk_points = sdk81_points_for_landmarks(landmarks, limit)
    sdk_valid = np.asarray(landmark_points_valid(sdk_points, axis=(1, 2)), dtype=bool)
    if not has_cached_sdk_points:
        sdk_valid &= np.asarray(landmark_points_valid(sdk_raw_points, axis=(1, 2)), dtype=bool)
    masks[SDK_FACEAREA_ROI_NAME] = sdk_valid
    raw_valid = np.asarray(landmark_points_valid(sdk_raw_points, axis=(1, 2)), dtype=bool)
    skin_valid = raw_valid & sdk_valid
    for method in normalize_skin_roi_methods(skin_roi_methods):
        masks[SKIN_ROI_NAME_BY_METHOD[method]] = skin_valid.copy()
    return masks
