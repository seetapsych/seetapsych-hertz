"""Landmark CSV parsing helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .landmark_utils import (
    FrameLandmarks,
    LandmarkSeries,
    frame_has_valid_face_bbox,
    frame_landmark_valid_masks,
    frame_raw_landmarks_valid,
    load_landmark_mapper_280_to_81,
    roi_landmark_valid_masks,
    sdk81_points_for_landmarks,
    spipnet280_series_to_sdk81_points,
    spipnet280_to_sdk81_points,
)
from .roi_definitions import N_LANDMARKS


def read_landmark_csv(path: Path) -> LandmarkSeries:
    """Read the full SPIPNet-280 CSV into arrays for batch-style utilities."""

    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        header = handle.readline().strip().split(",")

    if len(header) < 4:
        raise ValueError(f"Invalid landmark csv header: {path}")

    try:
        numeric = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=range(1, len(header)),
            ndmin=2,
        )
    except ValueError:
        numeric = _read_landmark_csv_slow(path, header)

    header_to_numeric_col = {name: index - 1 for index, name in enumerate(header) if index > 0}
    frame_indices = numeric[:, header_to_numeric_col["frame_index"]].astype(np.int64)
    frame_time_seconds = numeric[:, header_to_numeric_col["frame_time_seconds"]].astype(float)
    fps_values = numeric[:, header_to_numeric_col["video_fps"]].astype(float)
    video_fps = float(np.nanmedian(fps_values)) if fps_values.size else 0.0

    points = np.empty((numeric.shape[0], N_LANDMARKS, 2), dtype=np.float32)
    for index in range(N_LANDMARKS):
        points[:, index, 0] = numeric[:, header_to_numeric_col[f"point_{index}_x"]]
        points[:, index, 1] = numeric[:, header_to_numeric_col[f"point_{index}_y"]]
    bbox_columns = ["face_x0", "face_y0", "face_x1", "face_y1"]
    face_bboxes = np.full((numeric.shape[0], 4), np.nan, dtype=np.float32)
    if all(column in header_to_numeric_col for column in bbox_columns):
        for bbox_index, column in enumerate(bbox_columns):
            face_bboxes[:, bbox_index] = numeric[:, header_to_numeric_col[column]]

    return LandmarkSeries(
        path=path,
        frame_indices=frame_indices,
        frame_time_seconds=frame_time_seconds,
        video_fps=video_fps,
        points=points,
        face_bboxes=face_bboxes,
    )


def _read_landmark_csv_slow(path: Path, header: list[str]) -> np.ndarray:
    """Fallback parser used when np.loadtxt cannot parse the CSV directly."""

    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([float(row[name]) for name in header[1:]])
    return np.asarray(rows, dtype=float)


def iter_frame_landmarks_csv(path: Path):
    """Stream one FrameLandmarks object per CSV row for framewise processing."""

    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Invalid landmark CSV header: {path}")
        for offset, row in enumerate(reader):
            points = np.empty((N_LANDMARKS, 2), dtype=np.float32)
            for index in range(N_LANDMARKS):
                points[index, 0] = float(row[f"point_{index}_x"])
                points[index, 1] = float(row[f"point_{index}_y"])
            face_bbox = None
            if all(name in row for name in ("face_x0", "face_y0", "face_x1", "face_y1")):
                face_bbox = np.asarray(
                    [
                        float(row["face_x0"]),
                        float(row["face_y0"]),
                        float(row["face_x1"]),
                        float(row["face_y1"]),
                    ],
                    dtype=np.float32,
                )
            yield FrameLandmarks(
                path=path,
                frame_offset=offset,
                frame_index=int(float(row["frame_index"])),
                frame_time_seconds=float(row["frame_time_seconds"]),
                video_fps=float(row["video_fps"]),
                points=points,
                face_bbox=face_bbox,
            )


def landmark_csv_frame_count_and_fps(path: Path) -> tuple[int, float]:
    """Return landmark row count and the first declared video_fps value."""

    path = Path(path)
    count = 0
    video_fps = 0.0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if count == 0:
                video_fps = float(row.get("video_fps") or 0.0)
            count += 1
    return count, video_fps
