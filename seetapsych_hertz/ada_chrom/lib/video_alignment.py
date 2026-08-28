"""Video reading, resampling, and frame/landmark alignment helpers."""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Iterable

import numpy as np

from .landmark_reader import landmark_csv_frame_count_and_fps
from .landmark_utils import FrameLandmarks


def read_source_video_frames(video_path: Path) -> tuple[list[np.ndarray], float, int]:
    """Load the source video into memory before length-based resampling."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(frame_bgr)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")
    return frames, source_fps, source_frame_count


def resample_video_frames_to_length(
    frames: list[np.ndarray],
    *,
    target_length: int,
) -> list[np.ndarray]:
    """Linearly resample video frames so frame count matches landmark rows."""

    if target_length <= 0:
        return []
    frame_array = np.asarray(frames)
    original_length = int(frame_array.shape[0])
    if original_length == target_length:
        return [frame for frame in frame_array.astype(np.uint8, copy=False)]

    from scipy.interpolate import interp1d

    height, width, channels = frame_array.shape[1:]
    interpolator = interp1d(
        np.arange(original_length),
        frame_array.reshape(original_length, -1),
        kind="linear",
        axis=0,
    )
    new_indices = np.linspace(0, original_length - 1, target_length, endpoint=False)
    resampled_data = interpolator(new_indices)
    resampled = resampled_data.reshape(target_length, height, width, channels)
    return [frame for frame in np.clip(resampled, 0, 255).astype(np.uint8)]


def iter_video_frames_for_landmarks(
    video_path: Path,
    landmark_csv: Path,
) -> Iterable[np.ndarray]:
    """Yield source video frames after resampling to the landmark CSV length."""

    video_path = Path(video_path)
    target_length, _target_fps = landmark_csv_frame_count_and_fps(landmark_csv)
    frames, _source_fps, _source_frame_count = read_source_video_frames(video_path)
    yield from resample_video_frames_to_length(frames, target_length=target_length)


def iter_aligned_frame_landmarks(
    frames: Iterable[np.ndarray],
    frame_landmarks: Iterable[FrameLandmarks],
) -> Iterable[tuple[np.ndarray, FrameLandmarks]]:
    """Pair one prepared video frame with one landmark row.

    A missing row is treated as an alignment error.  To represent face-tracking
    failure for a frame, keep the CSV row and mark its landmarks/bbox invalid.
    """

    missing = object()
    for offset, (frame_bgr, landmarks) in enumerate(zip_longest(frames, frame_landmarks, fillvalue=missing)):
        if frame_bgr is missing or landmarks is missing:
            raise ValueError(
                "frames and frame_landmarks must have the same length; "
                f"mismatch detected at pair offset {offset}."
            )
        yield frame_bgr, landmarks
