"""Single-video ROI heart-rate estimation delivery entrypoint.

This file is intentionally small and readable.  It keeps the public name
`estimate_roi_heart_rate_from_spipnet280.py`, while all numerical heart-rate
logic is delegated to focused delivery modules.

Why delegate instead of rewriting the equations here?
    The delivery requirement is to keep heart-rate performance unchanged.  ROI
    masking, CHROM projection, timestamp interpolation, Hamming-window FFT, and
    FFT fusion therefore follow the source implementation exactly through the
    shared delivery modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "heart_rate_delivery_20260825"

from .cli_utils import build_delivery_arg_parser, parsed_args_to_estimator_kwargs
from .config import DEFAULT_ESTIMATOR_CONFIG, EstimatorConfig
from .framewise_estimator import FramewiseHeartRateEstimator
from .landmark_reader import iter_frame_landmarks_csv
from .landmark_utils import FrameLandmarks
from .path_utils import required_path_argument, resolve_required_path
from .result_writer import write_compatible_rows_csv
from .roi_definitions import FFT_FUSION_MODES, ROI_REGION_CHOICES
from .video_alignment import iter_aligned_frame_landmarks, iter_video_frames_for_landmarks


# Script-level run paths.  Edit these strings for direct runs, or pass explicit
# arguments through the CLI/API when calling this entrypoint from another script.
VIDEO_PATH = r".\sample\video_resampled.mp4"
LANDMARK_CSV = r".\sample\video_landmarks280.csv"
OUTPUT_CSV = r".\sample_roi_heart_rate_output.csv"


@dataclass(frozen=True)
class EstimationSummary:
    """Compact runtime summary returned by the single-video estimator."""

    video_path: Path | None
    landmark_csv: Path | None
    output_csv: Path
    roi_names: tuple[str, ...]
    decoded_frames: int
    output_rows: int
    video_fps: float
    processing_fps: float
    elapsed_seconds: float


def estimate_prepared_heart_rate(
    *,
    frames: Iterable[object],
    frame_landmarks: Iterable[FrameLandmarks],
    output_csv: Path | str | None = None,
    estimator_config: EstimatorConfig | None = None,
) -> EstimationSummary:
    """Estimate compatible CSV rows from already aligned frame/landmark pairs.

    This path intentionally calls process_frame_row(), so it computes the same
    ROI/BVP/FFT state needed for HR but does not keep per-frame masks or BVP
    arrays in memory.
    """

    start = time.perf_counter()
    estimator_config = estimator_config or DEFAULT_ESTIMATOR_CONFIG
    output_csv = required_path_argument(output_csv, "output_csv")
    estimator = FramewiseHeartRateEstimator(
        config=estimator_config,
    )

    rows: list[dict[str, object]] = []
    for frame_bgr, current_landmarks in iter_aligned_frame_landmarks(frames, frame_landmarks):
        rows.append(estimator.process_frame_row(frame_bgr, current_landmarks))

    write_compatible_rows_csv(
        output_csv,
        rows,
        roi_names=estimator.roi_names,
        enable_fft_fusion=estimator_config.enable_fft_fusion,
        fft_fusion_mode=estimator_config.fft_fusion_mode,
    )
    video_fps = 0.0
    if rows:
        video_fps = float(rows[0].get("video_fps", 0.0) or 0.0)
    elapsed = time.perf_counter() - start

    return EstimationSummary(
        video_path=None,
        landmark_csv=None,
        output_csv=output_csv,
        roi_names=estimator.roi_names,
        decoded_frames=len(rows),
        output_rows=len(rows),
        video_fps=video_fps,
        processing_fps=(float(len(rows)) / elapsed if elapsed > 0 else 0.0),
        elapsed_seconds=elapsed,
    )


def estimate_video_heart_rate(
    *,
    video_path: Path | str | None = None,
    landmark_csv: Path | str | None = None,
    output_csv: Path | str | None = None,
    estimator_config: EstimatorConfig | None = None,
) -> EstimationSummary:
    """Estimate per-frame/window heart rate for all selected ROIs.

    Parameters intentionally mirror the original estimator defaults.  The
    default ROI selector is `all`, which includes polygon ROIs 1-5, the SDK
    face-area ROI, and all four skin ROI strategies.
    """

    start = time.perf_counter()
    video_path = resolve_required_path(video_path, VIDEO_PATH, "video_path")
    landmark_csv = resolve_required_path(landmark_csv, LANDMARK_CSV, "landmark_csv")
    output_csv = resolve_required_path(output_csv, OUTPUT_CSV, "output_csv")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not landmark_csv.exists():
        raise FileNotFoundError(f"Landmark CSV not found: {landmark_csv}")

    # Video frames are resampled to the landmark CSV length before the prepared
    # loop consumes one frame and one landmark row at a time.
    frames = iter_video_frames_for_landmarks(video_path, landmark_csv)
    frame_landmarks = iter_frame_landmarks_csv(landmark_csv)
    summary = estimate_prepared_heart_rate(
        frames=frames,
        frame_landmarks=frame_landmarks,
        output_csv=output_csv,
        estimator_config=estimator_config,
    )

    elapsed = time.perf_counter() - start
    return EstimationSummary(
        video_path=video_path,
        landmark_csv=landmark_csv,
        output_csv=output_csv,
        roi_names=summary.roi_names,
        decoded_frames=summary.decoded_frames,
        output_rows=summary.output_rows,
        video_fps=summary.video_fps,
        processing_fps=(float(summary.decoded_frames) / elapsed if elapsed > 0 else 0.0),
        elapsed_seconds=elapsed,
    )


def build_arg_parser():
    return build_delivery_arg_parser(
        "roi_heart_rate",
        roi_region_choices=ROI_REGION_CHOICES,
        fft_fusion_modes=FFT_FUSION_MODES,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = estimate_video_heart_rate(**parsed_args_to_estimator_kwargs(args, "roi_heart_rate"))
    print(f"output_csv={summary.output_csv}")
    print(f"decoded_frames={summary.decoded_frames}")
    print(f"output_rows={summary.output_rows}")
    print(f"processing_fps={summary.processing_fps:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
