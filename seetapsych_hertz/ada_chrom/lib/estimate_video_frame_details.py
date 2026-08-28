"""Frame-by-frame detailed heart-rate estimation entrypoint."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "heart_rate_delivery_20260825"

from .cli_utils import build_delivery_arg_parser, parsed_args_to_estimator_kwargs
from .config import DEFAULT_ESTIMATOR_CONFIG, EstimatorConfig
from .framewise_estimator import (
    DetailedVideoResult,
    FrameDetailResult,
    FramewiseHeartRateEstimator,
)
from .landmark_reader import iter_frame_landmarks_csv
from .landmark_utils import FrameLandmarks
from .path_utils import optional_path_argument, resolve_optional_path, resolve_required_path
from .result_writer import write_compatible_frame_results_csv, write_detail_npz
from .roi_definitions import FFT_FUSION_MODES, ROI_REGION_CHOICES
from .video_alignment import iter_aligned_frame_landmarks, iter_video_frames_for_landmarks


# Script-level run paths.  Edit these strings for quick local runs, or pass
# explicit function/CLI arguments from another script.
VIDEO_PATH = r".\sample\video_resampled.mp4"
LANDMARK_CSV = r".\sample\video_landmarks280.csv"
OUTPUT_CSV = r".\sample_frame_details_output.csv"
OUTPUT_NPZ = r".\sample_frame_details_output.npz"


def estimate_prepared_frame_details(
    *,
    frames: Iterable[object],
    frame_landmarks: Iterable[FrameLandmarks],
    output_csv: Path | str | None = None,
    output_npz: Path | str | None = None,
    estimator_config: EstimatorConfig | None = None,
) -> DetailedVideoResult:
    """Process already aligned frames one by one and collect detailed outputs.

    The caller owns video reading/resampling and landmark reading.  Each loop
    iteration feeds exactly one frame and its corresponding FrameLandmarks into
    FramewiseHeartRateEstimator.process_frame().
    """

    start = time.perf_counter()
    estimator_config = estimator_config or DEFAULT_ESTIMATOR_CONFIG
    output_csv = optional_path_argument(output_csv)
    output_npz = optional_path_argument(output_npz)
    estimator = FramewiseHeartRateEstimator(
        config=estimator_config,
    )

    frame_results: list[FrameDetailResult] = []
    for frame_bgr, current_landmarks in iter_aligned_frame_landmarks(frames, frame_landmarks):
        frame_results.append(estimator.process_frame(frame_bgr, current_landmarks))

    result = DetailedVideoResult(
        video_path=None,
        landmark_csv=None,
        output_csv=output_csv,
        output_npz=output_npz,
        roi_names=estimator.roi_names,
        decoded_frames=len(frame_results),
        frame_results=frame_results,
        elapsed_seconds=time.perf_counter() - start,
    )
    if output_csv is not None:
        write_compatible_frame_results_csv(
            output_csv,
            result.frame_results,
            roi_names=result.roi_names,
            enable_fft_fusion=estimator_config.enable_fft_fusion,
            fft_fusion_mode=estimator_config.fft_fusion_mode,
        )
    if output_npz is not None:
        write_detail_npz(output_npz, result.frame_results, result.roi_names)
    return result


def estimate_video_frame_details(
    *,
    video_path: Path | str | None = None,
    landmark_csv: Path | str | None = None,
    output_csv: Path | str | None = None,
    output_npz: Path | str | None = None,
    estimator_config: EstimatorConfig | None = None,
) -> DetailedVideoResult:
    """Read/resample one video to landmark length, then run framewise details."""

    start = time.perf_counter()
    video_path = resolve_required_path(video_path, VIDEO_PATH, "video_path")
    landmark_csv = resolve_required_path(landmark_csv, LANDMARK_CSV, "landmark_csv")
    output_csv = resolve_optional_path(output_csv, OUTPUT_CSV)
    output_npz = resolve_optional_path(output_npz, OUTPUT_NPZ)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not landmark_csv.exists():
        raise FileNotFoundError(f"Landmark CSV not found: {landmark_csv}")

    # The outer entrypoint prepares both streams; the prepared function below
    # shows the actual per-frame frame+landmarks processing loop.
    frames = iter_video_frames_for_landmarks(video_path, landmark_csv)
    frame_landmarks = iter_frame_landmarks_csv(landmark_csv)
    result = estimate_prepared_frame_details(
        frames=frames,
        frame_landmarks=frame_landmarks,
        output_csv=output_csv,
        output_npz=output_npz,
        estimator_config=estimator_config,
    )

    return DetailedVideoResult(
        video_path=video_path,
        landmark_csv=landmark_csv,
        output_csv=result.output_csv,
        output_npz=result.output_npz,
        roi_names=result.roi_names,
        decoded_frames=result.decoded_frames,
        frame_results=result.frame_results,
        elapsed_seconds=time.perf_counter() - start,
    )


def build_arg_parser():
    return build_delivery_arg_parser(
        "frame_details",
        roi_region_choices=ROI_REGION_CHOICES,
        fft_fusion_modes=FFT_FUSION_MODES,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = estimate_video_frame_details(**parsed_args_to_estimator_kwargs(args, "frame_details"))
    print(f"decoded_frames={result.decoded_frames}")
    print(f"frame_results={len(result.frame_results)}")
    if result.output_csv is not None:
        print(f"output_csv={result.output_csv}")
    if result.output_npz is not None:
        print(f"output_npz={result.output_npz}")
    print(f"elapsed_seconds={result.elapsed_seconds:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
