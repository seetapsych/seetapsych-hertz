"""Public package API for the 20260825 heart-rate delivery estimator."""

from __future__ import annotations

from .config import DEFAULT_ESTIMATOR_CONFIG, EstimatorConfig
from .framewise_estimator import DetailedVideoResult, FrameDetailResult, FramewiseHeartRateEstimator


def estimate_video_heart_rate(*args, **kwargs):
    """Run the compatible whole-video heart-rate estimator."""

    from .estimate_roi_heart_rate_from_spipnet280 import estimate_video_heart_rate as _estimate_video_heart_rate

    return _estimate_video_heart_rate(*args, **kwargs)


def estimate_frame_details(*args, **kwargs):
    """Run the detailed frame-by-frame estimator."""

    from .estimate_video_frame_details import estimate_video_frame_details as _estimate_video_frame_details

    return _estimate_video_frame_details(*args, **kwargs)


def __getattr__(name: str):
    if name == "EstimationSummary":
        from .estimate_roi_heart_rate_from_spipnet280 import EstimationSummary

        return EstimationSummary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_ESTIMATOR_CONFIG",
    "DetailedVideoResult",
    "EstimationSummary",
    "EstimatorConfig",
    "FrameDetailResult",
    "FramewiseHeartRateEstimator",
    "estimate_frame_details",
    "estimate_video_heart_rate",
]
