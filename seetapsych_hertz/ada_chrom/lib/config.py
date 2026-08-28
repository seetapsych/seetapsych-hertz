"""Shared delivery estimator configuration and CLI declarations.

Only algorithm/runtime switches live here.  Script-level input and output paths
stay in the two public entrypoints so that batch callers can pass paths
explicitly without changing estimator defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EstimatorConfig:
    """Algorithm options shared by both delivery entrypoints.

    The defaults intentionally match the validated delivery result in
    ``sample/heart_rate_estimation_result.csv``.
    """

    # ROI selectors can be explicit names such as "1" or groups such as "all".
    roi_regions: tuple[str, ...] = ("all",)

    # Number of valid samples held per ROI before a BPM estimate is produced.
    window_samples: int = 300

    # Update BPM every N processed frames; skipped frames reuse the last result.
    hr_update_stride: int = 1

    # Optional CHROM band-pass path.  Disabled by default to preserve reference output.
    use_butterworth: bool = False

    # Skin ROI post-processing switches used by adaptive skin methods.
    skin_roi_exclude_mouth: bool = False
    skin_roi_fill_holes: bool = True

    # FFT fusion combines valid ROI windows into additional fused BPM columns.
    enable_fft_fusion: bool = True
    fft_fusion_mode: str = "both"

    def __post_init__(self) -> None:
        object.__setattr__(self, "roi_regions", tuple(self.roi_regions))


DEFAULT_ESTIMATOR_CONFIG = EstimatorConfig()


BOOL_VALUE_METAVAR = "{True,False}"
ROI_SELECTOR_HELP = (
    "ROI selector. Names: 1 2 3 4 5 sdk skin_legacy "
    "skin_a_fixed_forehead skin_b_adaptive_forehead skin_c_connected_components. "
    "Groups: polygon, skin_all, all. Default: all."
)

VIDEO_ARG_SPEC = {
    "flags": ("--video",),
    "type": "path",
    "default": None,
}

LANDMARKS_ARG_SPEC = {
    "flags": ("--landmarks",),
    "type": "path",
    "default": None,
}

# CLI specs are data, not parser code.  cli_utils.py turns these declarations
# into argparse options and converts parsed values back into EstimatorConfig.
COMMON_ESTIMATOR_ARG_SPECS = (
    {
        "flags": ("--hr-window-samples",),
        "type": "int",
        "default": 300,
    },
    {
        "flags": ("--hr-update-stride",),
        "type": "int",
        "default": 1,
    },
    {
        "flags": ("--roi-regions",),
        "nargs": "+",
        "choices_from": "roi_region_choices",
        "default": ["all"],
        "help_from": "roi_help",
    },
    {
        "flags": ("--fft-fusion-mode",),
        "choices_from": "fft_fusion_modes",
        "default": "both",
        "help": "FFT fusion mode. Default: both.",
    },
    {
        "flags": ("--enable-fft-fusion",),
        "type": "bool",
        "default": True,
        "metavar": BOOL_VALUE_METAVAR,
    },
    {
        "flags": ("--use-butterworth",),
        "type": "bool",
        "default": False,
        "metavar": BOOL_VALUE_METAVAR,
    },
    {
        "flags": ("--skin-roi-exclude-mouth",),
        "type": "bool",
        "default": False,
        "metavar": BOOL_VALUE_METAVAR,
    },
    {
        "flags": ("--skin-roi-fill-holes",),
        "type": "bool",
        "default": True,
        "metavar": BOOL_VALUE_METAVAR,
    },
)

FRAME_DETAILS_ARG_SPECS = (
    VIDEO_ARG_SPEC,
    LANDMARKS_ARG_SPEC,
    {
        "flags": ("--output-csv",),
        "type": "path",
        "default": None,
    },
    {
        "flags": ("--output-npz",),
        "type": "path",
        "default": None,
    },
    *COMMON_ESTIMATOR_ARG_SPECS,
)

ROI_HEART_RATE_ARG_SPECS = (
    {
        **VIDEO_ARG_SPEC,
        "help": "Input video path.",
    },
    {
        **LANDMARKS_ARG_SPEC,
        "help": "SPIPNet-280 landmark CSV path.",
    },
    {
        "flags": ("--output-csv",),
        "type": "path",
        "default": None,
        "help": "Per-frame heart-rate CSV output path.",
    },
    *COMMON_ESTIMATOR_ARG_SPECS,
)

ESTIMATOR_KWARG_MAP = {
    "video": "video_path",
    "landmarks": "landmark_csv",
    "output_csv": "output_csv",
    "output_npz": "output_npz",
    "roi_regions": "roi_regions",
    "hr_window_samples": "window_samples",
    "hr_update_stride": "hr_update_stride",
    "use_butterworth": "use_butterworth",
    "skin_roi_exclude_mouth": "skin_roi_exclude_mouth",
    "skin_roi_fill_holes": "skin_roi_fill_holes",
    "enable_fft_fusion": "enable_fft_fusion",
    "fft_fusion_mode": "fft_fusion_mode",
}

ENTRYPOINT_CLI_CONFIGS = {
    "frame_details": {
        "description": "Estimate frame-by-frame HR details from one video and landmark CSV.",
        "arg_specs": FRAME_DETAILS_ARG_SPECS,
        "estimator_kwargs": ESTIMATOR_KWARG_MAP,
    },
    "roi_heart_rate": {
        "description": "Estimate ROI heart rate from one video and one SPIPNet-280 landmark CSV.",
        "arg_specs": ROI_HEART_RATE_ARG_SPECS,
        "roi_help": ROI_SELECTOR_HELP,
        "estimator_kwargs": {
            key: value
            for key, value in ESTIMATOR_KWARG_MAP.items()
            if key != "output_npz"
        },
    },
}
