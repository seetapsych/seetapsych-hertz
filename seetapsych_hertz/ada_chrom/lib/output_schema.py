"""Output column schema helpers for delivery result files."""

from __future__ import annotations

from typing import Iterable

from .fft_fusion import active_fft_fusion_modes, fft_fusion_output_prefix
from .roi_definitions import ALL_ROI_NAMES


def per_frame_fieldnames(
    *,
    enable_fft_fusion: bool = True,
    fft_fusion_mode: str = "both",
    roi_names: Iterable[str] | None = None,
) -> list[str]:
    roi_names = list(roi_names) if roi_names is not None else list(ALL_ROI_NAMES)
    fields = [
        "frame_index",
        "frame_time_seconds",
        "video_fps",
        "window_start_frame_index",
        "window_end_frame_index",
    ]
    for name in roi_names:
        fields.extend(
            [
                f"roi_{name}_hr_bpm",
                f"roi_{name}_valid_samples",
                f"roi_{name}_area_ratio",
                f"roi_{name}_hr_source_frame_index",
                f"roi_{name}_status",
            ]
        )
    if enable_fft_fusion:
        for mode in active_fft_fusion_modes(fft_fusion_mode):
            prefix = fft_fusion_output_prefix(mode)
            fields.extend(
                [
                    f"{prefix}_hr_bpm",
                    f"{prefix}_roi_count",
                    f"{prefix}_peak_power",
                    f"{prefix}_hr_source_frame_index",
                    f"{prefix}_status",
                ]
            )
    return fields
