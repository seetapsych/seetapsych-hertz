"""Framewise heart-rate estimation orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import EstimatorConfig
from .fft_fusion import (
    FftFusionResult,
    active_fft_fusion_modes,
    estimate_fft_fused_heart_rate,
    fft_fusion_output_prefix,
)
from .heart_rate_detector import BvpDetail, FftCurve, compute_bvp_detail, estimate_heart_rate_bgr
from .landmark_utils import (
    FrameLandmarks,
    frame_has_valid_face_bbox as _frame_has_valid_face_bbox,
)
from .roi_definitions import roi_names_for_regions, skin_roi_methods_for_roi_names
from .roi_extraction import extract_frame_roi_values, make_skin_roi_builders


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameDetailResult:
    """Detailed return value for one processed frame.

    ``hr_results`` is the compatibility CSV row.  The remaining fields expose
    intermediate per-frame details for callers that need masks, BVP curves,
    FFT curves, or fused values.
    """

    frame_offset: int
    frame_index: int
    frame_time_seconds: float
    roi_masks: dict[str, np.ndarray]
    roi_bgr: dict[str, np.ndarray]
    roi_pixel_counts: dict[str, int]
    roi_area_ratios: dict[str, float]
    bvp_curves: dict[str, np.ndarray]
    bvp_values: dict[str, float]
    fft_curves: dict[str, FftCurve]
    hr_results: dict[str, object]
    bvp_curve_fusion: np.ndarray
    bvp_value_fusion: float
    hr_value_fusion: float
    fft_fusion_results: dict[str, FftFusionResult]


@dataclass(frozen=True)
class DetailedVideoResult:
    """Container returned by the detailed video entrypoint."""

    video_path: Path | None
    landmark_csv: Path | None
    output_csv: Path | None
    output_npz: Path | None
    roi_names: tuple[str, ...]
    decoded_frames: int
    frame_results: list[FrameDetailResult]
    elapsed_seconds: float


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def _zscore_or_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return np.full(values.shape, np.nan, dtype=float)
    std = float(np.std(values))
    if std <= np.finfo(float).eps:
        return np.zeros(values.shape, dtype=float)
    return (values - float(np.mean(values))) / std


@dataclass
class FramewiseHeartRateEstimator:
    """Stateful per-frame estimator.

    One instance owns the temporal BGR buffers for one video.  Create a fresh
    instance for each new video, or reset ROI state before starting a new stream.
    """

    config: EstimatorConfig = field(default_factory=EstimatorConfig)

    roi_regions: tuple[str, ...] = field(init=False)
    window_samples: int = field(init=False)
    hr_update_stride: int = field(init=False)
    use_butterworth: bool = field(init=False)
    skin_roi_exclude_mouth: bool = field(init=False)
    skin_roi_fill_holes: bool = field(init=False)
    enable_fft_fusion: bool = field(init=False)
    fft_fusion_mode: str = field(init=False)
    roi_names: tuple[str, ...] = field(init=False)
    skin_roi_builders: object | None = field(init=False, default=None)
    coeff_cache: dict[tuple[float, float, float, int], tuple[np.ndarray, np.ndarray]] = field(init=False, default_factory=dict)
    # Per-ROI sliding windows.  BVP and BPM are derived from these buffers; the
    # buffers are cleared when face/ROI data becomes invalid.
    sample_buffers: dict[str, list[np.ndarray]] = field(init=False)
    time_buffers: dict[str, list[float]] = field(init=False)
    source_frame_buffers: dict[str, list[int]] = field(init=False)
    seen_frame_indices: list[int] = field(init=False)
    last_hr: dict[str, object] = field(init=False)
    last_status: dict[str, str] = field(init=False)
    last_source_frame: dict[str, object] = field(init=False)
    last_update_offset: dict[str, int | None] = field(init=False)
    last_fusion_hr: dict[str, object] = field(init=False)
    last_fusion_status: dict[str, str] = field(init=False)
    last_fusion_source_frame: dict[str, object] = field(init=False)
    last_fusion_roi_count: dict[str, int] = field(init=False)
    last_fusion_peak_power: dict[str, object] = field(init=False)
    last_fusion_update_offset: dict[str, int | None] = field(init=False)

    def __post_init__(self) -> None:
        self.roi_regions = tuple(self.config.roi_regions)
        self.window_samples = max(1, int(self.config.window_samples))
        self.hr_update_stride = max(1, int(self.config.hr_update_stride))
        self.use_butterworth = bool(self.config.use_butterworth)
        self.skin_roi_exclude_mouth = bool(self.config.skin_roi_exclude_mouth)
        self.skin_roi_fill_holes = bool(self.config.skin_roi_fill_holes)
        self.enable_fft_fusion = bool(self.config.enable_fft_fusion)
        self.fft_fusion_mode = str(self.config.fft_fusion_mode)

        self.roi_names = tuple(roi_names_for_regions(self.roi_regions))
        active_skin_methods = skin_roi_methods_for_roi_names(self.roi_names)
        self.skin_roi_builders = make_skin_roi_builders(
            active_skin_methods,
            exclude_mouth=self.skin_roi_exclude_mouth,
            fill_holes=self.skin_roi_fill_holes,
        )
        self.sample_buffers = {name: [] for name in self.roi_names}
        self.time_buffers = {name: [] for name in self.roi_names}
        self.source_frame_buffers = {name: [] for name in self.roi_names}
        self.seen_frame_indices = []
        self.last_hr = {name: "" for name in self.roi_names}
        self.last_status = {name: "insufficient_valid_frames" for name in self.roi_names}
        self.last_source_frame = {name: "" for name in self.roi_names}
        self.last_update_offset = {name: None for name in self.roi_names}
        active_fusion_modes = self.active_fusion_modes
        self.last_fusion_hr = {mode: "" for mode in active_fusion_modes}
        self.last_fusion_status = {mode: "insufficient_valid_roi" for mode in active_fusion_modes}
        self.last_fusion_source_frame = {mode: "" for mode in active_fusion_modes}
        self.last_fusion_roi_count = {mode: 0 for mode in active_fusion_modes}
        self.last_fusion_peak_power = {mode: "" for mode in active_fusion_modes}
        self.last_fusion_update_offset = {mode: None for mode in active_fusion_modes}

    @property
    def active_fusion_modes(self) -> list[str]:
        return active_fft_fusion_modes(self.fft_fusion_mode) if self.enable_fft_fusion else []

    def fuse_bvp_curves(self, bvp_curves: dict[str, np.ndarray]) -> np.ndarray:
        prepared = []
        min_size = min((np.asarray(curve).size for curve in bvp_curves.values()), default=0)
        if min_size <= 0:
            return np.empty(0, dtype=float)
        for curve in bvp_curves.values():
            z = _zscore_or_nan(np.asarray(curve, dtype=float)[-min_size:])
            if np.all(np.isfinite(z)):
                prepared.append(z)
        if not prepared:
            return np.empty(0, dtype=float)
        return np.mean(np.vstack(prepared), axis=0)

    def fuse_bvp_values(self, bvp_values: dict[str, float]) -> float:
        return _finite_mean(bvp_values.values())

    def fuse_hr_values(self, hr_row: dict[str, object]) -> float:
        values = []
        for name in self.roi_names:
            value = hr_row.get(f"roi_{name}_hr_bpm", np.nan)
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
        values = [value for value in values if value > 0.0]
        return _finite_mean(values)

    def process_frame(self, frame_bgr: np.ndarray, frame_landmarks: FrameLandmarks) -> FrameDetailResult:
        """Process one frame and return masks, BVP/FFT details, and HR fields."""

        frame_offset = int(frame_landmarks.frame_offset)
        current_time_ms = float(frame_landmarks.frame_time_seconds) * 1000.0
        self.seen_frame_indices.append(int(frame_landmarks.frame_index))
        roi_masks, roi_bgr, roi_pixels, roi_area_ratios = self._extract_frame_roi_values(frame_bgr, frame_landmarks)
        hr_row, bvp_details, fft_fusion_results = self._update_heart_rate_state(
            frame_landmarks,
            roi_bgr,
            roi_area_ratios,
            current_time_ms,
        )
        bvp_curves = {name: detail.curve for name, detail in bvp_details.items()}
        bvp_values = {name: detail.value for name, detail in bvp_details.items()}
        fft_curves = {name: detail.fft for name, detail in bvp_details.items()}
        return FrameDetailResult(
            frame_offset=frame_offset,
            frame_index=int(frame_landmarks.frame_index),
            frame_time_seconds=float(frame_landmarks.frame_time_seconds),
            roi_masks=roi_masks,
            roi_bgr=roi_bgr,
            roi_pixel_counts=roi_pixels,
            roi_area_ratios=roi_area_ratios,
            bvp_curves=bvp_curves,
            bvp_values=bvp_values,
            fft_curves=fft_curves,
            hr_results=hr_row,
            bvp_curve_fusion=self.fuse_bvp_curves(bvp_curves),
            bvp_value_fusion=self.fuse_bvp_values(bvp_values),
            hr_value_fusion=self.fuse_hr_values(hr_row),
            fft_fusion_results=fft_fusion_results,
        )

    def process_frame_row(self, frame_bgr: np.ndarray, frame_landmarks: FrameLandmarks) -> dict[str, object]:
        """Process one frame and return only the compatible CSV row."""

        current_time_ms = float(frame_landmarks.frame_time_seconds) * 1000.0
        self.seen_frame_indices.append(int(frame_landmarks.frame_index))
        _roi_masks, roi_bgr, _roi_pixels, roi_area_ratios = self._extract_frame_roi_values(frame_bgr, frame_landmarks)
        row, _bvp_details, _fft_fusion_results = self._update_heart_rate_state(
            frame_landmarks,
            roi_bgr,
            roi_area_ratios,
            current_time_ms,
            collect_bvp_details=False,
        )
        return row

    def _extract_frame_roi_values(
        self,
        frame_bgr: np.ndarray,
        frame_landmarks: FrameLandmarks,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], dict[str, float]]:
        return extract_frame_roi_values(
            frame_bgr,
            frame_landmarks,
            roi_names=self.roi_names,
            skin_roi_builders=self.skin_roi_builders,
            skin_roi_exclude_mouth=self.skin_roi_exclude_mouth,
            skin_roi_fill_holes=self.skin_roi_fill_holes,
        )

    def _update_heart_rate_state(
        self,
        frame_landmarks: FrameLandmarks,
        roi_bgr: dict[str, np.ndarray],
        roi_area_ratios: dict[str, float],
        current_time_ms: float,
        *,
        collect_bvp_details: bool = True,
    ) -> tuple[dict[str, object], dict[str, BvpDetail], dict[str, FftFusionResult]]:
        frame_offset = int(frame_landmarks.frame_offset)
        start = max(0, frame_offset + 1 - self.window_samples)
        window_start_frame_index = (
            int(self.seen_frame_indices[start])
            if start < len(self.seen_frame_indices)
            else int(frame_landmarks.frame_index)
        )
        row: dict[str, object] = {
            "frame_index": int(frame_landmarks.frame_index),
            "frame_time_seconds": float(frame_landmarks.frame_time_seconds),
            "video_fps": float(frame_landmarks.video_fps),
            "window_start_frame_index": window_start_frame_index,
            "window_end_frame_index": int(frame_landmarks.frame_index),
        }
        bvp_details: dict[str, BvpDetail] = {}
        has_face = _frame_has_valid_face_bbox(frame_landmarks)

        for name in self.roi_names:
            area_ratio = roi_area_ratios.get(name, float("nan"))
            row[f"roi_{name}_area_ratio"] = round(float(area_ratio), 8) if np.isfinite(area_ratio) else ""
            row[f"roi_{name}_hr_bpm"] = 0.0
            row[f"roi_{name}_hr_source_frame_index"] = ""
            row[f"roi_{name}_status"] = "insufficient_valid_frames"

            # A missing/invalid face bbox means this frame cannot anchor any ROI,
            # so stale temporal samples for every ROI are discarded.
            if not has_face:
                self._reset_roi_state(name, "reset_no_face")
                row[f"roi_{name}_valid_samples"] = 0
                row[f"roi_{name}_status"] = "reset_no_face"
                continue

            current_signal = np.asarray(roi_bgr.get(name, np.full(3, np.nan)), dtype=float)
            # ROI extraction can fail independently for each ROI.  In that case
            # only this ROI's BVP window is reset; other valid ROIs keep running.
            if current_signal.shape != (3,) or not np.all(np.isfinite(current_signal)):
                self._reset_roi_state(name, "reset_empty_roi")
                row[f"roi_{name}_valid_samples"] = 0
                row[f"roi_{name}_status"] = "reset_empty_roi"
                continue
            if not np.isfinite(current_time_ms):
                row[f"roi_{name}_valid_samples"] = len(self.sample_buffers[name])
                row[f"roi_{name}_status"] = "invalid_time"
                continue
            if self.time_buffers[name] and current_time_ms <= self.time_buffers[name][-1]:
                row[f"roi_{name}_valid_samples"] = len(self.sample_buffers[name])
                row[f"roi_{name}_status"] = "invalid_time"
                continue

            # Keep at most window_samples valid BGR samples per ROI.
            self.sample_buffers[name].append(current_signal.astype(float, copy=True))
            self.time_buffers[name].append(current_time_ms)
            self.source_frame_buffers[name].append(int(frame_landmarks.frame_index))
            if len(self.sample_buffers[name]) > self.window_samples:
                del self.sample_buffers[name][0]
                del self.time_buffers[name][0]
                del self.source_frame_buffers[name][0]

            valid_count = len(self.sample_buffers[name])
            row[f"roi_{name}_valid_samples"] = valid_count
            # Four samples are enough to expose a BVP/FFT detail curve, while
            # BPM follows the validated full-window rule below.
            if collect_bvp_details and valid_count >= 4:
                try:
                    bvp_details[name] = compute_bvp_detail(
                        np.asarray(self.sample_buffers[name], dtype=float),
                        np.asarray(self.time_buffers[name], dtype=float),
                        use_butterworth=self.use_butterworth,
                        coeff_cache=self.coeff_cache,
                    )
                except Exception as exc:
                    LOGGER.debug(
                        "BVP detail computation failed for ROI %s at frame %s: %s",
                        name,
                        frame_landmarks.frame_index,
                        exc,
                        exc_info=True,
                    )
            if valid_count < self.window_samples:
                continue

            previous_update = self.last_update_offset[name]
            if previous_update is not None and frame_offset - previous_update < self.hr_update_stride:
                row[f"roi_{name}_hr_bpm"] = self.last_hr[name] if self.last_hr[name] != "" else 0.0
                row[f"roi_{name}_hr_source_frame_index"] = self.last_source_frame[name]
                row[f"roi_{name}_status"] = "held" if self.last_hr[name] != "" else self.last_status[name]
                continue

            try:
                hr_bpm = estimate_heart_rate_bgr(
                    np.asarray(self.sample_buffers[name], dtype=float),
                    np.asarray(self.time_buffers[name], dtype=float),
                    use_butterworth=self.use_butterworth,
                )
            except Exception as exc:
                row[f"roi_{name}_status"] = f"hr_failed:{type(exc).__name__}"
                row[f"roi_{name}_hr_bpm"] = 0.0
                self.last_status[name] = str(row[f"roi_{name}_status"])
                self.last_update_offset[name] = frame_offset
                continue

            if hr_bpm > 0.0:
                row[f"roi_{name}_hr_bpm"] = round(float(hr_bpm), 4)
                row[f"roi_{name}_status"] = "ok"
                row[f"roi_{name}_hr_source_frame_index"] = int(frame_landmarks.frame_index)
            else:
                row[f"roi_{name}_hr_bpm"] = 0.0
                row[f"roi_{name}_status"] = "invalid_hr"
                row[f"roi_{name}_hr_source_frame_index"] = ""
            self.last_hr[name] = row[f"roi_{name}_hr_bpm"]
            self.last_status[name] = str(row[f"roi_{name}_status"])
            self.last_source_frame[name] = row[f"roi_{name}_hr_source_frame_index"]
            self.last_update_offset[name] = frame_offset

        fft_fusion_results = self._update_fft_fusion(frame_landmarks, row, has_face)
        return row, bvp_details, fft_fusion_results

    def _reset_roi_state(self, name: str, status: str) -> None:
        """Clear one ROI's temporal state after an invalid face/ROI event."""

        self.sample_buffers[name].clear()
        self.time_buffers[name].clear()
        self.source_frame_buffers[name].clear()
        self.last_hr[name] = ""
        self.last_status[name] = status
        self.last_source_frame[name] = ""
        self.last_update_offset[name] = None

    def _update_fft_fusion(
        self,
        frame_landmarks: FrameLandmarks,
        row: dict[str, object],
        has_face: bool,
    ) -> dict[str, FftFusionResult]:
        frame_offset = int(frame_landmarks.frame_offset)
        results: dict[str, FftFusionResult] = {}
        active_fusion_modes = self.active_fusion_modes
        if not active_fusion_modes:
            return results
        if not has_face:
            for mode in active_fusion_modes:
                self.last_fusion_hr[mode] = ""
                self.last_fusion_status[mode] = "reset_no_face"
                self.last_fusion_source_frame[mode] = ""
                self.last_fusion_roi_count[mode] = 0
                self.last_fusion_peak_power[mode] = ""
                self.last_fusion_update_offset[mode] = None

        fusion_times = None
        for name in self.roi_names:
            if len(self.time_buffers[name]) >= self.window_samples:
                fusion_times = np.asarray(self.time_buffers[name], dtype=float)
                break
        fusion_windows: dict[str, np.ndarray] = {}
        if fusion_times is not None:
            for name in self.roi_names:
                if len(self.sample_buffers[name]) < self.window_samples or len(self.time_buffers[name]) < self.window_samples:
                    continue
                roi_times = np.asarray(self.time_buffers[name], dtype=float)
                if roi_times.shape != fusion_times.shape or not np.allclose(roi_times, fusion_times):
                    continue
                fusion_windows[name] = np.asarray(self.sample_buffers[name], dtype=float)

        for mode in active_fusion_modes:
            prefix = fft_fusion_output_prefix(mode)
            row[f"{prefix}_hr_bpm"] = 0.0
            row[f"{prefix}_roi_count"] = 0
            row[f"{prefix}_peak_power"] = ""
            row[f"{prefix}_hr_source_frame_index"] = ""
            row[f"{prefix}_status"] = "insufficient_valid_roi"
            previous_update = self.last_fusion_update_offset[mode]
            if previous_update is not None and frame_offset - previous_update < self.hr_update_stride:
                row[f"{prefix}_hr_bpm"] = self.last_fusion_hr[mode] if self.last_fusion_hr[mode] != "" else 0.0
                row[f"{prefix}_roi_count"] = self.last_fusion_roi_count[mode]
                row[f"{prefix}_peak_power"] = self.last_fusion_peak_power[mode]
                row[f"{prefix}_hr_source_frame_index"] = self.last_fusion_source_frame[mode]
                row[f"{prefix}_status"] = "held" if self.last_fusion_hr[mode] != "" else self.last_fusion_status[mode]
                continue
            if fusion_times is None or not fusion_windows:
                continue
            try:
                fusion_result = estimate_fft_fused_heart_rate(
                    fusion_windows,
                    fusion_times,
                    mode=mode,
                    use_butterworth=self.use_butterworth,
                    coeff_cache=self.coeff_cache,
                )
            except Exception as exc:
                row[f"{prefix}_status"] = f"hr_failed:{type(exc).__name__}"
                self.last_fusion_status[mode] = str(row[f"{prefix}_status"])
                self.last_fusion_update_offset[mode] = frame_offset
                continue

            results[mode] = fusion_result
            row[f"{prefix}_roi_count"] = int(fusion_result.roi_count)
            row[f"{prefix}_peak_power"] = (
                round(float(fusion_result.selected_power), 8) if np.isfinite(fusion_result.selected_power) else ""
            )
            if fusion_result.status == "ok" and fusion_result.hr_bpm > 0.0:
                row[f"{prefix}_hr_bpm"] = round(float(fusion_result.hr_bpm), 4)
                row[f"{prefix}_hr_source_frame_index"] = int(frame_landmarks.frame_index)
                row[f"{prefix}_status"] = "ok"
            else:
                row[f"{prefix}_hr_bpm"] = 0.0
                row[f"{prefix}_hr_source_frame_index"] = ""
                row[f"{prefix}_status"] = fusion_result.status
            self.last_fusion_hr[mode] = row[f"{prefix}_hr_bpm"] if row[f"{prefix}_status"] == "ok" else ""
            self.last_fusion_status[mode] = str(row[f"{prefix}_status"])
            self.last_fusion_source_frame[mode] = row[f"{prefix}_hr_source_frame_index"]
            self.last_fusion_roi_count[mode] = int(row[f"{prefix}_roi_count"])
            self.last_fusion_peak_power[mode] = row[f"{prefix}_peak_power"]
            self.last_fusion_update_offset[mode] = frame_offset
        return results
