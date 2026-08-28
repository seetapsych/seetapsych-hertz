"""FFT-based ROI heart-rate fusion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .heart_rate_detector import (
    _as_channels_first_bgr,
    _cpp_even_time_grid,
    _cpp_hamming_pre_fft,
    _gaussian_smooth_channels,
    _natural_cubic_interp,
    chrominance_based_method,
    compute_den_coeffs,
    compute_num_coeffs,
)
from .roi_definitions import FFT_FUSION_MODES


@dataclass
class FftFusionResult:
    mode: str
    hr_bpm: float
    status: str
    roi_count: int = 0
    selected_power: float = 0.0


def active_fft_fusion_modes(fft_fusion_mode: str) -> list[str]:
    if fft_fusion_mode not in FFT_FUSION_MODES:
        raise ValueError(f"Unknown FFT fusion mode: {fft_fusion_mode}")
    if fft_fusion_mode == "both":
        return ["bvp-zscore", "bandpower-normalized"]
    return [fft_fusion_mode]


def fft_fusion_output_prefix(mode: str) -> str:
    if mode == "bvp-zscore":
        return "fft_fusion_bvp_zscore"
    if mode == "bandpower-normalized":
        return "fft_fusion_bandpower_norm"
    raise ValueError(f"Unknown FFT fusion mode: {mode}")


def _zscore_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    std = float(np.std(signal))
    if std <= np.finfo(float).eps:
        return np.zeros_like(signal, dtype=float)
    return (signal - float(np.mean(signal))) / std


def _normalize_band_power(power: np.ndarray, band_mask: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(power, dtype=float)
    denom = float(np.sum(power[band_mask]))
    if denom > np.finfo(float).eps:
        normalized[band_mask] = power[band_mask] / denom
    return normalized


def compute_fft_fusion_from_bvp(
    pulses_by_roi: dict[str, Iterable[float]],
    fps: float,
    *,
    mode: str,
    bpm_min: float = 50.0,
    bpm_max: float = 120.0,
) -> FftFusionResult:
    """Fuse per-ROI BVP spectra and pick the fused peak BPM."""
    if mode not in ("bvp-zscore", "bandpower-normalized"):
        raise ValueError(f"Unknown FFT fusion mode: {mode}")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        return FftFusionResult(mode=mode, hr_bpm=-1.0, status="invalid_fps")

    fused_parts: list[np.ndarray] = []
    reference_bpm: np.ndarray | None = None
    reference_band: np.ndarray | None = None

    for raw_pulse in pulses_by_roi.values():
        pulse = np.asarray(raw_pulse, dtype=float)
        pulse = pulse[np.isfinite(pulse)]
        if pulse.size < 4:
            continue
        analysis_pulse = _zscore_signal(pulse) if mode == "bvp-zscore" else pulse
        pre_fft = _cpp_hamming_pre_fft(analysis_pulse)
        magnitude = np.abs(np.fft.rfft(pre_fft))
        power = magnitude * magnitude
        hz = np.fft.rfftfreq(pulse.size, d=1.0 / fps)
        bpm = hz * 60.0
        band_mask = (bpm > float(bpm_min)) & (bpm < float(bpm_max))
        if not np.any(band_mask):
            continue
        normalized_power = _normalize_band_power(power, band_mask)
        curve_power = normalized_power if mode == "bandpower-normalized" else power
        if float(np.sum(curve_power[band_mask])) <= np.finfo(float).eps:
            continue
        if reference_bpm is None:
            reference_bpm = bpm
            reference_band = band_mask
        elif bpm.shape != reference_bpm.shape or not np.allclose(bpm, reference_bpm):
            continue
        fused_parts.append(curve_power[band_mask])

    if reference_bpm is None or reference_band is None or not fused_parts:
        return FftFusionResult(mode=mode, hr_bpm=-1.0, status="insufficient_valid_roi")

    fused_power = np.mean(np.vstack(fused_parts), axis=0)
    selected = int(np.argmax(fused_power))
    band_bpm = reference_bpm[reference_band]
    return FftFusionResult(
        mode=mode,
        hr_bpm=float(band_bpm[selected]),
        status="ok",
        roi_count=len(fused_parts),
        selected_power=float(fused_power[selected]),
    )


def _regular_fps_from_sample_times(sample_time_ms: np.ndarray) -> tuple[float, bool]:
    if sample_time_ms.size < 4:
        return -1.0, False
    diffs = np.diff(sample_time_ms)
    if np.any(diffs <= 0.0):
        raise ValueError("sample_time_ms must be strictly increasing.")
    duration_ms = float(sample_time_ms[-1] - sample_time_ms[0])
    if duration_ms <= 0.0:
        return -1.0, False
    fps = float(sample_time_ms.size) / duration_ms * 1000.0
    regular = bool(diffs.size and np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-4))
    return fps, regular


def _prepare_even_signal_and_time_for_fft_fusion(
    signal_bgr: Iterable[Iterable[float]],
    sample_time_ms: np.ndarray,
    *,
    regular_times: bool,
) -> tuple[np.ndarray, np.ndarray]:
    signal = _as_channels_first_bgr(signal_bgr)
    if signal.shape[1] != sample_time_ms.size:
        raise ValueError("sample_time_ms length must match the BGR signal length.")
    if regular_times:
        return _gaussian_smooth_channels(signal, kernel_size=3, sigma=1.0), sample_time_ms.astype(float, copy=True)
    even_time_ms = _cpp_even_time_grid(sample_time_ms, sample_time_ms.size)
    even_signal = np.vstack([_natural_cubic_interp(sample_time_ms, channel, even_time_ms) for channel in signal])
    return _gaussian_smooth_channels(even_signal, kernel_size=3, sigma=1.0), even_time_ms


def estimate_fft_fused_heart_rate(
    roi_bgr_windows: dict[str, np.ndarray],
    sample_time_ms: Iterable[float],
    *,
    mode: str,
    use_butterworth: bool = False,
    lowpass: float = 0.8,
    highpass: float = 4.0,
    filt_order: int = 4,
    coeff_cache: dict[tuple[float, float, float, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> FftFusionResult:
    sample_time_ms = np.asarray(sample_time_ms, dtype=float)
    fps, regular_times = _regular_fps_from_sample_times(sample_time_ms)
    if fps <= 0.0:
        return FftFusionResult(mode=mode, hr_bpm=-1.0, status="invalid_time")

    if use_butterworth:
        cache_key = (round(fps, 6), float(lowpass), float(highpass), int(filt_order))
        if coeff_cache is not None and cache_key in coeff_cache:
            b, a = coeff_cache[cache_key]
        else:
            frequency_bands = (lowpass / fps * 2.0, highpass / fps * 2.0)
            a = compute_den_coeffs(filt_order, frequency_bands[0], frequency_bands[1])
            b = compute_num_coeffs(filt_order, frequency_bands[0], frequency_bands[1], a)
            if coeff_cache is not None:
                coeff_cache[cache_key] = (b, a)
    else:
        b = None
        a = None

    pulses: dict[str, np.ndarray] = {}
    for name, signal in roi_bgr_windows.items():
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 2 or signal.shape[0] < 4 or signal.shape[1] != 3:
            continue
        if not np.all(np.isfinite(signal)):
            continue
        even_signal, _even_time_ms = _prepare_even_signal_and_time_for_fft_fusion(
            signal,
            sample_time_ms,
            regular_times=regular_times,
        )
        process_signal = chrominance_based_method(
            even_signal,
            b=b,
            a=a,
            use_butterworth=use_butterworth,
        )
        pulse = _gaussian_smooth_channels(process_signal, kernel_size=3, sigma=1.0)[0]
        pulses[str(name)] = pulse

    return compute_fft_fusion_from_bvp(
        pulses,
        fps,
        mode=mode,
    )
