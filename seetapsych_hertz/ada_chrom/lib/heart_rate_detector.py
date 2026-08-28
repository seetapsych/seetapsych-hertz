"""BGR CHROM heart-rate estimation ported from the C++ SDK source."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

from .roi_mask_utils import (
    _load_cv2,
    _mask_active,
    _points_to_array,
    _trunc_div,
    _update_bbox,
    combine_mask,
    get_bgr,
    initialize_facearea_roi,
    skin_detection,
)


PI = 3.14159


def compute_lp(filter_order: int) -> np.ndarray:
    """Port of ComputeLP from nxsEdson/Butterworth-Filter."""
    coeffs = np.zeros(filter_order + 1, dtype=float)
    coeffs[0] = 1.0
    coeffs[1] = float(filter_order)
    m = filter_order // 2

    for i in range(2, m + 1):
        coeffs[i] = float(filter_order - i + 1) * coeffs[i - 1] / float(i)
        coeffs[filter_order - i] = coeffs[i]

    coeffs[filter_order - 1] = float(filter_order)
    coeffs[filter_order] = 1.0
    return coeffs


def compute_hp(filter_order: int) -> np.ndarray:
    """Port of ComputeHP from nxsEdson/Butterworth-Filter."""
    coeffs = compute_lp(filter_order)
    for i in range(filter_order + 1):
        if i % 2:
            coeffs[i] = -coeffs[i]
    return coeffs


def trinomial_multiply(filter_order: int, b: Iterable[float], c: Iterable[float]) -> np.ndarray:
    """Port of TrinomialMultiply from nxsEdson/Butterworth-Filter."""
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ret = np.zeros(4 * filter_order, dtype=float)

    ret[2] = c[0]
    ret[3] = c[1]
    ret[0] = b[0]
    ret[1] = b[1]

    for i in range(1, filter_order):
        ret[2 * (2 * i + 1)] += (
            c[2 * i] * ret[2 * (2 * i - 1)]
            - c[2 * i + 1] * ret[2 * (2 * i - 1) + 1]
        )
        ret[2 * (2 * i + 1) + 1] += (
            c[2 * i] * ret[2 * (2 * i - 1) + 1]
            + c[2 * i + 1] * ret[2 * (2 * i - 1)]
        )

        for j in range(2 * i, 1, -1):
            ret[2 * j] += (
                b[2 * i] * ret[2 * (j - 1)]
                - b[2 * i + 1] * ret[2 * (j - 1) + 1]
                + c[2 * i] * ret[2 * (j - 2)]
                - c[2 * i + 1] * ret[2 * (j - 2) + 1]
            )
            ret[2 * j + 1] += (
                b[2 * i] * ret[2 * (j - 1) + 1]
                + b[2 * i + 1] * ret[2 * (j - 1)]
                + c[2 * i] * ret[2 * (j - 2) + 1]
                + c[2 * i + 1] * ret[2 * (j - 2)]
            )

        ret[2] += b[2 * i] * ret[0] - b[2 * i + 1] * ret[1] + c[2 * i]
        ret[3] += b[2 * i] * ret[1] + b[2 * i + 1] * ret[0] + c[2 * i + 1]
        ret[0] += b[2 * i]
        ret[1] += b[2 * i + 1]

    return ret


def compute_den_coeffs(filter_order: int, lcutoff: float, ucutoff: float) -> np.ndarray:
    """Port of ComputeDenCoeffs, trimmed to the coefficients used by filter()."""
    r_coeffs = np.zeros(2 * filter_order, dtype=float)
    t_coeffs = np.zeros(2 * filter_order, dtype=float)

    cp = math.cos(PI * (ucutoff + lcutoff) / 2.0)
    theta = PI * (ucutoff - lcutoff) / 2.0
    st = math.sin(theta)
    ct = math.cos(theta)
    s2t = 2.0 * st * ct
    c2t = 2.0 * ct * ct - 1.0

    for k in range(filter_order):
        pole_angle = PI * float(2 * k + 1) / float(2 * filter_order)
        sin_pole_angle = math.sin(pole_angle)
        cos_pole_angle = math.cos(pole_angle)
        work = 1.0 + s2t * sin_pole_angle
        r_coeffs[2 * k] = c2t / work
        r_coeffs[2 * k + 1] = s2t * cos_pole_angle / work
        t_coeffs[2 * k] = -2.0 * cp * (ct + st * sin_pole_angle) / work
        t_coeffs[2 * k + 1] = -2.0 * cp * st * cos_pole_angle / work

    den = trinomial_multiply(filter_order, t_coeffs, r_coeffs)
    den[1] = den[0]
    den[0] = 1.0
    for k in range(3, 2 * filter_order + 1):
        den[k] = den[2 * k - 2]

    return den[: 2 * filter_order + 1].copy()


def compute_num_coeffs(
    filter_order: int,
    lcutoff: float,
    ucutoff: float,
    den_coeffs: Iterable[float],
) -> np.ndarray:
    """Port of ComputeNumCoeffs from nxsEdson/Butterworth-Filter."""
    den_coeffs = np.asarray(den_coeffs, dtype=float)
    num_coeffs = np.zeros(2 * filter_order + 1, dtype=float)
    normalized_kernel = np.zeros(2 * filter_order + 1, dtype=complex)
    numbers = np.arange(2 * filter_order + 1, dtype=float)

    t_coeffs = compute_hp(filter_order)
    for i in range(filter_order):
        num_coeffs[2 * i] = t_coeffs[i]
        num_coeffs[2 * i + 1] = 0.0
    num_coeffs[2 * filter_order] = t_coeffs[filter_order]

    cp0 = 2.0 * 2.0 * math.tan(PI * lcutoff / 2.0)
    cp1 = 2.0 * 2.0 * math.tan(PI * ucutoff / 2.0)
    wn = math.sqrt(cp0 * cp1)
    wn = 2.0 * math.atan2(wn, 4.0)

    for k in range(2 * filter_order + 1):
        normalized_kernel[k] = np.exp(-1j * wn * numbers[k])

    numerator_gain = 0.0
    denominator_gain = 0.0
    for d in range(2 * filter_order + 1):
        numerator_gain += float(np.real(normalized_kernel[d] * num_coeffs[d]))
        denominator_gain += float(np.real(normalized_kernel[d] * den_coeffs[d]))

    if abs(numerator_gain) < np.finfo(float).eps:
        raise ZeroDivisionError("Butterworth numerator gain is too close to zero.")

    return (num_coeffs * denominator_gain) / numerator_gain


def apply_butterworth_filter(
    x: Iterable[float],
    coeff_b: Iterable[float],
    coeff_a: Iterable[float],
) -> np.ndarray:
    """Apply SciPy Butterworth filtering with zero initial state."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.asarray(x, dtype=float).copy()
    from scipy.signal import lfilter

    return np.asarray(lfilter(coeff_b, coeff_a, x), dtype=float)


@lru_cache(maxsize=64)
def _butterworth_coefficients(
    fps: float,
    lowpass: float,
    highpass: float,
    filt_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    frequency_bands = (lowpass / fps * 2.0, highpass / fps * 2.0)
    a = compute_den_coeffs(filt_order, frequency_bands[0], frequency_bands[1])
    b = compute_num_coeffs(filt_order, frequency_bands[0], frequency_bands[1], a)
    return b, a


def chrominance_based_method(
    signal_bgr: Iterable[Iterable[float]],
    b: Iterable[float] | None = None,
    a: Iterable[float] | None = None,
    *,
    use_butterworth: bool = False,
) -> np.ndarray:
    """Return one CHROM pulse signal from BGR channel traces."""
    signal = _as_channels_first_bgr(signal_bgr)
    b_chan, g_chan, r_chan = signal

    b_mean = float(np.mean(b_chan))
    g_mean = float(np.mean(g_chan))
    r_mean = float(np.mean(r_chan))
    if min(abs(b_mean), abs(g_mean), abs(r_mean)) < np.finfo(float).eps:
        raise ValueError("BGR channel means must be non-zero for CHROM normalization.")

    b_norm = b_chan / b_mean
    g_norm = g_chan / g_mean
    r_norm = r_chan / r_mean

    x = 3.0 * r_norm - 2.0 * g_norm
    y = 1.5 * r_norm + g_norm - 1.5 * b_norm

    if use_butterworth:
        if b is None or a is None:
            raise ValueError("b and a coefficients are required when use_butterworth=True.")
        x = apply_butterworth_filter(x, b, a)
        y = apply_butterworth_filter(y, b, a)

    y_std = float(np.std(y))
    alpha = 0.0 if y_std < np.finfo(float).eps else float(np.std(x)) / y_std
    pulse = x - alpha * y
    pulse_mean = float(np.mean(pulse))
    return np.asarray([(pulse - pulse_mean) * 150.0], dtype=float)


@dataclass(frozen=True)
class FftCurve:
    """FFT spectrum for one selected CHROM pulse channel."""

    bpm: np.ndarray
    magnitude: np.ndarray
    power: np.ndarray
    selected_bpm: float
    selected_magnitude: float


@dataclass(frozen=True)
class BvpDetail:
    """Intermediate BVP output returned for one ROI window."""

    time_ms: np.ndarray
    channels: np.ndarray
    selected_channel: int
    curve: np.ndarray
    value: float
    fft: FftCurve
    fps: float


def compute_chrom_bvp_channels(
    signal_bgr: Iterable[Iterable[float]],
    sample_time_ms: Iterable[float],
    *,
    lowpass: float = 0.8,
    highpass: float = 4.0,
    filt_order: int = 4,
    use_butterworth: bool = False,
    coeff_cache: dict[tuple[float, float, float, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Prepare CHROM BVP channels on an evenly spaced time grid.

    Both the compact BPM API and the detailed per-frame API call this helper so
    their interpolation, smoothing, and optional filtering paths stay identical.
    """

    signal = _as_channels_first_bgr(signal_bgr)
    sample_time_ms = np.asarray(sample_time_ms, dtype=float)
    if signal.shape[1] != sample_time_ms.size:
        raise ValueError("sample_time_ms length must match the BGR signal length.")
    if sample_time_ms.size < 4:
        raise ValueError("At least 4 samples are required to compute BVP.")
    if np.any(np.diff(sample_time_ms) <= 0.0):
        raise ValueError("sample_time_ms must be strictly increasing.")

    duration_ms = float(sample_time_ms[-1] - sample_time_ms[0])
    if duration_ms <= 0.0:
        raise ValueError("BVP sample duration must be positive.")
    fps = float(sample_time_ms.size) / duration_ms * 1000.0

    even_time = _cpp_even_time_grid(sample_time_ms, sample_time_ms.size)
    even_signal = np.vstack(
        [_natural_cubic_interp(sample_time_ms, channel, even_time) for channel in signal]
    )
    even_signal = _gaussian_smooth_channels(even_signal, kernel_size=3, sigma=1.0)

    if use_butterworth:
        cache_key = (round(fps, 6), float(lowpass), float(highpass), int(filt_order))
        if coeff_cache is not None:
            if cache_key not in coeff_cache:
                frequency_bands = (lowpass / fps * 2.0, highpass / fps * 2.0)
                a = compute_den_coeffs(filt_order, frequency_bands[0], frequency_bands[1])
                b = compute_num_coeffs(filt_order, frequency_bands[0], frequency_bands[1], a)
                coeff_cache[cache_key] = (b, a)
            b, a = coeff_cache[cache_key]
        else:
            b, a = _butterworth_coefficients(float(fps), float(lowpass), float(highpass), int(filt_order))
    else:
        b = None
        a = None

    process_signal = chrominance_based_method(
        even_signal,
        b=b,
        a=a,
        use_butterworth=use_butterworth,
    )
    process_signal = _gaussian_smooth_channels(process_signal, kernel_size=3, sigma=1.0)
    return process_signal, float(fps), even_time


def compute_bvp_detail(
    signal_bgr: Iterable[Iterable[float]],
    sample_time_ms: Iterable[float],
    *,
    use_butterworth: bool = False,
    lowpass: float = 0.8,
    highpass: float = 4.0,
    filt_order: int = 4,
    coeff_cache: dict[tuple[float, float, float, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> BvpDetail:
    """Return BVP curve/value plus FFT spectrum for the strongest channel."""

    process_signal, fps, even_time = compute_chrom_bvp_channels(
        signal_bgr,
        sample_time_ms,
        lowpass=lowpass,
        highpass=highpass,
        filt_order=filt_order,
        use_butterworth=use_butterworth,
        coeff_cache=coeff_cache,
    )

    best_channel = 0
    best_bpm = -1.0
    best_mag = -np.inf
    best_curve = np.empty(0, dtype=float)
    best_fft = FftCurve(
        bpm=np.empty(0, dtype=float),
        magnitude=np.empty(0, dtype=float),
        power=np.empty(0, dtype=float),
        selected_bpm=-1.0,
        selected_magnitude=-np.inf,
    )
    for channel_index, pulse in enumerate(process_signal):
        bpm, max_fft = _estimate_bpm_from_fft(pulse, fps)
        pre_fft = _cpp_hamming_pre_fft(pulse)
        magnitude = np.abs(np.fft.rfft(pre_fft))
        hz = np.fft.rfftfreq(pulse.size, d=1.0 / fps)
        curve = FftCurve(
            bpm=hz * 60.0,
            magnitude=magnitude,
            power=magnitude * magnitude,
            selected_bpm=float(bpm),
            selected_magnitude=float(max_fft),
        )
        if max_fft > best_mag:
            best_channel = channel_index
            best_bpm = float(bpm)
            best_mag = float(max_fft)
            best_curve = np.asarray(pulse, dtype=float)
            best_fft = curve

    return BvpDetail(
        time_ms=even_time.astype(float, copy=True),
        channels=np.asarray(process_signal, dtype=float),
        selected_channel=int(best_channel),
        curve=best_curve.astype(float, copy=True),
        value=float(best_curve[-1]) if best_curve.size else float("nan"),
        fft=FftCurve(
            bpm=best_fft.bpm.copy(),
            magnitude=best_fft.magnitude.copy(),
            power=best_fft.power.copy(),
            selected_bpm=best_bpm,
            selected_magnitude=best_mag,
        ),
        fps=float(fps),
    )


def estimate_heart_rate_bgr(
    signal_bgr: Iterable[Iterable[float]],
    sample_time_ms: Iterable[float],
    *,
    lowpass: float = 0.8,
    highpass: float = 4.0,
    filt_order: int = 4,
    use_butterworth: bool = False,
) -> float:
    """Estimate BPM from OpenCV-style BGR traces."""
    signal = _as_channels_first_bgr(signal_bgr)
    sample_time_ms = np.asarray(sample_time_ms, dtype=float)
    if signal.shape[1] != sample_time_ms.size:
        raise ValueError("sample_time_ms length must match the BGR signal length.")
    if sample_time_ms.size < 4:
        return -1.0
    if np.any(np.diff(sample_time_ms) <= 0.0):
        raise ValueError("sample_time_ms must be strictly increasing.")

    duration_ms = float(sample_time_ms[-1] - sample_time_ms[0])
    if duration_ms <= 0.0:
        return -1.0
    process_signal, fps, _even_time = compute_chrom_bvp_channels(
        signal,
        sample_time_ms,
        lowpass=lowpass,
        highpass=highpass,
        filt_order=filt_order,
        use_butterworth=use_butterworth,
    )

    best_bpm = -1.0
    best_fft = -np.inf
    for pulse in process_signal:
        bpm, max_fft = _estimate_bpm_from_fft(pulse, fps)
        if max_fft > best_fft:
            best_bpm = bpm
            best_fft = max_fft

    return float(best_bpm)


def _estimate_bpm_from_fft(pulse: np.ndarray, fps: float) -> tuple[float, float]:
    temp_n = pulse.size
    if temp_n < 4:
        return -1.0, -np.inf

    pre_fft = _cpp_hamming_pre_fft(pulse)
    mag = np.abs(np.fft.rfft(pre_fft))
    freqs = np.fft.rfftfreq(temp_n, d=1.0 / fps)

    limit_left = 10.0 / 60.0
    limit_right = 120.0 / 60.0
    peak_left = limit_left + 40.0 / 60.0
    mask = (freqs > peak_left) & (freqs < limit_right)
    if not np.any(mask):
        return -1.0, -np.inf

    masked_indices = np.flatnonzero(mask)
    idx = masked_indices[int(np.argmax(mag[mask]))]
    return float(freqs[idx] * 60.0), float(mag[idx])


def _cpp_even_time_grid(sample_time_ms: np.ndarray, n: int) -> np.ndarray:
    sample_time_ms = np.asarray(sample_time_ms, dtype=float)
    if sample_time_ms.size == 0:
        return np.empty(0, dtype=float)
    if n <= 0:
        raise ValueError("n must be positive.")
    delta = float(sample_time_ms[-1] - sample_time_ms[0]) / float(n)
    return sample_time_ms[0] + delta * np.arange(n, dtype=float)


def _cpp_hamming_pre_fft(pulse: Iterable[float]) -> np.ndarray:
    pulse = np.asarray(pulse, dtype=float)
    n = pulse.size
    if n <= 0:
        return np.empty(0, dtype=float)
    if n == 1:
        windowed = 0.08 * pulse
    else:
        indices = np.arange(n, dtype=float)
        window = 0.54 - 0.46 * np.cos(2.0 * PI * indices / float(n - 1))
        windowed = pulse * window
    return windowed - float(np.mean(windowed))


def _as_channels_first_bgr(signal_bgr: Iterable[Iterable[float]]) -> np.ndarray:
    signal = np.asarray(signal_bgr, dtype=float)
    if signal.ndim != 2:
        raise ValueError("signal_bgr must be a 2D array.")
    if signal.shape[0] == 3 and signal.shape[1] != 3:
        return signal.copy()
    if signal.shape[1] == 3:
        return signal.T.copy()
    if signal.shape[0] == 3:
        return signal.copy()
    raise ValueError("signal_bgr must have shape (N, 3) or (3, N).")


def _gaussian_smooth_channels(
    signal: np.ndarray,
    *,
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    cv2 = _load_cv2()
    if cv2 is not None:
        smoothed = []
        for channel in signal:
            blurred = cv2.GaussianBlur(
                np.asarray(channel, dtype=float).reshape(1, -1),
                (int(kernel_size), 1),
                float(sigma),
                float(sigma),
                borderType=cv2.BORDER_DEFAULT,
            )
            smoothed.append(blurred.reshape(-1))
        return np.vstack(smoothed)
    kernel = _gaussian_kernel(kernel_size, sigma)
    return np.vstack([np.convolve(channel, kernel, mode="same") for channel in signal])


def _gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    if kernel_size <= 1:
        return np.ones(1, dtype=float)
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)


def _natural_cubic_interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return np.interp(x_new, x, y)

    n = x.size
    h = np.diff(x)
    alpha = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        alpha[i] = 3.0 / h[i] * (y[i + 1] - y[i]) - 3.0 / h[i - 1] * (y[i] - y[i - 1])

    l = np.ones(n, dtype=float)
    mu = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c = np.zeros(n, dtype=float)
    b = np.zeros(n - 1, dtype=float)
    d = np.zeros(n - 1, dtype=float)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    indices = np.searchsorted(x, x_new, side="right") - 1
    indices = np.clip(indices, 0, n - 2)
    dx = x_new - x[indices]
    return y[indices] + b[indices] * dx + c[indices] * dx * dx + d[indices] * dx * dx * dx
