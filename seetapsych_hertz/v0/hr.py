# -*- coding: utf-8 -*-

import cv2
import numpy


def _generate_interp_times(start: float, end: float, step: float) -> numpy.ndarray:
    """
    Generate interpolation timestamps.

    Floating point accumulation is intentionally used to preserve the
    sampling behavior when the input interval cannot be represented exactly.
    """
    result = []
    current = start

    while current < end:
        result.append(current)
        current += step

    return numpy.asarray(result, dtype=numpy.float64)


def _cubic_spline_interp_1d(
        values: numpy.ndarray, times: numpy.ndarray, target_times: numpy.ndarray
) -> numpy.ndarray:
    """
    Cubic spline interpolation for one-dimensional signals.

    :param values: Original signal values.
    :param times: Original timestamps.
    :param target_times: Target interpolation timestamps.
    """
    values = numpy.asarray(values, dtype=numpy.float64)
    times = numpy.asarray(times, dtype=numpy.float64)

    n = len(values)

    if n < 2:
        raise ValueError('At least two samples are required')

    h = numpy.diff(times)

    if numpy.any(h <= 0):
        raise ValueError('Timestamps must be strictly increasing')

    # Calculate second derivatives for natural cubic spline.
    second = numpy.zeros(n, dtype=numpy.float64)
    inner = n - 2

    if inner > 0:
        diagonal = 2.0 * (h[:-1] + h[1:])
        rhs = 6.0 * ((values[2:] - values[1:-1]) / h[1:] - (values[1:-1] - values[:-2]) / h[:-1])

        if inner == 1:
            second[1] = rhs[0] / diagonal[0]
        else:
            c = numpy.zeros(inner - 1, dtype=numpy.float64)
            d = numpy.zeros(inner, dtype=numpy.float64)

            c[0] = h[1] / diagonal[0]
            d[0] = rhs[0] / diagonal[0]

            for i in range(1, inner):
                denominator = diagonal[i] - h[i] * c[i - 1]

                if i < inner - 1:
                    c[i] = h[i + 1] / denominator

                d[i] = (rhs[i] - h[i] * d[i - 1]) / denominator

            second[-2] = d[-1]

            for i in range(inner - 2, -1, -1):
                second[i + 1] = d[i] - c[i] * second[i + 2]

    index = numpy.searchsorted(times, target_times, side='right') - 1
    index = numpy.clip(index, 0, n - 2)

    dt = target_times - times[index]
    hi = h[index]

    a0 = values[index]
    a1 = (
            (values[index + 1] - values[index]) / hi
            - hi * second[index] / 2.0
            - hi * (second[index + 1] - second[index]) / 6.0
    )
    a2 = second[index] / 2.0
    a3 = (second[index + 1] - second[index]) / (6.0 * hi)

    return a0 + a1 * dt + a2 * dt ** 2 + a3 * dt ** 3


def _gaussian_blur_1d(signal: numpy.ndarray) -> numpy.ndarray:
    """
    Apply one-dimensional Gaussian smoothing.
    """
    return cv2.GaussianBlur(signal.reshape(1, -1), (3, 1), 1, 1).reshape(-1)


def _chrominance_method(signal: numpy.ndarray) -> numpy.ndarray:
    """
    Extract pulse signal using chrominance analysis.

    :param signal: Shape [3, N], channel order is B, G, R.
    """
    b = signal[0]
    g = signal[1]
    r = signal[2]

    b = b / numpy.mean(b)
    g = g / numpy.mean(g)
    r = r / numpy.mean(r)

    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b
    alpha = numpy.std(x) / numpy.std(y)

    return x - alpha * y


def compute_heart_beat(signal: numpy.ndarray) -> float:
    """
    Compute heart rate from BGR signal sequence.

    :param signal:
        Signal array with shape [N, 4].
        Columns are:
            [timestamp, B, G, R]

        Timestamp unit is seconds.

    :return:
        Heart rate in BPM.
        Returns -1 when a valid frequency cannot be found.
    """
    signal = numpy.asarray(signal, dtype=numpy.float64)

    if signal.ndim != 2 or signal.shape[1] != 4:
        raise ValueError(f'Invalid signal shape: {signal.shape}')

    sample_count = signal.shape[0]

    if sample_count < 2:
        return -1.0

    timestamps = signal[:, 0]
    duration = timestamps[-1] - timestamps[0]

    if duration <= 0:
        return -1.0

    # Generate evenly spaced timestamps.
    step = duration / sample_count
    interp_times = _generate_interp_times(timestamps[0], timestamps[-1], step)

    interp_signal = numpy.stack(
        [_cubic_spline_interp_1d(signal[:, channel], timestamps, interp_times) for channel in range(1, 4)], axis=0
    )

    # Smooth color channel signals.
    for i in range(3):
        interp_signal[i] = _gaussian_blur_1d(interp_signal[i])

    pulse_signal = _chrominance_method(interp_signal)

    # Smooth pulse signal.
    pulse_signal = _gaussian_blur_1d(pulse_signal)

    # Normalize signal amplitude.
    smean = numpy.mean(pulse_signal)
    smax = numpy.max(pulse_signal)
    smin = numpy.min(pulse_signal)

    # Keep the original normalization behavior.
    # The dynamic scale calculation is reserved for future evaluation.
    scale = 150.0

    pulse_signal = (pulse_signal - smean) * scale
    length = len(pulse_signal)

    if length < 2:
        return -1.0

    # Apply Hamming window.
    pulse_signal *= numpy.hamming(length)
    pulse_signal -= numpy.mean(pulse_signal)

    fft_result = numpy.fft.rfft(pulse_signal)
    magnitude = numpy.abs(fft_result)
    fps = length / duration
    frequencies = numpy.fft.rfftfreq(length, d=1.0 / fps)

    valid = (frequencies > 50.0 / 60.0) & (frequencies < 120.0 / 60.0)
    indices = numpy.where(valid)[0]

    if len(indices) == 0:
        return -1.0

    peak = indices[numpy.argmax(magnitude[indices])]
    return float(frequencies[peak] * 60.0)


def main():
    pass


if __name__ == '__main__':
    main()
