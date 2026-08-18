# -*- coding: utf-8 -*-

from typing import TypedDict, Optional


import numpy


from .hr import compute_heart_beat
from .landmark_mapper_280281 import LandmarkMapper
from .roi import extract_roi
from .signal import extract_signal


class HeartRateResult(TypedDict, total=False):
    fps: float
    wait_seconds: float
    hr: float


class Estimator(object):
    def __init__(
            self,
            min_seconds: float | None = None,
            min_frames: int | None = None,
            max_frames: int | None = None,
    ):
        if min_seconds is None:
            min_seconds = 1
        if min_frames is None:
            min_frames = 10
        if max_frames is None:
            max_frames = 300

        self.__signal_min_seconds: float = min_seconds
        self.__signal_min_frames: int = min_frames
        self.__signal_max_frames: int = max_frames
        self.__signal = numpy.zeros([0, 4], dtype=numpy.float64)

    def inference(self, image: numpy.ndarray, landmarks: numpy.ndarray, timestamp: float) -> HeartRateResult:
        """
        :param image: Image in HWC format with BGR layout.
        :param landmarks: Facial landmarks with shape [x, 2]. x could be 280/81
        :param timestamp: timestamp in seconds
        :return:
        """
        input_data = image
        point2ds = numpy.asarray(landmarks).reshape([-1, 2])

        num_points = point2ds.shape[0]
        if num_points == 81:
            pass
        elif num_points == 280:
            mapper = LandmarkMapper()
            point2ds = mapper.map_280_to_81(point2ds)
        else:
            raise RuntimeError('Number landmarks must be 81/280')

        roi_image, signal_mask = extract_roi(input_data, point2ds)
        if roi_image is None or signal_mask is None:
            # treat like no face
            self.reset()
            return {
                'fps': self.get_fps(),
                'wait_seconds': self.get_wait_time(),
            }

        signal = extract_signal(roi_image, signal_mask)
        signal_time = timestamp

        self.__append_signal(
            signal_time,
            signal,
        )

        wait_time = self.get_wait_time()

        if wait_time > 0:
            return {
                'fps': self.get_fps(),
                'wait_seconds': float(wait_time),
            }

        hr = compute_heart_beat(self.__signal)

        return {
            'fps': self.get_fps(),
            'wait_seconds': 0,
            'hr': hr,
        }

    def reset(self):
        self.__signal = numpy.zeros([0, 4], dtype=numpy.float64)

    def __append_signal(
            self,
            signal_time: float,
            signal: numpy.ndarray,
    ) -> None:
        """
        Append one signal sample and maintain the maximum signal window size.

        Each signal row has the format:
            [time, B, G, R]

        The time value is measured in seconds.
        """
        signal = numpy.asarray(signal, dtype=numpy.float64).reshape(-1)

        if signal.shape != (3,):
            raise ValueError(
                f'Signal must have shape [3], got {signal.shape}'
            )

        sample = numpy.empty([1, 4], dtype=numpy.float64)
        sample[0, 0] = signal_time
        sample[0, 1:] = signal

        self.__signal = numpy.concatenate(
            [self.__signal, sample],
            axis=0,
        )

        if len(self.__signal) > self.__signal_max_frames:
            self.__signal = self.__signal[-self.__signal_max_frames:]


    def get_wait_time(self) -> float:
        """
        Get the estimated waiting time before heart-rate calculation is available.

        Both minimum requirements must be satisfied:
          1. The signal duration reaches signal_min_seconds.
          2. The number of signal samples reaches signal_min_window.

        :return: Estimated waiting time in seconds, or 0 if enough data is available.
        """
        size = len(self.__signal)

        if size == 0:
            return float(self.__signal_min_seconds)

        time_span = self.__signal[-1, 0] - self.__signal[0, 0]

        time_wait = max(
            0.0,
            self.__signal_min_seconds - time_span,
            )

        if size >= self.__signal_min_frames:
            window_wait = 0.0
        elif size >= 2 and time_span > 0:
            fps = (size - 1) / time_span

            remaining_samples = self.__signal_min_frames - size

            window_wait = remaining_samples / fps
        else:
            # FPS cannot be estimated reliably with fewer than two samples.
            window_wait = time_wait

        return max(
            time_wait,
            window_wait,
        )

    def get_fps(self) -> float:
        """
        Calculate FPS.

        :return:
            Estimated FPS. Returns 0 when FPS cannot be calculated.
        """
        signal = self.__signal
        length = len(signal)

        if length <= 1:
            return 0.0

        duration = signal[-1, 0] - signal[0, 0]

        if duration <= 0:
            return 0.0

        return float((length - 1) / duration)


def main():
    pass


if __name__ == '__main__':
    main()
