# -*- coding: utf-8 -*-
import time
from typing import Any

import numpy

from fabopsy_lib import api

from fabopsy_hertz.v0.hr import compute_heart_beat
from fabopsy_hertz.v0.landmark_mapper_280281 import LandmarkMapper
from fabopsy_hertz.v0.roi import extract_roi
from fabopsy_hertz.v0.signal import extract_signal


class Instance(api.Instance):
    def __init__(self, device: api.Device):
        self.__device = device

        self.__signal_min_seconds: float = 1
        self.__signal_min_frames: int = 10
        self.__signal_max_frames: int = 300
        self.__signal = numpy.zeros([0, 4], dtype=numpy.float64)

    def inference(self, *,
                  data: dict[str, Any],
                  report: dict[str, Any],
                  **kwargs) -> dict[str, Any]:
        input_data = data['default']
        input_data = numpy.ascontiguousarray(input_data)  # [H, W, C] format, BGR layout

        face_dense_landmarks = report.get('face_dense_landmarks', [])

        no_face = not len(face_dense_landmarks)

        if no_face:
            self.reset()
            return {
                'face_heart_rate': {
                    'fps': self.get_fps(),
                    'wait_seconds': self.get_wait_time(),
                }
            }

        landmarks = face_dense_landmarks[0].get('landmarks', [])

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
                'face_heart_rate': {
                    'fps': self.get_fps(),
                    'wait_seconds': self.get_wait_time(),
                }
            }

        signal = extract_signal(roi_image, signal_mask)
        signal_time = report.get('timestamp', time.time())

        self.__append_signal(
            signal_time,
            signal,
        )

        wait_time = self.get_wait_time()

        if wait_time > 0:
            return {
                'face_heart_rate': {
                    'fps': self.get_fps(),
                    'wait_seconds': float(wait_time),
                }
            }

        hr = compute_heart_beat(self.__signal)

        return {
            'face_heart_rate': {
                'fps': self.get_fps(),
                'wait_seconds': 0,
                'hr': hr,
            }
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


class Package(api.Package):
    def create(self, *,
               models: list[api.UsageModel],
               parameters: dict[str, Any],
               device: api.Device | None,
               **kwargs) -> Instance:
        return Instance(
            api.Device('cpu') if device is None else device,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == '__main__':
    main()
