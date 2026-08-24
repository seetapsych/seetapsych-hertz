# -*- coding: utf-8 -*-
import time
from typing import Any

import numpy

from fabopsy_lib import api

from .estimator import Estimator


class Instance(api.Instance):
    def __init__(
            self, device: api.Device,
            min_seconds: float | None = None,
            min_frames: int | None = None,
            max_frames: int | None = None,
    ):
        self.__device = device
        self.__estimator = Estimator(
            min_seconds=min_seconds,
            min_frames=min_frames,
            max_frames=max_frames,
        )

    def inference(self, *,
                  data: dict[str, Any],
                  report: dict[str, Any],
                  **kwargs) -> dict[str, Any]:
        face_dense_landmarks = report.get('face_dense_landmarks', [])
        no_face = not len(face_dense_landmarks)

        if no_face:
            self.__estimator.reset()
            return {
                'face_heart_rate': {
                    'fps': self.__estimator.get_fps(),
                    'wait_seconds': self.__estimator.get_wait_time(),
                }
            }

        image = data['default']
        image = numpy.ascontiguousarray(image)  # [H, W, C] format, BGR layout
        landmarks = face_dense_landmarks[0].get('landmarks', [])
        landmarks = numpy.asarray(landmarks).reshape([-1, 2])
        timestamp = report.get('timestamp', time.time())

        return {
            'face_heart_rate': self.__estimator.inference(image, landmarks, timestamp)
        }

    def reset(self):
        self.__estimator.reset()


class Package(api.Package):
    def create(self, *,
               models: list[api.UsageModel],
               parameters: dict[str, Any],
               device: api.Device | None,
               **kwargs) -> Instance:
        min_seconds = parameters.get('min_seconds', None)
        min_frames = parameters.get('min_frames', None)
        max_frames = parameters.get('max_frames', None)

        return Instance(
            api.Device('cpu') if device is None else device,
            min_seconds=min_seconds,
            min_frames=min_frames,
            max_frames=max_frames,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == '__main__':
    main()
