# -*- coding: utf-8 -*-
import time
from typing import Any, Optional

import numpy
import onnxruntime

from seetapsych_lib import api

from .heartrate_onnx import CameraHRTracker, FPS_DEFAULT


def onnx_providers(device: api.Device):
    available_providers = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        device: Optional[api.Device]
        if device is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif not device.type == 'cuda':
            providers = ['CPUExecutionProvider']
        else:
            device_id = device.index
            if device_id is None:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    return providers


class Instance(api.Instance):
    def __init__(
            self, device: api.Device,
            model_path: str,
            fps: Optional[float] = None,
            interval: Optional[float] = None,
    ):
        if fps is None:
            fps = FPS_DEFAULT
        if interval is None:
            interval = 1

        self.__device = device
        self.__estimator = CameraHRTracker(
            fps=fps,
            model_path=model_path,
            providers=onnx_providers(device),
        )
        self.__interval = interval

        self.__buffer_frames = None
        self.__buffer_faces = None
        self.__timestamp = None

        self.__last_push_timestamp = None

    def inference(self, *,
                  data: dict[str, Any],
                  report: dict[str, Any],
                  **kwargs) -> dict[str, Any]:
        face_detection = report.get('face_detection', [])
        no_face = not len(face_detection)

        if no_face:
            self.reset()
            return {
                'face_heart_rate': self.try_update()
            }

        image = data['default']
        image = numpy.ascontiguousarray(image)  # [H, W, C] format, BGR layout
        xyxy = numpy.ascontiguousarray(face_detection[0].get('xyxy', []))   # [x1, y1, x2, y2]
        timestamp = report.get('timestamp', time.time())

        self.push_frame(image, xyxy, timestamp)

        return {
            'face_heart_rate': self.try_update()
        }

    def try_update(self) -> dict:
        if self.__timestamp is None:
            return {
                'fps': self.__estimator.fps,
                'wait_seconds': self.__interval,
            }

        if self.__last_push_timestamp is None:
            self.__last_push_timestamp = self.__timestamp

        spent = self.__timestamp - self.__last_push_timestamp

        if spent < self.__interval:
            return {
                'fps': self.__estimator.fps,
                'wait_seconds': self.__interval - spent,
            }

        hr = self.__estimator.push(self.__buffer_frames, self.__buffer_faces)

        self.__last_push_timestamp = self.__timestamp
        self.clear_buffer()

        return {
            'fps': self.__estimator.fps,
            'wait_seconds': 0,
            'hr': hr,
        }

    def push_frame(self, frame: numpy.ndarray, face: numpy.ndarray, timestamp: float):
        frames = numpy.expand_dims(frame, 0)
        faces = numpy.expand_dims(face, 0)

        if self.__timestamp is None or \
                self.__buffer_frames.shape[1:] != frames.shape[1:] or \
                self.__buffer_faces.shape[1:] != faces.shape[1:]:
            self.__timestamp = timestamp
            self.__buffer_frames = frames
            self.__buffer_faces = faces
            return

        self.__timestamp = timestamp
        self.__buffer_frames = numpy.concatenate([self.__buffer_frames, frames], axis=0)
        self.__buffer_faces = numpy.concatenate([self.__buffer_faces, faces], axis=0)

    def clear_buffer(self):
        self.__buffer_frames = None
        self.__buffer_faces = None
        self.__timestamp = None

    def reset(self):
        self.__estimator.reset()

        self.__last_push_timestamp = None
        self.clear_buffer()

    def dispose(self):
        self.__estimator.cleanup()


class Package(api.Package):
    def create(self, *,
               models: list[api.UsageModel],
               parameters: dict[str, Any],
               device: api.Device | None,
               **kwargs) -> Instance:
        assert len(models) >= 1, api.MissingModelError('At least one model required')

        fps = parameters.get('fps', None)
        interval = parameters.get('interval', None)

        model_path = models[0].cache()
        return Instance(
            api.Device('cpu') if device is None else device,
            model_path,
            fps=fps,
            interval=interval,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == '__main__':
    main()
