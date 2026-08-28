# -*- coding: utf-8 -*-
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy

from seetapsych_lib import api

from .lib.config import DEFAULT_ESTIMATOR_CONFIG, EstimatorConfig
from .lib.landmark_utils import FrameLandmarks
from .lib.framewise_estimator import FrameDetailResult, FramewiseHeartRateEstimator


class Instance(api.Instance):
    def __init__(
            self,
            device: api.Device,
            estimator_config: EstimatorConfig | None = None,
            window_samples: int | None = None,
    ):
        base_config = estimator_config or DEFAULT_ESTIMATOR_CONFIG
        roi_regions = ['skin_b_adaptive_forehead']
        
        if window_samples is None:
            window_samples = base_config.window_samples

        estimator_config = replace(
            base_config,
            roi_regions=tuple(roi_regions),
            window_samples=window_samples,
            enable_fft_fusion=False,
        )

        self.__device = device
        self.__estimator_config: EstimatorConfig = estimator_config
        self.__window_samples: int = int(estimator_config.window_samples)
        self.__estimator = FramewiseHeartRateEstimator(
            config=estimator_config,
        )

        self.__frame_count: int = 0
        self.__timestamps: list[float] = []
        self.__stream_path: Path = Path('stream')

    def inference(self, *,
                  data: dict[str, Any],
                  report: dict[str, Any],
                  **kwargs) -> dict[str, Any]:
        face_dense_landmarks = report.get('face_dense_landmarks', [])
        no_face = not len(face_dense_landmarks)

        if no_face:
            self.reset()
            return {
                'face_heart_rate': {
                    'fps': self._get_fps(),
                    'wait_seconds': self._get_wait_time(),
                }
            }

        image = data['default']
        image = numpy.ascontiguousarray(image)  # [H, W, C] format, BGR layout
        landmarks = face_dense_landmarks[0].get('landmarks', [])
        landmarks = numpy.asarray(landmarks).reshape([-1, 2])   # 280 points
        timestamp = float(report.get('timestamp', time.time()))
        frame_tick = int(report.get('frame_tick', self.__frame_count))

        # update frame count and timestamps
        frame_offset = self.__frame_count
        self.__frame_count += 1
        self.__timestamps.append(timestamp)
        if len(self.__timestamps) > self.__window_samples:
            del self.__timestamps[:len(self.__timestamps) - self.__window_samples]

        current_fps = self._get_fps()
        if current_fps <= 0:
            current_fps = 30.0

        frame_landmarks = FrameLandmarks(
            path=self.__stream_path,
            frame_offset=frame_offset,
            frame_index=frame_tick,
            frame_time_seconds=timestamp,
            video_fps=float(current_fps),
            points=landmarks,
        )

        result: FrameDetailResult = self.__estimator.process_frame(image, frame_landmarks)

        # hr_available, hr_value = self._resolve_hr(result)
        hr_value = result.hr_results.get('roi_skin_b_adaptive_forehead_hr_bpm', 0)
        hr_available = numpy.isfinite(hr_value) and hr_value > 0.0

        if hr_available:
            return {
                'face_heart_rate': {
                    'fps': current_fps,
                    'wait_seconds': 0.0,
                    'hr_bpm': float(hr_value),
                }
            }

        return {
            'face_heart_rate': {
                'fps': current_fps,
                'wait_seconds': self._get_wait_time(),
            }
        }

    def reset(self):
        self.__estimator = FramewiseHeartRateEstimator(
            config=self.__estimator_config,
        )
        self.__frame_count = 0
        self.__timestamps.clear()

    def _resolve_hr(self, result: FrameDetailResult) -> tuple[bool, float]:
        hr_value = float(result.hr_value_fusion)
        if numpy.isfinite(hr_value) and hr_value > 0.0:
            return True, hr_value

        for name in self.__estimator.roi_names:
            status_key = f"roi_{name}_status"
            hr_key = f"roi_{name}_hr_bpm"
            if str(result.hr_results.get(status_key)) == 'ok':
                try:
                    val = float(result.hr_results.get(hr_key))
                    if numpy.isfinite(val) and val > 0.0:
                        return True, val
                except (TypeError, ValueError):
                    continue

        for mode in self.__estimator.active_fusion_modes:
            from .lib.fft_fusion import fft_fusion_output_prefix
            prefix = fft_fusion_output_prefix(mode)
            status_key = f"{prefix}_status"
            hr_key = f"{prefix}_hr_bpm"
            if str(result.hr_results.get(status_key)) == 'ok':
                try:
                    val = float(result.hr_results.get(hr_key))
                    if numpy.isfinite(val) and val > 0.0:
                        return True, val
                except (TypeError, ValueError):
                    continue

        return False, 0.0

    def _get_fps(self) -> float:
        size = len(self.__timestamps)
        if size <= 1:
            return 0.0
        duration = self.__timestamps[-1] - self.__timestamps[0]
        if duration <= 0:
            return 0.0
        return float((size - 1) / duration)

    def _get_wait_time(self) -> float:
        # Rough estimate only: use received frame count vs window_samples
        received = self.__frame_count
        need = self.__window_samples
        if received >= need:
            return 0.0

        remaining = need - received

        fps = self._get_fps()
        if fps <= 0:
            fps = 30.0

        return float(remaining) / fps


class Package(api.Package):
    def create(self, *,
               models: list[api.UsageModel],
               parameters: dict[str, Any],
               device: api.Device | None,
               **kwargs) -> Instance:
        window_samples: int | None = parameters.get('window_samples', None)

        return Instance(
            api.Device('cpu') if device is None else device,
            window_samples=window_samples,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == '__main__':
    main()
