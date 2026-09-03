# -*- coding: utf-8 -*-
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy
from seetapsych_lib import api

from .lib.config import DEFAULT_ESTIMATOR_CONFIG, EstimatorConfig
from .lib.framewise_estimator import FrameDetailResult, FramewiseHeartRateEstimator
from .lib.landmark_utils import FrameLandmarks


def _is_valid_bpm(value: float | None) -> bool:
    return value is not None and numpy.isfinite(value) and value > 0.0


class Instance(api.Instance):
    def __init__(
        self,
        device: api.Device,
        estimator_config: EstimatorConfig | None = None,
        window_samples: int | None = None,
        roi_regions: list[str] | None = None,
    ):
        base_config = estimator_config or DEFAULT_ESTIMATOR_CONFIG

        if window_samples is None:
            window_samples = base_config.window_samples
        if not roi_regions:
            roi_regions = ["skin_b_adaptive_forehead"]

        estimator_config = replace(
            base_config,
            roi_regions=tuple(roi_regions),
            window_samples=window_samples,
            hr_update_stride=1,
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
        self.__stream_path: Path = Path("stream")

    def inference(self, *, data: dict[str, Any], report: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        face_dense_landmarks = report.get("face_dense_landmarks", [])
        no_face = not len(face_dense_landmarks)

        if no_face:
            self.reset()
            return {
                "face_heart_rate": {
                    "fps": self._get_fps(),
                    "wait_seconds": self._get_wait_time(),
                }
            }

        image = data["default"]
        image = numpy.ascontiguousarray(image)  # [H, W, C] format, BGR layout
        landmarks = face_dense_landmarks[0].get("landmarks", [])
        landmarks = numpy.asarray(landmarks).reshape([-1, 2])  # 280 points
        timestamp = float(report.get("timestamp", time.time()))
        frame_tick = int(report.get("frame_tick", self.__frame_count))

        # update frame count and timestamps
        frame_offset = self.__frame_count
        self.__frame_count += 1
        self.__timestamps.append(timestamp)
        if len(self.__timestamps) > self.__window_samples:
            del self.__timestamps[: len(self.__timestamps) - self.__window_samples]

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

        hr_value, roi_hr_values = self._resolve_hr(result)
        hr_available = _is_valid_bpm(hr_value)

        if hr_available:
            return {
                "face_heart_rate": {
                    "fps": current_fps,
                    "wait_seconds": 0.0,
                    "hr_bpm": hr_value,
                    "roi_hr_bpm": roi_hr_values,
                }
            }

        return {
            "face_heart_rate": {
                "fps": current_fps,
                "wait_seconds": self._get_wait_time(),
            }
        }

    def reset(self):
        self.__estimator = FramewiseHeartRateEstimator(
            config=self.__estimator_config,
        )
        self.__frame_count = 0
        self.__timestamps.clear()

    def _resolve_hr(self, result: FrameDetailResult) -> tuple[float | None, dict[str, float]]:
        hr_value: float | None = None
        roi_hr_values: dict[str, float] = {}

        # summary roi
        for name in self.__estimator.roi_names:
            status_key = f"roi_{name}_status"
            hr_key = f"roi_{name}_hr_bpm"

            hr_status = cast(str, result.hr_results.get(status_key, ""))
            if hr_status != "ok":
                continue

            roi_hr_value = cast(float | None, result.hr_results.get(hr_key))
            if _is_valid_bpm(roi_hr_value):
                roi_hr_values[name] = cast(float, roi_hr_value)

        # get fusion hr value or mean of roi hr values
        try:
            fusion_hr_value = float(result.hr_value_fusion)
            if _is_valid_bpm(fusion_hr_value):
                hr_value = fusion_hr_value
        except (TypeError, ValueError):
            pass

        if hr_value is None:
            if not roi_hr_values:
                return None, {}

            hr_value = float(numpy.mean(list(roi_hr_values.values())))

        return hr_value, roi_hr_values

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
    def create(
        self,
        *,
        models: list[api.UsageModel],
        parameters: dict[str, Any],
        device: api.Device | None,
        **kwargs: Any,
    ) -> Instance:
        window_samples: int | None = parameters.get("window_samples", None)
        roi_regions: list[str] | None = cast(list[str] | None, parameters.get("roi_regions", None))

        return Instance(
            api.Device("cpu") if device is None else device,
            window_samples=window_samples,
            roi_regions=roi_regions,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == "__main__":
    main()
