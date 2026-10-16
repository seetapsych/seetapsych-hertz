# -*- coding: utf-8 -*-

import os
import platform
import time
from typing import Any

import cv2
import numpy
from camera_window import CameraWindow, draw_text, fit_geometry
from seetapsych_lib.runtime.factory import Factory

# from seetapsych_lib.runtime.runner import Runner
from seetapsych_lib.runtime.parallel.future import Future
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_hertz/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-face-hub/seetapsych_face_hub/modules"),
]


def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
    """Open a camera with a platform backend and default-backend fallback.

    Args:
        camera_id: Camera device index.

    Returns:
        Capture whose open state must be checked by the caller.
    """
    system = platform.system()

    backend = {
        "Windows": cv2.CAP_DSHOW,
        "Linux": cv2.CAP_V4L2,
        "Darwin": cv2.CAP_AVFOUNDATION,
    }.get(system, cv2.CAP_ANY)

    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)

    return cap


def draw_results(
    frame: numpy.ndarray,
    report: dict[str, Any],
    inference_fps: float,
    display_size: tuple[int, int] | None = None,
) -> numpy.ndarray:
    """Render mirrored detections and the latest heart-rate measurement.

    Args:
        frame: Current BGR camera frame.
        report: Latest report; empty when no recent result is available.
        inference_fps: Actual sampling throughput.
        display_size: Optional viewport bounds; inference coordinates stay in source pixels.

    Returns:
        Annotated live preview, without retaining BPM from a previous subject.
    """
    source_height, source_width = frame.shape[:2]
    if display_size is not None:
        _, _, rw, rh = fit_geometry((source_width, source_height), display_size)
        interpolation = cv2.INTER_AREA if rw < source_width else cv2.INTER_CUBIC
        frame = cv2.resize(frame, (rw, rh), interpolation=interpolation)
    canvas = cv2.flip(frame, 1)
    height, width = canvas.shape[:2]
    scale_x, scale_y = width / source_width, height / source_height
    ui_scale = min(width / 640, height / 480)
    detections = report.get("face_detection", [])
    for bbox in detections:
        x1, y1, x2, y2 = (
            int(round(v * scale)) for v, scale in zip(bbox["xyxy"], (scale_x, scale_y, scale_x, scale_y), strict=True)
        )
        cv2.rectangle(
            canvas, (width - 1 - x2, y1), (width - 1 - x1, y2), (0, 255, 0), max(1, round(2 * ui_scale)), cv2.LINE_AA
        )
    heart_rate = report.get("face_heart_rate", {})
    bpm = heart_rate.get("hr_bpm")
    if not detections:
        status = "Waiting for face"
    elif bpm is not None and numpy.isfinite(bpm) and bpm > 0:
        status = f"Heart rate: {bpm:.0f} BPM"
    else:
        status = f"Collecting signal: {heart_rate.get('wait_seconds', 0):.1f}s"
    panel_height = min(round(145 * ui_scale), height)
    panel = canvas[:panel_height]
    canvas[:panel_height] = cv2.addWeighted(panel, 0.35, numpy.zeros_like(panel), 0.65, 0)
    lines = [
        status,
        f"Sampling: {inference_fps:.1f} FPS | Keep still in steady light",
        "F = Fullscreen | R = Restart | Esc / Q / X = Exit",
        "Demo estimate, not a medical measurement",
    ]
    for index, text in enumerate(lines):
        draw_text(
            canvas,
            text,
            (round(12 * ui_scale), round((28 + index * 32) * ui_scale)),
            (0, 220, 255) if index == 0 else (255, 255, 255),
            scale=0.58 * ui_scale,
        )
    return canvas


def run_camera(cap: cv2.VideoCapture) -> None:
    """Sample asynchronously with capture timestamps and no queued backlog.

    Args:
        cap: Open camera owned by the caller.
    """
    factory = Factory()
    # Attempt to override default modules with local module files, can still run even if no local files exist
    for root in override_modules:
        factory.load_dir_modules(root)

    pipeline = Pipeline(
        factory,
        packages=[
            # '08e29e70-e21b-4ef2-8486-5aaec5f8295b', # SeetaHeartRateDetector
            # '23871da8-0968-4034-ac48-35641ae67d63', # TinyHR
            "3d98f435-d484-4b91-acf4-f690c28b409f",  # AdaChrom
        ],
        attributes=[
            # 'face/heart_rate',
        ],
    )

    # print(pipeline.problem())
    pipeline.solve()

    # print(pipeline.satisfied())
    pipeline.install_requirements()
    pipeline.cache_models()

    runner = Runner(pipeline)

    window_name = "HeartRate"
    pending: Future[dict[str, Any]] | None = None
    report: dict[str, Any] = {}
    submitted_at = result_at = 0.0
    inference_fps = 0.0
    reset_requested = False
    try:
        window = CameraWindow(window_name, (960, 720))
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames. Check the connection and camera permissions.")
                break
            captured_at = time.time()
            now = time.monotonic()
            if pending is not None and pending.done():
                output = pending.get()
                pending = None
                if not reset_requested:
                    report = output
                    result_at = submitted_at
                    inference_fps = 1.0 / max(now - submitted_at, 1e-6)
            if pending is None:
                if reset_requested:
                    runner.reset()
                    report = {}
                    reset_requested = False
                submitted_at = now
                # Preserve real sampling intervals for timestamp-based signal resampling.
                pending = runner.run_async(data={"default": frame.copy()}, timestamp=captured_at)
            result_ttl = max(1.0, 2.0 / max(inference_fps, 0.01))
            canvas = draw_results(frame, report if now - result_at < result_ttl else {}, inference_fps, window.size())
            window.show(canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q"), ord("x"), ord("X")):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            window.handle_key(key)
            if key in (ord("r"), ord("R")):
                reset_requested = True
                report = {}
    finally:
        if pending is not None:
            pending.cancel()
        runner.dispose()


def main() -> None:
    """Open the camera before preparing models and release resources on exit."""
    cap = open_camera(0)
    try:
        if not cap.isOpened():
            print("Could not open camera 0. Check the device index, camera permissions, and other camera apps.")
            return
        run_camera(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
