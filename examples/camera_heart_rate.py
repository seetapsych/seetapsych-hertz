# -*- coding: utf-8 -*-

import os
import platform

import cv2
import numpy
from seetapsych_lib.runtime.factory import Factory

# from seetapsych_lib.runtime.runner import Runner
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../../seetapsych-hertz/seetapsych_hertz/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-face-hub/seetapsych_face_hub/modules"),
]


def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
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


def main():
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
    cap = open_camera(0)

    hr: float | None = None

    while cap.isOpened():
        ok = cap.grab()
        if not ok:
            break
        ok, frame = cap.retrieve()
        if not ok:
            break
        report = runner.run(data={"default": frame})
        _frame_height, _frame_width = frame.shape[:2]
        face_detection = report.get("face_detection", [])
        face_selection = report.get("face_selection", {"pid": 0})
        for bbox in face_detection:
            xyxy = bbox["xyxy"]
            # score = bbox['score']
            xyxy = list(map(int, xyxy))
            cv2.rectangle(frame, xyxy[:2], xyxy[2:], (255, 0, 0), 2)

        # print(face_detection)
        heart_rate = report.get("face_heart_rate", {})

        frame: numpy.ndarray = cv2.flip(frame, 1)

        if face_detection:
            xyxy = face_detection[0]["xyxy"]
            xyxy = list(map(int, xyxy))
            p = [frame.shape[1] - xyxy[2], xyxy[1]]
            pid = face_selection["pid"]

            cv2.putText(frame, f"{pid}", p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            wait_seconds = heart_rate.get("wait_seconds", 0)
            hr = heart_rate.get("hr_bpm", None) or hr
            cv2.putText(frame, f"wait: {wait_seconds:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if hr is not None:
                cv2.putText(frame, f"hr: {hr:.0f}bpm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, "Press Esc/Q/X to exit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Default WINDOW_AUTOSIZE: window size strictly matches frame size, not user-resizable
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

        key &= 0xFF
        if key in {ord("q"), ord("x")}:
            break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
        cv2.destroyWindow(window_name)

    cap.release()
    runner.dispose()


if __name__ == "__main__":
    main()
