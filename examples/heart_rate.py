# -*- coding: utf-8 -*-
import os

import cv2
import numpy
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

module_roots = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_hertz/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-face-hub/seetapsych_face_hub/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-face-ex/seetapsych_face_ex/modules"),
]


def main():
    factory = Factory(disable_default=True)
    for root in module_roots:
        factory.load_dir_modules(root)
    factory.load_default_modules()

    pipeline = Pipeline(
        factory,
        packages=[
            # '23871da8-0968-4034-ac48-35641ae67d63', # TinyHR
            "3d98f435-d484-4b91-acf4-f690c28b409f",  # AdaChrom
        ],
    )

    pipeline.solve()

    package = pipeline.get_package(provide="face/heart_rate")
    assert package is not None
    pipeline.set_parameters(package.uid, {"roi_regions": ["skin_b_adaptive_forehead"]})

    pipeline.install_requirements()
    pipeline.cache_models()

    print(pipeline.problem())
    print(pipeline.satisfied())

    runner = Runner(pipeline)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open camera")
        exit(1)

    hr: float | None = None

    while True:
        ok = cap.grab()
        if not ok:
            break
        ok, frame = cap.retrieve()
        if not ok:
            break
        report = runner.run(data={"default": frame})
        frame_height, frame_width = frame.shape[:2]
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
            cv2.putText(
                frame,
                f"wait: {wait_seconds:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            if hr is not None:
                cv2.putText(frame, f"hr: {hr:.0f}bpm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("face", frame)
        key = cv2.waitKey(1)
        if key >= 0:
            break

    runner.dispose()


if __name__ == "__main__":
    main()
