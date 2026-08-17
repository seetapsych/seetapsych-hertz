# -*- coding: utf-8 -*-
import json
import os

import cv2
import numpy

from fabopsy_lib.runtime.factory import Factory
from fabopsy_lib.runtime.pipeline import Pipeline
from fabopsy_lib.runtime.parallel_runner import ParallelRunner as Runner


def main():
    factory = Factory()
    factory.load_builtin_modules()

    factory.load_dir_modules(os.path.join(os.path.dirname(__file__), '../fabopsy_hertz/modules'))
    factory.load_dir_modules(os.path.join(os.path.dirname(__file__), '../../fabopsy-face/fabopsy_face/modules'))
    factory.load_dir_modules(os.path.join(os.path.dirname(__file__), '../../fabopsy-affect/fabopsy_affect/modules'))
    factory.load_dir_modules(os.path.join(os.path.dirname(__file__), '../../fabopsy-face-open/fabopsy_face_open/modules'))

    pipeline = Pipeline(factory, packages=[
        'c938b879-44db-45b0-9a5d-8377f0ace5e5', # insightface's face/detection
        'bb212f54-aace-438f-9cb7-f6519f4fef48', # face/selection
    ], attributes=[
        'face/dense_landmarks',
        'face/heart_rate'
    ])
    pipeline.set_models(
        'c938b879-44db-45b0-9a5d-8377f0ace5e5',
        ['1deb39b1-1074-4f6e-b81e-a6a843d011eb']    # using minimum retinaface model
    )

    # print(pipeline.config.model_dump_json(indent=2, exclude_none=True))

    pipeline.solve()

    # print(pipeline.config.model_dump_json(indent=2, exclude_none=True))

    print(pipeline.problem())
    print(pipeline.satisfied())

    runner = Runner(pipeline)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('Could not open camera')
        exit(1)

    while True:
        ok = cap.grab()
        if not ok:
            break
        ok, frame = cap.retrieve()
        if not ok:
            break
        report = runner.run(data={
            'default': frame
        })
        frame_height, frame_width = frame.shape[:2]
        face_detection = report.get('face_detection', [])
        face_selection = report.get('face_selection', {'pid': 0})
        face_landmarks = report.get('face_dense_landmarks', [])
        for bbox, landmarks in zip(face_detection, face_landmarks):
            xyxy = bbox['xyxy']
            # score = bbox['score']
            xyxy = list(map(int, xyxy))

            point2ds = numpy.asarray(landmarks['landmarks']).reshape([-1, 2])

            cv2.rectangle(frame, xyxy[:2], xyxy[2:], (255, 0, 0), 2)
            for p in point2ds:
                p = list(map(int, p))
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

        # print(face_detection)
        heart_rate = report.get('face_heart_rate', {})

        frame: numpy.ndarray = cv2.flip(frame, 1)

        if face_detection:
            xyxy = face_detection[0]['xyxy']
            xyxy = list(map(int, xyxy))
            p = [frame.shape[1] - xyxy[2], xyxy[1]]
            pid = face_selection['pid']

            cv2.putText(frame, f'{pid}', p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            wait_seconds = heart_rate.get('wait_seconds', 0)
            hr = heart_rate.get('hr', None)
            cv2.putText(frame, f'wait: {wait_seconds:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if hr is not None:
                cv2.putText(frame, f'hr: {hr:.0f}bpm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('face', frame)
        key = cv2.waitKey(1)
        if key >= 0:
            break

    runner.dispose()


if __name__ == '__main__':
    main()
