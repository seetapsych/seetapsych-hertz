# -*- coding: utf-8 -*-

import cv2
import numpy


def initialize_roi(
        points81: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, tuple[int, int], tuple[int, int]]:
    """
    Initialize facial ROI polygons and bounding box from 81 landmarks.

    :param points81: Facial landmarks with shape [81, 2].
    :return:
        cheek_points: Polygon points representing the included facial area.
        mouth_points: Polygon points representing the excluded mouth area.
        left: Top-left point of the ROI bounding box.
        right: Bottom-right point of the ROI bounding box.
    """
    points81 = numpy.asarray(points81)

    if points81.shape != (81, 2):
        raise ValueError(
            f"points81 must have shape [81, 2], got {points81.shape}"
        )

    # Match the implicit float-to-int conversion.
    points = points81.astype(numpy.int32)

    left_eye_x, left_eye_y = points[0]
    right_eye_x, right_eye_y = points[9]

    length = int(
        numpy.sqrt(
            (right_eye_y - left_eye_y) ** 2
            + (right_eye_x - left_eye_x) ** 2
        )
    )

    delta1 = int(length / 4.5)
    delta2 = length // 50

    cheek_points: list[tuple[int, int]] = []

    cheek_points.append(
        (
            int(points[0, 0]),
            int(points[0, 1] + delta1),
        )
    )

    for i in range(65, 73):
        cheek_points.append(
            (
                int(points[i, 0] + delta2),
                int(points[i, 1]),
            )
        )

    for i in range(80, 72, -1):
        cheek_points.append(
            (
                int(points[i, 0] - delta2),
                int(points[i, 1]),
            )
        )

    cheek_points.append(
        (
            int(points[9, 0]),
            int(points[9, 1] + delta1),
        )
    )

    cheek_points = numpy.asarray(cheek_points, dtype=numpy.int32)

    left_x = int(numpy.min(cheek_points[:, 0]))
    left_y = int(numpy.min(cheek_points[:, 1]))
    right_x = int(numpy.max(cheek_points[:, 0]))
    right_y = int(numpy.max(cheek_points[:, 1]))

    delta = length // 50

    left_x -= delta
    left_y -= delta
    right_x += delta
    right_y += delta

    mouth_indices = [46, 50, 48, 51, 47, 59, 55, 58]

    mouth_points = numpy.asarray(
        [
            (
                int(points[index, 0]),
                int(points[index, 1]),
            )
            for index in mouth_indices
        ],
        dtype=numpy.int32,
    )

    return (
        cheek_points,
        mouth_points,
        (left_x, left_y),
        (right_x, right_y),
    )


def landmarks_visible(
        points81: numpy.ndarray,
        width: int,
        height: int,
        threshold: float = 0.3,
) -> bool:
    """
    Check whether enough landmarks are inside image.
    """

    inside = (
            (points81[:, 0] >= 0)
            &
            (points81[:, 0] < width)
            &
            (points81[:, 1] >= 0)
            &
            (points81[:, 1] < height)
    )

    return numpy.mean(inside) >= threshold


def extract_roi(
        image: numpy.ndarray,
        points81: numpy.ndarray,
) -> tuple[numpy.ndarray | None, numpy.ndarray | None]:
    """
    Extract the facial ROI and signal mask used for heart-rate estimation.

    :param image: Image in HWC format with BGR layout.
    :param points81: Facial landmarks with shape [81, 2].
    :return: Two images representing the ROI image area and signal mask.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must have shape [H, W, 3], got {image.shape}"
        )

    if not landmarks_visible(
            points81,
            image.shape[1],
            image.shape[0],
    ):
        return None, None

    cheek_points, mouth_points, left, right = initialize_roi(points81)

    left_x, left_y = left
    right_x, right_y = right

    height, width = image.shape[:2]

    # Check whether ROI has any intersection with image.
    if (
            right_x <= 0
            or right_y <= 0
            or left_x >= width
            or left_y >= height
    ):
        return None, None

    # Clamp ROI into image boundary.
    left_x = max(0, left_x)
    left_y = max(0, left_y)

    right_x = min(width, right_x)
    right_y = min(height, right_y)

    roi_width = right_x - left_x
    roi_height = right_y - left_y

    if roi_width <= 0 or roi_height <= 0:
        raise ValueError(
            f"Invalid clipped ROI size: {roi_width}x{roi_height}"
        )

    if roi_width <= 0 or roi_height <= 0:
        raise ValueError(
            f"Invalid ROI size: {roi_width}x{roi_height}"
        )

    roi_image = image[
        left_y:right_y,
        left_x:right_x,
    ]

    # Convert polygons from image coordinates to ROI-local coordinates.
    roi_cheek_points = cheek_points.copy()
    roi_cheek_points[:, 0] -= left_x
    roi_cheek_points[:, 1] -= left_y

    roi_mouth_points = mouth_points.copy()
    roi_mouth_points[:, 0] -= left_x
    roi_mouth_points[:, 1] -= left_y

    # Build the facial area mask.
    face_mask = numpy.zeros(
        (roi_height, roi_width),
        dtype=numpy.uint8,
    )

    cv2.fillConvexPoly(
        face_mask,
        roi_cheek_points,
        255,
    )

    # Exclude the mouth area from the signal region.
    cv2.fillConvexPoly(
        face_mask,
        roi_mouth_points,
        0,
    )

    skin_mask = skin_detect(roi_image)

    signal_mask = cv2.bitwise_and(
        face_mask,
        skin_mask,
    )

    # # Debug visualization.
    # debug_roi = roi_image.copy()
    #
    # # Green pixels represent the final signal extraction area.
    # debug_roi[signal_mask > 0] = (
    #         debug_roi[signal_mask > 0] * 0.5
    #         + numpy.array([0, 255, 0]) * 0.5
    # ).astype(numpy.uint8)
    #
    # cv2.imshow("Debug(ROI)", roi_image)
    # cv2.imshow("Debug(face)", face_mask)
    # cv2.imshow("Debug(skin)", skin_mask)
    # cv2.imshow("Debug(combine)", signal_mask)
    # cv2.imshow("Debug(signal ROI)", debug_roi)
    #
    # # Required for OpenCV to actually refresh the imshow windows.
    # cv2.waitKey(1)

    return roi_image, signal_mask


def skin_detect(face: numpy.ndarray) -> numpy.ndarray:
    """
    Detect skin pixels using the YCrCb algorithm.

    :param face: BGR image.
    :return: uint8 mask where 255 represents detected skin.
    """
    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)

    y = ycrcb[:, :, 0].astype(numpy.int32)
    cr = ycrcb[:, :, 1].astype(numpy.int32)
    cb = ycrcb[:, :, 2].astype(numpy.int32)

    cb -= 109
    cr -= 152

    # integer division truncates toward zero.
    x1 = numpy.trunc(
        (819 * cr - 614 * cb) / 32.0
    ).astype(numpy.int32) + 51

    y1 = numpy.trunc(
        (819 * cr + 614 * cb) / 32.0
    ).astype(numpy.int32) + 77

    x1 = numpy.trunc(
        x1 * 41 / 1024.0
    ).astype(numpy.int32)

    y1 = numpy.trunc(
        y1 * 73 / 1024.0
    ).astype(numpy.int32)

    value = x1 * x1 + y1 * y1

    skin = numpy.where(
        ((y < 100) & (value < 700))
        | ((y >= 100) & (value < 850)),
        255,
        0,
        )

    return skin.astype(numpy.uint8)


def main():
    pass


if __name__ == '__main__':
    main()
