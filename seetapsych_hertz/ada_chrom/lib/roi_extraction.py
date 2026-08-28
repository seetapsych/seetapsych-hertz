"""Shared per-frame ROI extraction helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .landmark_geometry import face_area_from_bbox, face_bbox_area_from_landmarks
from .landmark_utils import (
    FrameLandmarks,
    frame_landmark_valid_masks,
    spipnet280_to_sdk81_points,
)
from .roi_definitions import (
    POLYGON_ROI_NAMES,
    ROI_POLYGON_INDICES,
    SDK_FACEAREA_ROI_NAME,
    SKIN_ROI_NAME_BY_METHOD,
    normalize_skin_roi_methods,
    skin_roi_methods_for_roi_names,
)
from .roi_mask_utils import (
    _load_cv2,
    _mask_active,
    _points_to_array,
    _trunc_div,
    _update_bbox,
    combine_mask,
    get_bgr,
    initialize_facearea_roi,
    skin_detection,
)


def store_roi_value(
    name: str,
    mask: np.ndarray,
    mean_bgr: np.ndarray,
    pixel_count: int,
    face_area: float,
    roi_masks: dict[str, np.ndarray],
    roi_bgr: dict[str, np.ndarray],
    roi_pixels: dict[str, int],
    roi_area_ratios: dict[str, float],
) -> None:
    """Store one ROI's mask/statistics only when that ROI was requested."""

    if name not in roi_bgr:
        return
    roi_masks[name] = np.asarray(mask, dtype=bool)
    roi_bgr[name] = np.asarray(mean_bgr, dtype=np.float32).astype(float)
    roi_pixels[name] = int(pixel_count)
    if pixel_count > 0 and np.isfinite(face_area) and face_area > 0.0:
        roi_area_ratios[name] = float(np.float32(float(pixel_count) / float(face_area)))


def mean_bgr_and_pixels_from_bool_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute OpenCV-style mean BGR and active-pixel count from a full mask."""

    import cv2

    active = np.asarray(mask, dtype=bool)
    pixel_count = int(np.count_nonzero(active))
    if pixel_count <= 0:
        return np.full(3, np.nan, dtype=float), 0
    mask_u8 = np.where(active, 255, 0).astype(np.uint8)
    mean_bgr = np.asarray(cv2.mean(frame_bgr[:, :, :3], mask=mask_u8)[:3], dtype=float)
    return mean_bgr, pixel_count


def mean_bgr_and_pixels_from_crop_mask(
    frame_bgr: np.ndarray,
    crop_bounds: tuple[int, int, int, int] | None,
    crop_mask: np.ndarray | None,
) -> tuple[np.ndarray, int] | None:
    """Compute mean BGR from a cropped adaptive-ROI mask when available."""

    if crop_bounds is None or crop_mask is None:
        return None
    y0, y1, x0, x1 = crop_bounds
    frame_crop = frame_bgr[y0:y1, x0:x1]
    active = np.asarray(crop_mask, dtype=bool)
    pixel_count = int(np.count_nonzero(active))
    if pixel_count <= 0:
        return np.full(3, np.nan, dtype=float), 0
    try:
        import cv2
    except ModuleNotFoundError:
        return np.asarray(frame_crop[:, :, :3][active].mean(axis=0), dtype=float), pixel_count
    mask_u8 = np.where(active, 255, 0).astype(np.uint8)
    mean_bgr = np.asarray(cv2.mean(frame_crop[:, :, :3], mask=mask_u8)[:3], dtype=float)
    return mean_bgr, pixel_count


def make_skin_roi_builders(
    skin_roi_methods: Iterable[str],
    *,
    exclude_mouth: bool = False,
    fill_holes: bool = True,
):
    """Build adaptive skin ROI helpers for non-legacy skin methods."""

    methods = normalize_skin_roi_methods(skin_roi_methods)
    adaptive_methods = [method for method in methods if method != "legacy-ycrcb"]
    if adaptive_methods:
        from .adaptive_skin_roi import MultiFullFaceRoiBuilder

        return MultiFullFaceRoiBuilder(
            adaptive_methods,
            face_mask_mode="convex-hull",
            exclude_mouth=exclude_mouth,
            fill_holes=fill_holes,
        )
    return None


def sdk_facearea_mean_bgr_and_pixels_with_skin_mask(
    frame_bgr: np.ndarray,
    sdk_points: np.ndarray,
    frame_skin_mask: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Compute SDK face-area mean BGR with optional precomputed skin mask."""

    import cv2

    roi = initialize_facearea_roi(sdk_points)
    if roi is None:
        return np.full(3, np.nan, dtype=float), 0

    left, right, cheek_points, mouth_points = roi
    lx, ly = left
    rx, ry = right
    height, width = frame_bgr.shape[:2]
    if lx <= 0 or ly <= 0 or rx >= width or ry >= height:
        return np.full(3, np.nan, dtype=float), 0
    if rx <= lx or ry <= ly:
        return np.full(3, np.nan, dtype=float), 0

    roi_rect = frame_bgr[ly:ry, lx:rx, :3]
    if roi_rect.size == 0:
        return np.full(3, np.nan, dtype=float), 0

    local_cheek = np.asarray([(x - lx, y - ly) for x, y in cheek_points], dtype=np.int32)
    local_mouth = np.asarray([(x - lx, y - ly) for x, y in mouth_points], dtype=np.int32)
    face_mask = np.zeros(roi_rect.shape[:2], dtype=np.uint8)
    mouth_mask = np.zeros(roi_rect.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, local_cheek.reshape((-1, 1, 2)), 255)
    cv2.fillConvexPoly(mouth_mask, local_mouth.reshape((-1, 1, 2)), 255)
    face_mask = (face_mask > 10) & ~(mouth_mask > 10)

    skin_crop = (
        np.asarray(skin_detection(roi_rect), dtype=bool)
        if frame_skin_mask is None
        else np.asarray(frame_skin_mask[ly:ry, lx:rx])
    )
    combined = combine_mask(face_mask, skin_crop)
    if not np.any(combined):
        combined = np.where(face_mask, 255, 0).astype(np.uint8)
    pixel_count = int(np.count_nonzero(combined > 10))
    if pixel_count <= 0:
        return np.full(3, np.nan, dtype=float), 0

    return get_bgr(roi_rect, combined), pixel_count

def skin_roi_mean_bgr_and_pixels(
    frame_bgr: np.ndarray,
    frame_points: np.ndarray,
    sdk_points81: np.ndarray | None,
    *,
    method: str,
    builder=None,
    exclude_mouth: bool = False,
    fill_holes: bool = True,
) -> tuple[np.ndarray, int]:
    """Compute one skin ROI's mean BGR, returning NaNs when it is unavailable."""

    if method == "legacy-ycrcb":
        from .adaptive_skin_roi import build_masks_for_frame

        face_mask, _seed_mask = build_masks_for_frame(
            frame_bgr.shape,
            frame_points,
            sdk_points81,
            face_mask_mode="convex-hull",
        )
        active_face = np.asarray(face_mask, dtype=bool)
        if not np.any(active_face):
            return np.full(3, np.nan, dtype=float), 0
        rows, cols = np.nonzero(active_face)
        y0 = int(rows.min())
        y1 = int(rows.max()) + 1
        x0 = int(cols.min())
        x1 = int(cols.max()) + 1
        frame_crop = frame_bgr[y0:y1, x0:x1]
        face_crop = active_face[y0:y1, x0:x1]
        skin_crop = np.asarray(skin_detection(frame_crop), dtype=bool)
        return mean_bgr_and_pixels_from_bool_mask(frame_crop, face_crop & skin_crop)

    if builder is None:
        builder = make_skin_roi_builders(
            [method],
            exclude_mouth=exclude_mouth,
            fill_holes=fill_holes,
        )
    if builder is None:
        return np.full(3, np.nan, dtype=float), 0
    results = builder.build_many(frame_bgr, frame_points, sdk_points81, methods=[method])
    result = results.get(method)
    if result is None:
        return np.full(3, np.nan, dtype=float), 0
    return mean_bgr_and_pixels_from_bool_mask(frame_bgr, result.final_mask)

def valid_rounded_polygon(frame_bgr: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
    """Validate, clip, and integer-round one polygon ROI for OpenCV masking."""

    polygon = np.asarray(polygon, dtype=np.float32)
    if polygon.shape[0] < 3 or not np.all(np.isfinite(polygon)):
        return None
    if np.allclose(polygon, 0.0):
        return None

    height, width = frame_bgr.shape[:2]
    rounded = np.rint(polygon).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    if np.unique(rounded, axis=0).shape[0] < 3:
        return None
    return rounded

def roi_mean_bgr_from_polygon_bbox_cv2(
    frame_bgr: np.ndarray,
    polygon: np.ndarray,
    cv2_module,
    frame_skin_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Compute one polygon ROI's skin-filtered mean BGR inside its bbox."""

    rounded = valid_rounded_polygon(frame_bgr, polygon)
    if rounded is None:
        return np.full(3, np.nan, dtype=float), 0

    x, y, w, h = cv2_module.boundingRect(rounded)
    if w <= 0 or h <= 0:
        return np.full(3, np.nan, dtype=float), 0

    local_polygon = rounded - np.array([x, y], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2_module.fillPoly(mask, [local_polygon], 255)
    if frame_skin_mask is None:
        skin_mask = skin_detection(frame_bgr[y : y + h, x : x + w])
    else:
        skin_mask = np.asarray(frame_skin_mask[y : y + h, x : x + w])
    mask = np.where((mask > 0) & skin_mask, 255, 0).astype(np.uint8)
    pixel_count = int(cv2_module.countNonZero(mask))
    if pixel_count <= 0:
        return np.full(3, np.nan, dtype=float), 0

    mean_bgr = np.asarray(cv2_module.mean(frame_bgr[y : y + h, x : x + w], mask=mask)[:3], dtype=float)
    return mean_bgr, pixel_count

def sdk_facearea_mask_full_frame_cv2(frame_bgr: np.ndarray, sdk_points: np.ndarray, cv2_module) -> np.ndarray:
    """Return the SDK face-area mask in full-frame coordinates."""

    roi = initialize_facearea_roi(sdk_points)
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    if roi is None:
        return mask

    left, right, cheek_points, mouth_points = roi
    lx, ly = left
    rx, ry = right
    height, width = frame_bgr.shape[:2]
    if lx <= 0 or ly <= 0 or rx >= width or ry >= height:
        return mask
    if rx <= lx or ry <= ly:
        return mask

    cheek = np.asarray(cheek_points, dtype=np.int32).reshape((-1, 1, 2))
    mouth = np.asarray(mouth_points, dtype=np.int32).reshape((-1, 1, 2))
    face_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    mouth_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2_module.fillConvexPoly(face_mask, cheek, 255)
    cv2_module.fillConvexPoly(mouth_mask, mouth, 255)
    return np.where((face_mask > 10) & ~(mouth_mask > 10), 255, 0).astype(np.uint8)

def roi_union_skin_mask(
    frame_bgr: np.ndarray,
    frame_points: np.ndarray,
    *,
    sdk_points81: np.ndarray | None = None,
    valid_polygon_names: Iterable[str] | None = None,
    include_sdk_facearea_roi: bool,
) -> np.ndarray:
    """Build one shared skin mask over requested polygon/SDK ROIs."""

    import cv2

    union_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    valid_names = set(POLYGON_ROI_NAMES if valid_polygon_names is None else valid_polygon_names)
    for name, indices in ROI_POLYGON_INDICES:
        if name not in valid_names:
            continue
        rounded = valid_rounded_polygon(frame_bgr, np.asarray(frame_points)[indices])
        if rounded is not None:
            cv2.fillPoly(union_mask, [rounded], 255)

    if include_sdk_facearea_roi and sdk_points81 is not None:
        sdk_mask = sdk_facearea_mask_full_frame_cv2(frame_bgr, sdk_points81, cv2)
        union_mask = np.where((union_mask > 0) | (sdk_mask > 0), 255, 0).astype(np.uint8)

    if not np.any(union_mask > 0):
        return np.zeros(frame_bgr.shape[:2], dtype=bool)

    x, y, w, h = cv2.boundingRect(union_mask)
    if w <= 0 or h <= 0:
        return np.zeros(frame_bgr.shape[:2], dtype=bool)

    skin_crop = skin_detection(frame_bgr[y : y + h, x : x + w])
    result = np.zeros(frame_bgr.shape[:2], dtype=bool)
    result[y : y + h, x : x + w] = (union_mask[y : y + h, x : x + w] > 0) & skin_crop
    return result


def extract_frame_roi_values(
    frame_bgr: np.ndarray,
    frame_landmarks: FrameLandmarks,
    *,
    roi_names: Iterable[str],
    skin_roi_builders: object | None,
    skin_roi_exclude_mouth: bool = False,
    skin_roi_fill_holes: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], dict[str, float]]:
    """Extract masks, BGR values, pixel counts, and area ratios for one frame."""

    import cv2

    roi_names = tuple(roi_names)
    frame_points = frame_landmarks.points
    active_polygon_names = [name for name in POLYGON_ROI_NAMES if name in roi_names]
    sdk_requested = SDK_FACEAREA_ROI_NAME in roi_names
    active_skin_methods = skin_roi_methods_for_roi_names(roi_names)
    sdk_points81 = (
        np.asarray(frame_landmarks.sdk_points81, dtype=float)
        if frame_landmarks.sdk_points81 is not None
        else spipnet280_to_sdk81_points(frame_points)
        if (sdk_requested or bool(active_skin_methods))
        else None
    )
    valid_landmark_masks = frame_landmark_valid_masks(
        frame_landmarks,
        roi_names=roi_names,
        sdk_points81=sdk_points81,
    )
    valid_polygon_names = [
        name
        for name in active_polygon_names
        if valid_landmark_masks.get(name, False)
    ]
    sdk_valid = (
        (sdk_requested or bool(active_skin_methods))
        and sdk_points81 is not None
        and valid_landmark_masks.get(SDK_FACEAREA_ROI_NAME, False)
    )
    valid_skin_methods = [
        method
        for method in active_skin_methods
        if valid_landmark_masks.get(SKIN_ROI_NAME_BY_METHOD[method], False)
    ]
    frame_sdk_points81 = sdk_points81 if sdk_valid else None
    needs_frame_skin_mask = bool(valid_polygon_names) or sdk_valid
    frame_skin_mask = (
        roi_union_skin_mask(
            frame_bgr,
            frame_points,
            valid_polygon_names=valid_polygon_names,
            sdk_points81=frame_sdk_points81,
            include_sdk_facearea_roi=sdk_valid,
        )
        if needs_frame_skin_mask
        else None
    )

    face_area = float("nan")
    if frame_landmarks.face_bbox is not None:
        face_area = face_area_from_bbox(frame_landmarks.face_bbox)
    if not np.isfinite(face_area) or face_area <= 0.0:
        face_area = face_bbox_area_from_landmarks(frame_points)

    roi_masks = {name: np.zeros(frame_bgr.shape[:2], dtype=bool) for name in roi_names}
    roi_bgr = {name: np.full(3, np.nan, dtype=float) for name in roi_names}
    roi_pixels = {name: 0 for name in roi_names}
    roi_area_ratios = {name: float("nan") for name in roi_names}

    for name, indices in ROI_POLYGON_INDICES:
        if name not in roi_names:
            continue
        if not valid_landmark_masks.get(name, False):
            continue
        rounded = valid_rounded_polygon(frame_bgr, frame_points[indices])
        if rounded is None:
            continue
        x, y, w, h = cv2.boundingRect(rounded)
        if w <= 0 or h <= 0:
            continue
        local_polygon = rounded - np.array([x, y], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [local_polygon], 255)
        if frame_skin_mask is None:
            skin_mask = skin_detection(frame_bgr[y : y + h, x : x + w])
        else:
            skin_mask = np.asarray(frame_skin_mask[y : y + h, x : x + w])
        full_mask = np.zeros(frame_bgr.shape[:2], dtype=bool)
        full_mask[y : y + h, x : x + w] = (mask > 0) & skin_mask
        mean_bgr, pixel_count = roi_mean_bgr_from_polygon_bbox_cv2(
            frame_bgr,
            frame_points[indices],
            cv2,
            frame_skin_mask=frame_skin_mask,
        )
        store_roi_value(name, full_mask, mean_bgr, pixel_count, face_area, roi_masks, roi_bgr, roi_pixels, roi_area_ratios)

    if sdk_requested and sdk_valid and frame_sdk_points81 is not None:
        face_mask = sdk_facearea_mask_full_frame_cv2(frame_bgr, frame_sdk_points81, cv2) > 10
        if frame_skin_mask is None:
            combined = face_mask
        else:
            combined = combine_mask(face_mask, frame_skin_mask) > 10
            if not np.any(combined):
                combined = face_mask
        mean_bgr, pixel_count = sdk_facearea_mean_bgr_and_pixels_with_skin_mask(
            frame_bgr,
            frame_sdk_points81,
            frame_skin_mask,
        )
        store_roi_value(SDK_FACEAREA_ROI_NAME, combined, mean_bgr, pixel_count, face_area, roi_masks, roi_bgr, roi_pixels, roi_area_ratios)

    if "legacy-ycrcb" in valid_skin_methods:
        from .adaptive_skin_roi import build_masks_for_frame

        name = SKIN_ROI_NAME_BY_METHOD["legacy-ycrcb"]
        face_mask, _seed_mask = build_masks_for_frame(
            frame_bgr.shape,
            frame_points,
            frame_sdk_points81,
            face_mask_mode="convex-hull",
        )
        active_face = np.asarray(face_mask, dtype=bool)
        full_mask = np.zeros(frame_bgr.shape[:2], dtype=bool)
        if np.any(active_face):
            rows, cols = np.nonzero(active_face)
            y0 = int(rows.min())
            y1 = int(rows.max()) + 1
            x0 = int(cols.min())
            x1 = int(cols.max()) + 1
            skin_crop = np.asarray(skin_detection(frame_bgr[y0:y1, x0:x1]), dtype=bool)
            full_mask[y0:y1, x0:x1] = active_face[y0:y1, x0:x1] & skin_crop
        mean_bgr, pixel_count = skin_roi_mean_bgr_and_pixels(
            frame_bgr,
            frame_points,
            frame_sdk_points81,
            method="legacy-ycrcb",
            exclude_mouth=skin_roi_exclude_mouth,
            fill_holes=skin_roi_fill_holes,
        )
        store_roi_value(name, full_mask, mean_bgr, pixel_count, face_area, roi_masks, roi_bgr, roi_pixels, roi_area_ratios)

    adaptive_skin_methods = [method for method in valid_skin_methods if method != "legacy-ycrcb"]
    if adaptive_skin_methods:
        builder = skin_roi_builders or make_skin_roi_builders(
            adaptive_skin_methods,
            exclude_mouth=skin_roi_exclude_mouth,
            fill_holes=skin_roi_fill_holes,
        )
        if builder is not None:
            results = builder.build_many(frame_bgr, frame_points, frame_sdk_points81, methods=adaptive_skin_methods)
            for method in adaptive_skin_methods:
                name = SKIN_ROI_NAME_BY_METHOD[method]
                result = results[method]
                full_mask = np.asarray(result.final_mask, dtype=bool)
                crop_mean = mean_bgr_and_pixels_from_crop_mask(
                    frame_bgr,
                    getattr(result, "crop_bounds", None),
                    getattr(result, "final_crop_mask", None),
                )
                mean_bgr, pixel_count = (
                    crop_mean if crop_mean is not None else mean_bgr_and_pixels_from_bool_mask(frame_bgr, full_mask)
                )
                store_roi_value(name, full_mask, mean_bgr, pixel_count, face_area, roi_masks, roi_bgr, roi_pixels, roi_area_ratios)

    return roi_masks, roi_bgr, roi_pixels, roi_area_ratios
