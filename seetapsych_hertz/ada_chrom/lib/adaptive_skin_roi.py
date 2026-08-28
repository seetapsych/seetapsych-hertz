from __future__ import annotations

"""Adaptive skin ROI builders used by the delivery estimator.

Only the mask-building classes and helpers are required by the delivery path.
Source-project utilities unrelated to ROI mask construction are omitted.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .roi_mask_utils import initialize_facearea_roi
from .roi_definitions import ROI_POLYGON_INDICES


DEFAULT_INCLUDE_FOREHEAD = True
DEFAULT_EXCLUDE_EYES_BROWS = True
DEFAULT_INCLUDE_NOSE_SEED = True
DEFAULT_USE_BRIGHTNESS_GATE = True
ROI_METHODS = ("a-fixed-forehead", "b-adaptive-forehead", "c-connected-components")
LEFT_EYEBROW_INDICES = (33, 34, 35, 36, 37, 67, 66, 65, 64)
RIGHT_EYEBROW_INDICES = (38, 39, 40, 41, 42, 71, 70, 69, 68)
LEFT_EYE_INDICES = (52, 53, 54, 55, 56, 57, 72, 73, 74)
RIGHT_EYE_INDICES = (58, 59, 60, 61, 62, 63, 75, 76, 77)
NOSE_SEED_INDICES = (78, 43, 79, 51, 50, 49, 48, 47)


@dataclass
class SkinSegmentationResult:
    mask: np.ndarray
    model_ready: bool
    seed_pixels: int
    face_pixels: int
    skin_pixels: int
    mean: np.ndarray | None
    covariance: np.ndarray | None
    status: str


@dataclass
class FullFaceRoiResult:
    method: str
    face_mask: np.ndarray
    seed_mask: np.ndarray
    candidate_mask: np.ndarray
    final_mask: np.ndarray
    skin_result: SkinSegmentationResult
    status: str
    crop_bounds: tuple[int, int, int, int] | None = None
    final_crop_mask: np.ndarray | None = None


class LandmarkAdaptiveSkinSegmenter:
    """Per-video adaptive skin model learned from landmark-defined seed pixels."""

    def __init__(
        self,
        *,
        mahalanobis_threshold: float = 9.0,
        min_seed_pixels: int = 80,
        update_alpha: float = 0.05,
        covariance_regularization: float = 8.0,
        use_luma: bool = False,
    ) -> None:
        self.mahalanobis_threshold = float(mahalanobis_threshold)
        self.min_seed_pixels = int(min_seed_pixels)
        self.update_alpha = float(update_alpha)
        self.covariance_regularization = float(covariance_regularization)
        self.use_luma = bool(use_luma)
        self.mean_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.inv_covariance_: np.ndarray | None = None

    @property
    def model_ready(self) -> bool:
        return self.mean_ is not None and self.inv_covariance_ is not None

    def reset(self) -> None:
        self.mean_ = None
        self.covariance_ = None
        self.inv_covariance_ = None

    def fit_features(self, features: np.ndarray) -> bool:
        if features.shape[0] < self.min_seed_pixels:
            return False
        mean, covariance = self._estimate_distribution(features)
        self._set_model(mean, covariance)
        return True

    def update_features(self, features: np.ndarray) -> bool:
        if features.shape[0] < self.min_seed_pixels:
            return False
        mean, covariance = self._estimate_distribution(features)
        if not self.model_ready:
            self._set_model(mean, covariance)
            return True
        alpha = float(np.clip(self.update_alpha, 0.0, 1.0))
        self._set_model((1.0 - alpha) * self.mean_ + alpha * mean, (1.0 - alpha) * self.covariance_ + alpha * covariance)
        return True

    def segment_with_masks(
        self,
        frame_bgr: np.ndarray,
        face_mask: np.ndarray,
        *,
        seed_mask: np.ndarray | None = None,
        update_model: bool = True,
        frame_features: np.ndarray | None = None,
    ) -> SkinSegmentationResult:
        face = np.asarray(face_mask, dtype=bool)
        seed = np.zeros(face.shape, dtype=bool) if seed_mask is None else np.asarray(seed_mask, dtype=bool)
        seed = seed & face
        seed_pixels = int(np.count_nonzero(seed))
        face_pixels = int(np.count_nonzero(face))
        features = self._frame_features(frame_bgr) if frame_features is None else np.asarray(frame_features)

        if update_model and seed_mask is not None:
            self.update_features(features[seed])
        elif not self.model_ready and seed_mask is not None:
            self.fit_features(features[seed])

        if not self.model_ready:
            return SkinSegmentationResult(
                mask=np.zeros(face.shape, dtype=bool),
                model_ready=False,
                seed_pixels=seed_pixels,
                face_pixels=face_pixels,
                skin_pixels=0,
                mean=None,
                covariance=None,
                status="insufficient_seed",
            )

        mask = np.zeros(face.shape, dtype=bool)
        if face_pixels:
            face_distance2 = _mahalanobis_distance2(features[face], self.mean_, self.inv_covariance_)
            mask[face] = np.isfinite(face_distance2) & (face_distance2 <= self.mahalanobis_threshold)
        return SkinSegmentationResult(
            mask=mask,
            model_ready=True,
            seed_pixels=seed_pixels,
            face_pixels=face_pixels,
            skin_pixels=int(np.count_nonzero(mask)),
            mean=self.mean_.copy(),
            covariance=self.covariance_.copy(),
            status="ok" if np.any(mask) else "empty_skin_mask",
        )

    def _frame_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        ycrcb = bgr_to_ycrcb_float(frame_bgr)
        if self.use_luma:
            return ycrcb
        return ycrcb[:, :, 1:3]

    def _estimate_distribution(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(features, dtype=np.float64)
        low, high = np.percentile(features, [5.0, 95.0], axis=0)
        keep = np.all((features >= low) & (features <= high), axis=1)
        robust = features[keep]
        if robust.shape[0] >= max(3, self.min_seed_pixels // 4):
            features = robust
        mean = np.mean(features, axis=0)
        if features.shape[0] <= 1:
            covariance = np.eye(features.shape[1], dtype=np.float64)
        else:
            covariance = np.cov(features, rowvar=False)
            if covariance.ndim == 0:
                covariance = np.asarray([[float(covariance)]], dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)
        covariance += np.eye(covariance.shape[0], dtype=np.float64) * self.covariance_regularization
        return mean, covariance

    def _set_model(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.covariance_ = np.asarray(covariance, dtype=np.float64)
        self.inv_covariance_ = np.linalg.pinv(self.covariance_)


def _mahalanobis_distance2(features: np.ndarray, mean: np.ndarray, inv_covariance: np.ndarray) -> np.ndarray:
    if np.shape(features)[-1] == 2 and np.shape(inv_covariance) == (2, 2):
        features = np.asarray(features, dtype=np.float32)
        mean = np.asarray(mean, dtype=np.float32)
        inv_covariance = np.asarray(inv_covariance, dtype=np.float32)
        delta = features - mean
        dx0 = delta[..., 0]
        dx1 = delta[..., 1]
        a = inv_covariance[0, 0]
        b = inv_covariance[0, 1] + inv_covariance[1, 0]
        c = inv_covariance[1, 1]
        return a * dx0 * dx0 + b * dx0 * dx1 + c * dx1 * dx1
    features = np.asarray(features, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    inv_covariance = np.asarray(inv_covariance, dtype=np.float64)
    delta = features - mean
    return np.einsum("...i,ij,...j->...", delta, inv_covariance, delta)


def _uncrop_bool_mask(mask: np.ndarray, full_shape: tuple[int, int], crop_bounds: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = crop_bounds
    result = np.zeros(full_shape, dtype=bool)
    result[y0:y1, x0:x1] = np.asarray(mask, dtype=bool)
    return result


class FullFaceRoiBuilder:
    """Build full-face skin ROI masks with one of the A/B/C forehead strategies."""

    def __init__(
        self,
        *,
        method: str,
        face_mask_mode: str = "convex-hull",
        mahalanobis_threshold: float = 9.0,
        min_seed_pixels: int = 80,
        include_forehead: bool = DEFAULT_INCLUDE_FOREHEAD,
        connected_max_seed_distance: int = 36,
        connected_min_component_area: int = 64,
        exclude_eyes_brows: bool = DEFAULT_EXCLUDE_EYES_BROWS,
        exclude_mouth: bool = False,
        include_nose_seed: bool = DEFAULT_INCLUDE_NOSE_SEED,
        use_brightness_gate: bool = DEFAULT_USE_BRIGHTNESS_GATE,
        fill_holes: bool = True,
    ) -> None:
        if method not in ROI_METHODS:
            raise ValueError(f"Unknown ROI method: {method}")
        self.method = method
        self.face_mask_mode = face_mask_mode
        self.include_forehead = bool(include_forehead)
        self.exclude_eyes_brows = bool(exclude_eyes_brows)
        self.exclude_mouth = bool(exclude_mouth)
        self.include_nose_seed = bool(include_nose_seed)
        self.use_brightness_gate = bool(use_brightness_gate)
        self.fill_holes = bool(fill_holes)
        self.connected_max_seed_distance = int(connected_max_seed_distance)
        self.connected_min_component_area = int(connected_min_component_area)
        self.segmenter = LandmarkAdaptiveSkinSegmenter(
            mahalanobis_threshold=mahalanobis_threshold,
            min_seed_pixels=min_seed_pixels,
        )

    def reset(self) -> None:
        self.segmenter.reset()

    def build_from_masks(
        self,
        frame_bgr: np.ndarray,
        points280: np.ndarray,
        face_mask: np.ndarray,
        seed_mask: np.ndarray,
        *,
        frame_features: np.ndarray | None,
        luma: np.ndarray | None,
        exclusion_mask: np.ndarray | None,
    ) -> FullFaceRoiResult:
        skin_result = self.segmenter.segment_with_masks(
            frame_bgr,
            face_mask,
            seed_mask=seed_mask,
            update_model=True,
            frame_features=frame_features,
        )
        candidate = skin_result.mask
        exclusion = (
            semantic_exclusion_mask_from_landmarks(frame_bgr.shape, points280)
            if exclusion_mask is None and self.exclude_eyes_brows
            else exclusion_mask
        )
        if self.use_brightness_gate:
            candidate = candidate & brightness_gate_mask(frame_bgr, face_mask=face_mask, luma=luma)
        if exclusion is not None:
            candidate = candidate & ~exclusion
        if self.method == "c-connected-components":
            final = filter_connected_skin_components(
                candidate,
                seed_mask,
                max_seed_distance=self.connected_max_seed_distance,
                min_component_area=self.connected_min_component_area,
            )
            status = skin_result.status if np.any(final) else "empty_connected_mask"
        else:
            final = candidate
            status = skin_result.status
        if self.fill_holes:
            final = refine_skin_mask(final, exclusion_mask=exclusion)
        else:
            final = refine_skin_mask(final, exclusion_mask=exclusion, fill_holes=False)
        return FullFaceRoiResult(
            method=self.method,
            face_mask=face_mask,
            seed_mask=seed_mask,
            candidate_mask=candidate,
            final_mask=final,
            skin_result=skin_result,
            status=status,
            crop_bounds=None,
            final_crop_mask=None,
        )

    def build_from_crop_masks(
        self,
        frame_bgr: np.ndarray,
        face_mask: np.ndarray,
        seed_mask: np.ndarray,
        crop_ycrcb: np.ndarray,
        *,
        crop_bounds: tuple[int, int, int, int],
        exclusion_mask: np.ndarray | None,
    ) -> FullFaceRoiResult:
        y0, y1, x0, x1 = crop_bounds
        frame_crop = frame_bgr[y0:y1, x0:x1]
        face_crop = np.asarray(face_mask[y0:y1, x0:x1], dtype=bool)
        seed_crop = np.asarray(seed_mask[y0:y1, x0:x1], dtype=bool)
        exclusion_crop = None if exclusion_mask is None else np.asarray(exclusion_mask[y0:y1, x0:x1], dtype=bool)
        frame_features = crop_ycrcb if self.segmenter.use_luma else crop_ycrcb[:, :, 1:3]
        skin_result = self.segmenter.segment_with_masks(
            frame_crop,
            face_crop,
            seed_mask=seed_crop,
            update_model=True,
            frame_features=frame_features,
        )
        candidate_crop = skin_result.mask
        if self.use_brightness_gate:
            candidate_crop = candidate_crop & brightness_gate_mask(frame_crop, face_mask=face_crop, luma=crop_ycrcb[:, :, 0])
        if exclusion_crop is not None:
            candidate_crop = candidate_crop & ~exclusion_crop
        if self.method == "c-connected-components":
            final_crop = filter_connected_skin_components(
                candidate_crop,
                seed_crop,
                max_seed_distance=self.connected_max_seed_distance,
                min_component_area=self.connected_min_component_area,
            )
            status = skin_result.status if np.any(final_crop) else "empty_connected_mask"
        else:
            final_crop = candidate_crop
            status = skin_result.status
        full_shape = frame_bgr.shape[:2]
        candidate = _uncrop_bool_mask(candidate_crop, full_shape, crop_bounds)
        if self.fill_holes:
            final = refine_skin_mask(candidate, exclusion_mask=exclusion_mask)
        else:
            final = refine_skin_mask(candidate, exclusion_mask=exclusion_mask, fill_holes=False)
        skin_full_mask = _uncrop_bool_mask(skin_result.mask, full_shape, crop_bounds)
        full_skin_result = SkinSegmentationResult(
            mask=skin_full_mask,
            model_ready=skin_result.model_ready,
            seed_pixels=skin_result.seed_pixels,
            face_pixels=skin_result.face_pixels,
            skin_pixels=skin_result.skin_pixels,
            mean=skin_result.mean,
            covariance=skin_result.covariance,
            status=skin_result.status,
        )
        return FullFaceRoiResult(
            method=self.method,
            face_mask=face_mask,
            seed_mask=seed_mask,
            candidate_mask=candidate,
            final_mask=final,
            skin_result=full_skin_result,
            status=status,
            crop_bounds=crop_bounds,
            final_crop_mask=final[y0:y1, x0:x1],
        )


@dataclass
class FullFaceRoiFrameContext:
    frame_bgr: np.ndarray
    points280: np.ndarray
    sdk_points81: np.ndarray | None
    exclusion_mask: np.ndarray | None
    base_face_masks: dict[str, np.ndarray]


class MultiFullFaceRoiBuilder:
    """Build several full-face skin ROIs while reusing per-frame features and masks."""

    def __init__(
        self,
        methods: Iterable[str] = ROI_METHODS,
        *,
        face_mask_mode: str = "convex-hull",
        mahalanobis_threshold: float = 9.0,
        min_seed_pixels: int = 80,
        include_forehead: bool = DEFAULT_INCLUDE_FOREHEAD,
        connected_max_seed_distance: int = 36,
        connected_min_component_area: int = 64,
        exclude_eyes_brows: bool = DEFAULT_EXCLUDE_EYES_BROWS,
        exclude_mouth: bool = False,
        include_nose_seed: bool = DEFAULT_INCLUDE_NOSE_SEED,
        use_brightness_gate: bool = DEFAULT_USE_BRIGHTNESS_GATE,
        fill_holes: bool = True,
    ) -> None:
        method_list = []
        for method in methods:
            if method not in ROI_METHODS:
                raise ValueError(f"Unknown ROI method: {method}")
            if method not in method_list:
                method_list.append(method)
        self.methods = tuple(method_list)
        self.builders = {
            method: FullFaceRoiBuilder(
                method=method,
                face_mask_mode=face_mask_mode,
                mahalanobis_threshold=mahalanobis_threshold,
                min_seed_pixels=min_seed_pixels,
                include_forehead=include_forehead,
                connected_max_seed_distance=connected_max_seed_distance,
                connected_min_component_area=connected_min_component_area,
                exclude_eyes_brows=exclude_eyes_brows,
                exclude_mouth=exclude_mouth,
                include_nose_seed=include_nose_seed,
                use_brightness_gate=use_brightness_gate,
                fill_holes=fill_holes,
            )
            for method in self.methods
        }

    def reset(self) -> None:
        for builder in self.builders.values():
            builder.reset()

    def build_many(
        self,
        frame_bgr: np.ndarray,
        points280: np.ndarray,
        sdk_points81: np.ndarray | None,
        methods: Iterable[str] | None = None,
    ) -> dict[str, FullFaceRoiResult]:
        selected_methods = self.methods if methods is None else tuple(methods)
        context = self._make_context(frame_bgr, points280, sdk_points81)
        method_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        conversion_mask = np.zeros(frame_bgr.shape[:2], dtype=bool)
        for method in selected_methods:
            builder = self.builders[method]
            face_mask, seed_mask = self._masks_for_builder(context, builder)
            method_masks[method] = (face_mask, seed_mask)
            conversion_mask |= face_mask | seed_mask
        ycrcb, crop_bounds = self._ycrcb_for_mask_bbox(frame_bgr, conversion_mask)
        results: dict[str, FullFaceRoiResult] = {}
        for method in selected_methods:
            builder = self.builders[method]
            face_mask, seed_mask = method_masks[method]
            exclusion_mask = context.exclusion_mask if (builder.exclude_eyes_brows or builder.exclude_mouth) else None
            if crop_bounds is None:
                frame_features = ycrcb if builder.segmenter.use_luma else ycrcb[:, :, 1:3]
                results[method] = builder.build_from_masks(
                    frame_bgr,
                    points280,
                    face_mask,
                    seed_mask,
                    frame_features=frame_features,
                    luma=ycrcb[:, :, 0],
                    exclusion_mask=exclusion_mask,
                )
            else:
                results[method] = builder.build_from_crop_masks(
                    frame_bgr,
                    face_mask,
                    seed_mask,
                    ycrcb,
                    crop_bounds=crop_bounds,
                    exclusion_mask=exclusion_mask,
                )
        return results

    def _make_context(
        self,
        frame_bgr: np.ndarray,
        points280: np.ndarray,
        sdk_points81: np.ndarray | None,
    ) -> FullFaceRoiFrameContext:
        needs_exclusion = any(
            builder.exclude_eyes_brows or builder.exclude_mouth
            for builder in self.builders.values()
        )
        if any(builder.exclude_mouth for builder in self.builders.values()):
            exclusion = combined_semantic_exclusion_mask(
                frame_bgr.shape,
                points280,
                sdk_points81,
                exclude_eyes_brows=any(builder.exclude_eyes_brows for builder in self.builders.values()),
                exclude_mouth=True,
            )
        elif needs_exclusion:
            exclusion = semantic_exclusion_mask_from_landmarks(frame_bgr.shape, points280)
        else:
            exclusion = None
        face_modes = {builder.face_mask_mode for builder in self.builders.values()}
        base_masks = {
            mode: base_face_mask(frame_bgr.shape, points280, sdk_points81, face_mask_mode=mode)
            for mode in face_modes
        }
        return FullFaceRoiFrameContext(
            frame_bgr=frame_bgr,
            points280=points280,
            sdk_points81=sdk_points81,
            exclusion_mask=exclusion,
            base_face_masks=base_masks,
        )

    def _ycrcb_for_mask_bbox(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        selected = np.asarray(mask, dtype=bool)
        if not np.any(selected):
            return bgr_to_ycrcb_float(frame_bgr), None
        rows, cols = np.nonzero(selected)
        y0 = int(rows.min())
        y1 = int(rows.max()) + 1
        x0 = int(cols.min())
        x1 = int(cols.max()) + 1
        ycrcb = bgr_to_ycrcb_float(frame_bgr[y0:y1, x0:x1])
        return ycrcb, (y0, y1, x0, x1)

    def _masks_for_builder(
        self,
        context: FullFaceRoiFrameContext,
        builder: FullFaceRoiBuilder,
    ) -> tuple[np.ndarray, np.ndarray]:
        return build_masks_for_roi_method(
            context.frame_bgr.shape,
            context.points280,
            context.sdk_points81,
            method=builder.method,
            face_mask_mode=builder.face_mask_mode,
            include_forehead=builder.include_forehead,
            exclude_eyes_brows=builder.exclude_eyes_brows,
            exclude_mouth=builder.exclude_mouth,
            include_nose_seed=builder.include_nose_seed,
            exclusion_mask=context.exclusion_mask if (builder.exclude_eyes_brows or builder.exclude_mouth) else None,
            base_face_mask_value=context.base_face_masks[builder.face_mask_mode],
        )


def bgr_to_ycrcb_float(frame_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(frame_bgr, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("frame_bgr must have shape (H, W, 3).")
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    ycrcb = np.empty(image.shape[:2] + (3,), dtype=np.float32)
    y = ycrcb[:, :, 0]
    np.multiply(r, np.float32(0.299), out=y)
    y += np.float32(0.587) * g
    y += np.float32(0.114) * b
    ycrcb[:, :, 1] = (r - y) * np.float32(0.713) + np.float32(128.0)
    ycrcb[:, :, 2] = (b - y) * np.float32(0.564) + np.float32(128.0)
    return ycrcb


def brightness_gate_mask(
    frame_bgr: np.ndarray,
    *,
    face_mask: np.ndarray | None = None,
    min_luma_percentile: float = 8.0,
    min_luma_offset: float = -5.0,
    luma: np.ndarray | None = None,
) -> np.ndarray:
    y = bgr_to_ycrcb_float(frame_bgr)[:, :, 0] if luma is None else np.asarray(luma, dtype=np.float64)
    if face_mask is not None and np.any(face_mask):
        reference = y[np.asarray(face_mask, dtype=bool)]
    else:
        reference = y.reshape(-1)
    if reference.size == 0:
        return np.ones(y.shape, dtype=bool)
    threshold = float(np.percentile(reference, min_luma_percentile)) + float(min_luma_offset)
    return y >= threshold


def _valid_polygon_from_indices(
    frame_shape: tuple[int, ...],
    points: np.ndarray,
    indices: Iterable[int],
) -> np.ndarray | None:
    coords = np.asarray(points, dtype=np.float32)
    index_array = np.asarray(list(indices), dtype=int)
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] <= int(np.max(index_array)):
        return None
    polygon = coords[index_array, :2].copy()
    if polygon.shape[0] < 3 or not np.all(np.isfinite(polygon)):
        return None
    height, width = frame_shape[:2]
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    rounded = np.rint(polygon).astype(np.int32)
    if np.unique(rounded, axis=0).shape[0] < 3:
        return None
    return rounded


def _fill_landmark_polygon(
    mask: np.ndarray,
    frame_shape: tuple[int, ...],
    points: np.ndarray,
    indices: Iterable[int],
) -> None:
    import cv2

    polygon = _valid_polygon_from_indices(frame_shape, points, indices)
    if polygon is not None:
        cv2.fillConvexPoly(mask, polygon.reshape((-1, 1, 2)), 255)


def semantic_exclusion_mask_from_landmarks(
    frame_shape: tuple[int, ...],
    points: np.ndarray,
    *,
    dilate_pixels: int = 5,
) -> np.ndarray:
    import cv2

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for indices in (LEFT_EYEBROW_INDICES, RIGHT_EYEBROW_INDICES, LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        _fill_landmark_polygon(mask, frame_shape, points, indices)
    if dilate_pixels > 0 and np.any(mask):
        kernel = np.ones((dilate_pixels * 2 + 1, dilate_pixels * 2 + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 10


def mouth_exclusion_mask_from_sdk81(
    frame_shape: tuple[int, ...],
    sdk_points81: np.ndarray | None,
) -> np.ndarray:
    if sdk_points81 is None:
        return np.zeros(frame_shape[:2], dtype=bool)
    _face_mask, mouth_mask = sdk_facearea_mask(frame_shape, sdk_points81)
    return np.asarray(mouth_mask, dtype=bool)


def combined_semantic_exclusion_mask(
    frame_shape: tuple[int, ...],
    points: np.ndarray,
    sdk_points81: np.ndarray | None,
    *,
    exclude_eyes_brows: bool = DEFAULT_EXCLUDE_EYES_BROWS,
    exclude_mouth: bool = False,
    dilate_pixels: int = 5,
) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=bool)
    if exclude_eyes_brows:
        mask |= semantic_exclusion_mask_from_landmarks(frame_shape, points, dilate_pixels=dilate_pixels)
    if exclude_mouth:
        mask |= mouth_exclusion_mask_from_sdk81(frame_shape, sdk_points81)
    return mask


def nose_seed_mask_from_landmarks(
    frame_shape: tuple[int, ...],
    points: np.ndarray,
    *,
    erode_pixels: int = 1,
) -> np.ndarray:
    import cv2

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    _fill_landmark_polygon(mask, frame_shape, points, NOSE_SEED_INDICES)
    if erode_pixels > 0 and np.any(mask):
        kernel = np.ones((erode_pixels * 2 + 1, erode_pixels * 2 + 1), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask > 10


def refine_skin_mask(
    mask: np.ndarray,
    *,
    exclusion_mask: np.ndarray | None = None,
    close_pixels: int = 2,
    fill_holes: bool = True,
) -> np.ndarray:
    import cv2

    refined = np.asarray(mask, dtype=bool)
    if not np.any(refined):
        return refined.copy()
    work = refined.astype(np.uint8) * 255
    if close_pixels > 0:
        kernel = np.ones((close_pixels * 2 + 1, close_pixels * 2 + 1), dtype=np.uint8)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=1)
    if fill_holes:
        height, width = work.shape
        flood = work.copy()
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        work = cv2.bitwise_or(work, holes)
    refined = work > 10
    if exclusion_mask is not None:
        refined &= ~np.asarray(exclusion_mask, dtype=bool)
    return refined


def sdk_facearea_mask(frame_shape: tuple[int, ...], sdk_points81: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    face_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    mouth_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    roi = initialize_facearea_roi(sdk_points81, frame_shape)
    if roi is None:
        return face_mask.astype(bool), mouth_mask.astype(bool)
    _left, _right, cheek_points, mouth_points = roi
    cheek = np.asarray(cheek_points, dtype=np.int32)
    mouth = np.asarray(mouth_points, dtype=np.int32)
    if cheek.shape[0] >= 3:
        cv2.fillConvexPoly(face_mask, cheek.reshape((-1, 1, 2)), 255)
    if mouth.shape[0] >= 3:
        cv2.fillConvexPoly(mouth_mask, mouth.reshape((-1, 1, 2)), 255)
    return (face_mask > 10) & ~(mouth_mask > 10), mouth_mask > 10


def forehead_polygon_from_sdk81(
    frame_shape: tuple[int, ...],
    sdk_points81: np.ndarray | None,
    *,
    width_scale: float = 1.35,
    height_scale: float = 0.65,
    bottom_offset_scale: float = 0.20,
) -> np.ndarray | None:
    points = np.asarray(sdk_points81, dtype=np.float64) if sdk_points81 is not None else np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] <= 9 or points.shape[1] < 2:
        return None
    left_eye = points[0, :2]
    right_eye = points[9, :2]
    if not np.all(np.isfinite(left_eye)) or not np.all(np.isfinite(right_eye)):
        return None
    eye_vector = right_eye - left_eye
    eye_length = float(np.linalg.norm(eye_vector))
    if eye_length < 3.0:
        return None

    x_axis = eye_vector / eye_length
    up_axis = np.asarray([x_axis[1], -x_axis[0]], dtype=np.float64)
    center = (left_eye + right_eye) * 0.5
    bottom_center = center + up_axis * (bottom_offset_scale * eye_length)
    top_center = bottom_center + up_axis * (height_scale * eye_length)
    half_width = 0.5 * width_scale * eye_length
    polygon = np.asarray(
        [
            top_center - x_axis * half_width,
            top_center + x_axis * half_width,
            bottom_center + x_axis * half_width,
            bottom_center - x_axis * half_width,
        ],
        dtype=np.float64,
    )
    height, width = int(frame_shape[0]), int(frame_shape[1])
    polygon[:, 0] = np.clip(polygon[:, 0], 0, max(width - 1, 0))
    polygon[:, 1] = np.clip(polygon[:, 1], 0, max(height - 1, 0))
    if np.unique(np.rint(polygon).astype(np.int32), axis=0).shape[0] < 3:
        return None
    return polygon


def add_forehead_to_mask(mask: np.ndarray, frame_shape: tuple[int, ...], sdk_points81: np.ndarray | None) -> np.ndarray:
    import cv2

    result = np.asarray(mask, dtype=bool).copy()
    polygon = forehead_polygon_from_sdk81(frame_shape, sdk_points81)
    if polygon is None:
        return result
    forehead = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(forehead, np.rint(polygon).astype(np.int32).reshape((-1, 1, 2)), 255)
    return result | (forehead > 10)


def forehead_mask_from_sdk81(
    frame_shape: tuple[int, ...],
    sdk_points81: np.ndarray | None,
    *,
    width_scale: float,
    height_scale: float,
    bottom_offset_scale: float = 0.20,
) -> np.ndarray:
    import cv2

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    polygon = forehead_polygon_from_sdk81(
        frame_shape,
        sdk_points81,
        width_scale=width_scale,
        height_scale=height_scale,
        bottom_offset_scale=bottom_offset_scale,
    )
    if polygon is not None:
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32).reshape((-1, 1, 2)), 255)
    return mask > 10


def add_forehead_search_to_mask(
    mask: np.ndarray,
    frame_shape: tuple[int, ...],
    sdk_points81: np.ndarray | None,
    *,
    width_scale: float,
    height_scale: float,
    bottom_offset_scale: float = 0.20,
) -> np.ndarray:
    return np.asarray(mask, dtype=bool) | forehead_mask_from_sdk81(
        frame_shape,
        sdk_points81,
        width_scale=width_scale,
        height_scale=height_scale,
        bottom_offset_scale=bottom_offset_scale,
    )


def convex_hull_face_mask(
    frame_shape: tuple[int, ...],
    points: np.ndarray,
    sdk_points81: np.ndarray | None = None,
    *,
    include_forehead: bool = DEFAULT_INCLUDE_FOREHEAD,
) -> np.ndarray:
    import cv2

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 3:
        return mask.astype(bool)
    height, width = frame_shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    hull = cv2.convexHull(np.rint(pts).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if sdk_points81 is not None:
        _sdk_mask, mouth_mask = sdk_facearea_mask(frame_shape, sdk_points81)
        mask[mouth_mask] = 0
    result = mask > 10
    if include_forehead:
        result = add_forehead_to_mask(result, frame_shape, sdk_points81)
    return result


def base_face_mask(
    frame_shape: tuple[int, ...],
    points280: np.ndarray,
    sdk_points81: np.ndarray | None,
    *,
    face_mask_mode: str,
) -> np.ndarray:
    if face_mask_mode == "sdk-facearea":
        if sdk_points81 is None:
            return np.zeros(frame_shape[:2], dtype=bool)
        face_mask, _mouth = sdk_facearea_mask(frame_shape, sdk_points81)
        return face_mask
    if face_mask_mode == "convex-hull":
        return convex_hull_face_mask(frame_shape, points280, sdk_points81, include_forehead=False)
    raise ValueError(f"Unknown face mask mode: {face_mask_mode}")


def seed_mask_from_roi_polygons(
    frame_shape: tuple[int, ...],
    points280: np.ndarray,
    *,
    sdk_points81: np.ndarray | None = None,
    include_forehead: bool = DEFAULT_INCLUDE_FOREHEAD,
    include_nose_seed: bool = DEFAULT_INCLUDE_NOSE_SEED,
    exclusion_mask: np.ndarray | None = None,
    erode_pixels: int = 2,
) -> np.ndarray:
    import cv2

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    points = np.asarray(points280, dtype=np.float32)
    for _name, indices in ROI_POLYGON_INDICES:
        polygon = _valid_polygon_from_indices(frame_shape, points, indices)
        if polygon is not None:
            cv2.fillPoly(mask, [polygon], 255)
    if include_forehead:
        polygon = forehead_polygon_from_sdk81(frame_shape, sdk_points81)
        if polygon is not None:
            cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32).reshape((-1, 1, 2)), 255)
    if include_nose_seed:
        nose_mask = nose_seed_mask_from_landmarks(frame_shape, points)
        mask[nose_mask] = 255
    if erode_pixels > 0 and np.any(mask):
        kernel = np.ones((erode_pixels * 2 + 1, erode_pixels * 2 + 1), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    result = mask > 10
    if exclusion_mask is not None:
        result &= ~np.asarray(exclusion_mask, dtype=bool)
    return result


def filter_connected_skin_components(
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    *,
    max_seed_distance: int = 36,
    min_component_area: int = 64,
) -> np.ndarray:
    import cv2

    candidate = np.asarray(candidate_mask, dtype=bool)
    seed = np.asarray(seed_mask, dtype=bool)
    if candidate.shape != seed.shape:
        raise ValueError("candidate_mask and seed_mask must have the same shape.")
    if not np.any(candidate):
        return np.zeros(candidate.shape, dtype=bool)
    if not np.any(seed):
        return candidate.copy()

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    distance_to_seed = cv2.distanceTransform((~seed).astype(np.uint8), cv2.DIST_L2, 3)
    result = np.zeros(candidate.shape, dtype=bool)
    for label in range(1, labels_count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        if np.any(component & seed):
            result |= component
            continue
        if float(np.min(distance_to_seed[component])) <= float(max_seed_distance):
            result |= component
    return result


def build_masks_for_roi_method(
    frame_shape: tuple[int, ...],
    points280: np.ndarray,
    sdk_points81: np.ndarray | None,
    *,
    method: str,
    face_mask_mode: str,
    include_forehead: bool = DEFAULT_INCLUDE_FOREHEAD,
    exclude_eyes_brows: bool = DEFAULT_EXCLUDE_EYES_BROWS,
    exclude_mouth: bool = False,
    include_nose_seed: bool = DEFAULT_INCLUDE_NOSE_SEED,
    exclusion_mask: np.ndarray | None = None,
    base_face_mask_value: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if method not in ROI_METHODS:
        raise ValueError(f"Unknown ROI method: {method}")

    face_mask = (
        base_face_mask(frame_shape, points280, sdk_points81, face_mask_mode=face_mask_mode)
        if base_face_mask_value is None
        else np.asarray(base_face_mask_value, dtype=bool).copy()
    )
    exclusion = exclusion_mask
    if exclusion is None and exclude_mouth:
        exclusion = combined_semantic_exclusion_mask(
            frame_shape,
            points280,
            sdk_points81,
            exclude_eyes_brows=exclude_eyes_brows,
            exclude_mouth=True,
        )
    elif exclusion is None and exclude_eyes_brows:
        exclusion = semantic_exclusion_mask_from_landmarks(frame_shape, points280)
    if exclusion is not None:
        face_mask &= ~exclusion
    seed_mask = seed_mask_from_roi_polygons(
        frame_shape,
        points280,
        sdk_points81=sdk_points81,
        include_forehead=(include_forehead and method == "a-fixed-forehead"),
        include_nose_seed=include_nose_seed,
        exclusion_mask=exclusion,
    )

    if include_forehead and method == "a-fixed-forehead":
        face_mask = add_forehead_search_to_mask(face_mask, frame_shape, sdk_points81, width_scale=1.35, height_scale=0.65)
    elif include_forehead and method in {"b-adaptive-forehead", "c-connected-components"}:
        face_mask = add_forehead_search_to_mask(face_mask, frame_shape, sdk_points81, width_scale=1.95, height_scale=1.15)
    if exclusion is not None:
        face_mask &= ~exclusion

    return face_mask, seed_mask & face_mask


def build_masks_for_frame(
    frame_shape: tuple[int, ...],
    points280: np.ndarray,
    sdk_points81: np.ndarray | None,
    *,
    face_mask_mode: str,
    include_forehead: bool = DEFAULT_INCLUDE_FOREHEAD,
    exclude_eyes_brows: bool = DEFAULT_EXCLUDE_EYES_BROWS,
    include_nose_seed: bool = DEFAULT_INCLUDE_NOSE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    return build_masks_for_roi_method(
        frame_shape,
        points280,
        sdk_points81,
        method="a-fixed-forehead",
        face_mask_mode=face_mask_mode,
        include_forehead=include_forehead,
        exclude_eyes_brows=exclude_eyes_brows,
        include_nose_seed=include_nose_seed,
    )

