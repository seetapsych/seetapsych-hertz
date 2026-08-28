"""ROI names, groups, and selector helpers used by the delivery estimator."""

from __future__ import annotations

from typing import Iterable

from .draw_roi_visual_variants import ROI_POLYGON_INDICES


POLYGON_ROI_NAMES = tuple(name for name, _indices in ROI_POLYGON_INDICES)
SDK_FACEAREA_ROI_NAME = "sdk"
SKIN_ROI_METHODS = ("legacy-ycrcb", "a-fixed-forehead", "b-adaptive-forehead", "c-connected-components")
SKIN_ROI_NAME_BY_METHOD = {
    "legacy-ycrcb": "skin_legacy",
    "a-fixed-forehead": "skin_a_fixed_forehead",
    "b-adaptive-forehead": "skin_b_adaptive_forehead",
    "c-connected-components": "skin_c_connected_components",
}
SKIN_ROI_METHOD_BY_NAME = {name: method for method, name in SKIN_ROI_NAME_BY_METHOD.items()}
SKIN_ROI_NAMES = tuple(SKIN_ROI_NAME_BY_METHOD.values())
ALL_ROI_NAMES = (*POLYGON_ROI_NAMES, SDK_FACEAREA_ROI_NAME, *SKIN_ROI_NAMES)
ROI_GROUPS = {
    "polygon": tuple(POLYGON_ROI_NAMES),
    "skin_all": SKIN_ROI_NAMES,
    "all": ALL_ROI_NAMES,
}
ROI_REGION_CHOICES = tuple(
    dict.fromkeys(
        [
            *ALL_ROI_NAMES,
            *ROI_GROUPS.keys(),
            *[f"roi_{name}" for name in ALL_ROI_NAMES],
        ]
    )
)
N_LANDMARKS = 280
FFT_FUSION_MODES = ("both", "bvp-zscore", "bandpower-normalized")


def normalize_skin_roi_methods(methods: Iterable[str] | None) -> tuple[str, ...]:
    if methods is None:
        return ()
    normalized: list[str] = []
    for method in methods:
        method = str(method).strip()
        if not method:
            continue
        if method == "all":
            for item in SKIN_ROI_METHODS:
                if item not in normalized:
                    normalized.append(item)
            continue
        if method not in SKIN_ROI_METHODS:
            raise ValueError(f"Unknown skin ROI method: {method}")
        if method not in normalized:
            normalized.append(method)
    return tuple(normalized)


def canonical_roi_region_name(region: str) -> str:
    region = str(region).strip()
    if region.startswith("roi_"):
        region = region[len("roi_") :]
    return region


def roi_names_for_regions(
    roi_regions: Iterable[str] | None = None,
) -> list[str]:
    if roi_regions is None:
        return list(ALL_ROI_NAMES)

    names: list[str] = []
    for region in roi_regions:
        region = canonical_roi_region_name(region)
        if not region:
            continue
        if region in ROI_GROUPS:
            expanded = ROI_GROUPS[region]
        elif region in ALL_ROI_NAMES:
            expanded = (region,)
        else:
            raise ValueError(f"Unknown ROI region: {region}")
        for name in expanded:
            if name not in names:
                names.append(name)
    return names


def skin_roi_methods_for_roi_names(roi_names: Iterable[str]) -> tuple[str, ...]:
    methods: list[str] = []
    for name in roi_names:
        method = SKIN_ROI_METHOD_BY_NAME.get(str(name))
        if method is not None and method not in methods:
            methods.append(method)
    return tuple(methods)
