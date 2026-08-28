"""Result serialization helpers for delivery estimators."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from .fft_fusion import fft_fusion_output_prefix
from .output_schema import per_frame_fieldnames


def _safe_key(name: str) -> str:
    return str(name).replace("-", "_").replace("/", "_")


def _pad_1d(sequences: Iterable[np.ndarray], *, dtype=float, fill_value=np.nan) -> np.ndarray:
    """Pack variable-length per-frame curves into a rectangular NPZ array."""

    items = [np.asarray(item, dtype=dtype).reshape(-1) for item in sequences]
    width = max((item.size for item in items), default=0)
    output = np.full((len(items), width), fill_value, dtype=dtype)
    for index, item in enumerate(items):
        if item.size:
            output[index, : item.size] = item
    return output


def _fft_curve_values(frame_results: Iterable[object], roi_name: str, attribute: str) -> list[np.ndarray]:
    """Collect one FFT curve attribute, using an empty array when unavailable."""

    values: list[np.ndarray] = []
    for result in frame_results:
        curve = result.fft_curves.get(roi_name)
        values.append(np.empty(0) if curve is None else getattr(curve, attribute))
    return values


def write_compatible_rows_csv(
    output_csv: Path,
    rows: Iterable[dict[str, object]],
    *,
    roi_names: Iterable[str],
    enable_fft_fusion: bool,
    fft_fusion_mode: str,
) -> None:
    """Write rows with the legacy-compatible per-frame CSV schema."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=per_frame_fieldnames(
                enable_fft_fusion=enable_fft_fusion,
                fft_fusion_mode=fft_fusion_mode,
                roi_names=roi_names,
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_compatible_frame_results_csv(
    output_csv: Path,
    frame_results: Iterable[object],
    *,
    roi_names: Iterable[str],
    enable_fft_fusion: bool,
    fft_fusion_mode: str,
) -> None:
    """Write detailed frame results through their compatible CSV row payload."""

    write_compatible_rows_csv(
        output_csv,
        (result.hr_results for result in frame_results),
        roi_names=roi_names,
        enable_fft_fusion=enable_fft_fusion,
        fft_fusion_mode=fft_fusion_mode,
    )


def write_detail_npz(output_npz: Path, frame_results: list[object], roi_names: Iterable[str]) -> None:
    """Write masks, BGR traces, BVP curves, FFT curves, and fused values."""

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "frame_index": np.asarray([result.frame_index for result in frame_results], dtype=np.int64),
        "frame_time_seconds": np.asarray([result.frame_time_seconds for result in frame_results], dtype=float),
        "roi_names": np.asarray(list(roi_names), dtype="U64"),
        "bvp_value_fusion": np.asarray([result.bvp_value_fusion for result in frame_results], dtype=float),
        "hr_value_fusion": np.asarray([result.hr_value_fusion for result in frame_results], dtype=float),
    }
    for name in roi_names:
        key = _safe_key(name)
        payload[f"roi_{key}_mask"] = np.stack(
            [np.asarray(result.roi_masks.get(name, np.zeros((0, 0), dtype=bool)), dtype=np.uint8) for result in frame_results],
            axis=0,
        )
        payload[f"roi_{key}_bgr"] = np.vstack(
            [np.asarray(result.roi_bgr.get(name, np.full(3, np.nan)), dtype=float) for result in frame_results]
        )
        payload[f"roi_{key}_pixel_count"] = np.asarray(
            [result.roi_pixel_counts.get(name, 0) for result in frame_results],
            dtype=np.int64,
        )
        payload[f"roi_{key}_area_ratio"] = np.asarray(
            [result.roi_area_ratios.get(name, np.nan) for result in frame_results],
            dtype=float,
        )
        payload[f"roi_{key}_bvp_curve"] = _pad_1d([result.bvp_curves.get(name, np.empty(0)) for result in frame_results])
        payload[f"roi_{key}_bvp_value"] = np.asarray(
            [result.bvp_values.get(name, np.nan) for result in frame_results],
            dtype=float,
        )
        payload[f"roi_{key}_fft_bpm"] = _pad_1d(
            _fft_curve_values(frame_results, name, "bpm")
        )
        payload[f"roi_{key}_fft_power"] = _pad_1d(
            _fft_curve_values(frame_results, name, "power")
        )
        payload[f"roi_{key}_hr_bpm"] = np.asarray(
            [float(result.hr_results.get(f"roi_{name}_hr_bpm", 0.0) or 0.0) for result in frame_results],
            dtype=float,
        )
    payload["bvp_curve_fusion"] = _pad_1d([result.bvp_curve_fusion for result in frame_results])
    for mode in ("bvp-zscore", "bandpower-normalized"):
        prefix = fft_fusion_output_prefix(mode)
        payload[f"{prefix}_hr_bpm"] = np.asarray(
            [float(result.hr_results.get(f"{prefix}_hr_bpm", 0.0) or 0.0) for result in frame_results],
            dtype=float,
        )
        payload[f"{prefix}_roi_count"] = np.asarray(
            [int(result.hr_results.get(f"{prefix}_roi_count", 0) or 0) for result in frame_results],
            dtype=np.int64,
        )
    np.savez_compressed(output_npz, **payload)
