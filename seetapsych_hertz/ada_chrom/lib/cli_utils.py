"""Shared CLI helpers for delivery estimator entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import COMMON_ESTIMATOR_ARG_SPECS, ENTRYPOINT_CLI_CONFIGS, EstimatorConfig


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: True or False.")


def add_common_estimator_args(
    parser: argparse.ArgumentParser,
    *,
    roi_region_choices: tuple[str, ...],
    fft_fusion_modes: tuple[str, ...],
    roi_help: str | None = None,
) -> argparse.ArgumentParser:
    return add_arg_specs(
        parser,
        COMMON_ESTIMATOR_ARG_SPECS,
        roi_region_choices=roi_region_choices,
        fft_fusion_modes=fft_fusion_modes,
        roi_help=roi_help,
    )


def add_arg_specs(
    parser: argparse.ArgumentParser,
    arg_specs: tuple[dict[str, object], ...],
    *,
    roi_region_choices: tuple[str, ...],
    fft_fusion_modes: tuple[str, ...],
    roi_help: str | None = None,
) -> argparse.ArgumentParser:
    value_sources = {
        "roi_region_choices": roi_region_choices,
        "fft_fusion_modes": fft_fusion_modes,
        "roi_help": roi_help,
    }
    type_map = {
        "int": int,
        "bool": parse_bool,
        "path": Path,
    }

    for spec in arg_specs:
        kwargs = {
            key: value
            for key, value in spec.items()
            if key not in {"flags", "type", "choices_from", "help_from"}
        }
        if "type" in spec:
            kwargs["type"] = type_map[spec["type"]]
        if "choices_from" in spec:
            kwargs["choices"] = value_sources[spec["choices_from"]]
        if "help_from" in spec:
            kwargs["help"] = value_sources[spec["help_from"]]
        parser.add_argument(*spec["flags"], **kwargs)
    return parser


def build_delivery_arg_parser(
    entrypoint: str,
    *,
    roi_region_choices: tuple[str, ...],
    fft_fusion_modes: tuple[str, ...],
) -> argparse.ArgumentParser:
    config = ENTRYPOINT_CLI_CONFIGS[entrypoint]
    parser = argparse.ArgumentParser(description=config["description"])
    return add_arg_specs(
        parser,
        config["arg_specs"],
        roi_region_choices=roi_region_choices,
        fft_fusion_modes=fft_fusion_modes,
        roi_help=config.get("roi_help"),
    )


def parsed_args_to_estimator_kwargs(args: argparse.Namespace, entrypoint: str) -> dict[str, object]:
    config = ENTRYPOINT_CLI_CONFIGS[entrypoint]
    estimator_fields = set(EstimatorConfig.__dataclass_fields__)
    kwargs: dict[str, object] = {}
    estimator_values: dict[str, object] = {}
    for parsed_name, estimator_name in config["estimator_kwargs"].items():
        value = getattr(args, parsed_name)
        if estimator_name in estimator_fields:
            estimator_values[estimator_name] = value
        else:
            kwargs[estimator_name] = value
    kwargs["estimator_config"] = EstimatorConfig(**estimator_values)
    return kwargs
