# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from gait_batch.paths import CONFIGS_DIR, resolve_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(str(path))
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    merged = OmegaConf.create({})
    for cfg in configs:
        merged = OmegaConf.merge(merged, OmegaConf.create(cfg))
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def load_catalog(catalog_path: str | Path) -> dict[str, Any]:
    """Load catalog and merge included prompt/defaults files."""
    catalog_path = resolve_path(catalog_path)
    catalog = load_yaml(catalog_path)
    catalog_dir = catalog_path.parent

    defaults_ref = catalog.pop("defaults", None)
    defaults: dict[str, Any] = {}
    if defaults_ref:
        defaults = load_yaml(resolve_path(defaults_ref, base=catalog_dir))

    includes = catalog.pop("includes", [])
    prompt_data: dict[str, Any] = {"speed_bands": {}, "gait_types": {}}
    for include in includes:
        inc = load_yaml(resolve_path(include, base=catalog_dir))
        if "speed_bands" in inc:
            prompt_data["speed_bands"].update(inc["speed_bands"])
        if "gait_types" in inc:
            prompt_data["gait_types"].update(inc["gait_types"])

    return merge_configs(defaults, prompt_data, catalog)


def load_qc_config(qc_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_path(qc_path or CONFIGS_DIR / "qc.yaml")
    return load_yaml(path)
