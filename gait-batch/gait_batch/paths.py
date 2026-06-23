# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PACKAGE_ROOT / "configs"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a path relative to package root unless already absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or PACKAGE_ROOT) / p
