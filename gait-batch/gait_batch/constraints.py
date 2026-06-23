# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_direction(direction: list[float]) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError(f"Invalid zero direction vector: {direction}")
    return vec / norm


def build_root2d_json(
    motion_spec: dict[str, Any],
    *,
    num_frames: int,
    distance_m: float | None,
) -> list[dict[str, Any]] | None:
    """Build constraints.json-compatible root2d entries from motion spec.

    Returns None when the gait does not use root2d constraints.
    """
    if not motion_spec.get("use_root2d", False):
        return None

    path_type = motion_spec.get("type", "linear")
    density = motion_spec.get("density", "sparse")

    if path_type == "in_place":
        return None

    if path_type != "linear":
        raise ValueError(f"Unsupported motion path type: {path_type}")

    direction = _normalize_direction(motion_spec.get("direction", [0.0, 1.0]))
    end_xz = (direction * float(distance_m or 0.0)).tolist()
    start_xz = [0.0, 0.0]

    if density == "dense":
        frame_indices = list(range(num_frames))
        alphas = np.linspace(0.0, 1.0, num_frames)
        smooth_root_2d = [(alpha * end_xz[0], alpha * end_xz[1]) for alpha in alphas]
    elif density == "sparse":
        frame_indices = [0, max(0, num_frames - 1)]
        smooth_root_2d = [start_xz, end_xz]
    else:
        raise ValueError(f"Unknown root2d density: {density}")

    return [
        {
            "type": "root2d",
            "frame_indices": frame_indices,
            "smooth_root_2d": smooth_root_2d,
        }
    ]
