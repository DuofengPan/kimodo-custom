# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from kimodo.metrics.foot_skate import FootSkateFromHeight


def horizontal_root_displacement_m(root_positions: np.ndarray) -> float:
    """XZ-plane displacement between first and last root frame."""
    if root_positions.shape[0] < 2:
        return 0.0
    start = root_positions[0, [0, 2]]
    end = root_positions[-1, [0, 2]]
    return float(np.linalg.norm(end - start))


def measured_root_speed_mps(root_positions: np.ndarray, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return horizontal_root_displacement_m(root_positions) / duration_sec


def compute_motion_metrics(
    motion: dict[str, Any],
    *,
    skeleton,
    fps: float,
    duration_sec: float,
    target_speed_mps: float | None,
    use_root2d: bool,
) -> dict[str, float]:
    root_positions = np.asarray(motion["root_positions"])
    posed_joints = torch.from_numpy(np.asarray(motion["posed_joints"])).float().unsqueeze(0)
    lengths = torch.tensor([root_positions.shape[0]], dtype=torch.long)

    skate_metric = FootSkateFromHeight(skeleton, fps)
    skate_out = skate_metric(posed_joints=posed_joints, lengths=lengths)

    displacement = horizontal_root_displacement_m(root_positions)
    speed = measured_root_speed_mps(root_positions, duration_sec)

    metrics = {
        "root_displacement_m": displacement,
        "measured_root_speed_mps": speed,
        "foot_skate_from_height_mps": float(skate_out["foot_skate_from_height"][0].item()),
        "num_frames": float(root_positions.shape[0]),
    }
    if target_speed_mps is not None and use_root2d:
        metrics["target_root_speed_mps"] = float(target_speed_mps)
        metrics["speed_error_ratio"] = abs(speed - target_speed_mps) / max(target_speed_mps, 1e-6)
    return metrics
