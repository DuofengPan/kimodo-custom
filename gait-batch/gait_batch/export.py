# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kimodo.exports.bvh import save_motion_bvh
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.skeleton import global_rots_to_local_rots

from gait_batch.catalog import GenerationTask


def save_task_outputs(
    task: GenerationTask,
    output: dict[str, Any],
    out_dir: Path,
    *,
    skeleton,
    fps: float,
    save_npz: bool,
    save_bvh: bool,
    bvh_standard_tpose: bool,
    model_name: str,
    device: torch.device,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save NPZ, BVH, constraints, and meta.json for one task."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    single = {
        k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v)
        for k, v in output.items()
    }

    if save_npz:
        npz_path = out_dir / "motion.npz"
        save_kimodo_npz(str(npz_path), single)
        written["npz"] = npz_path

    if save_bvh:
        bvh_path = out_dir / "motion.bvh"
        joints_pos = torch.from_numpy(np.asarray(single["posed_joints"])).to(device)
        joints_rot = torch.from_numpy(np.asarray(single["global_rot_mats"])).to(device)
        local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
        root_positions = joints_pos[:, skeleton.root_idx, :]
        save_motion_bvh(
            bvh_path,
            local_rot_mats,
            root_positions,
            skeleton=skeleton,
            fps=fps,
            standard_tpose=bvh_standard_tpose,
        )
        written["bvh"] = bvh_path

    meta = task.to_meta(
        model=model_name,
        fps=fps,
        export={"save_npz": save_npz, "save_bvh": save_bvh, "bvh_standard_tpose": bvh_standard_tpose},
        qc={"status": "pending"},
    )
    if extra_meta:
        meta.update(extra_meta)

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written["meta"] = meta_path
    return written


def save_constraints_json(constraints: list[dict[str, Any]], out_dir: Path) -> Path | None:
    if not constraints:
        return None
    path = out_dir / "constraints.json"
    path.write_text(json.dumps(constraints, indent=2) + "\n", encoding="utf-8")
    return path


def task_is_complete(out_dir: Path, *, require_bvh: bool) -> bool:
    if not (out_dir / "motion.npz").is_file():
        return False
    if require_bvh and not (out_dir / "motion.bvh").is_file():
        return False
    return (out_dir / "meta.json").is_file()


def move_task_dir(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    return dst
