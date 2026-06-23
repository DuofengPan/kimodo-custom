# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kimodo.exports.motion_io import load_kimodo_npz
from kimodo.model import load_model
from kimodo.skeleton import SOMASkeleton77

from gait_batch.config_loader import load_qc_config
from gait_batch.export import move_task_dir
from gait_batch.paths import resolve_path
from gait_batch.qc.metrics import compute_motion_metrics


@dataclass
class QCResult:
    task_dir: Path
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_dir": str(self.task_dir),
            "passed": self.passed,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


def _load_meta(task_dir: Path) -> dict[str, Any]:
    meta_path = task_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json in {task_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def evaluate_task_dir(
    task_dir: Path,
    qc_cfg: dict[str, Any],
    *,
    skeleton: SOMASkeleton77,
    fps: float,
) -> QCResult:
    reasons: list[str] = []
    meta = _load_meta(task_dir)
    motion_path = task_dir / "motion.npz"
    if not motion_path.is_file():
        return QCResult(task_dir, False, ["missing motion.npz"])

    motion = load_kimodo_npz(str(motion_path))
    duration_sec = float(meta.get("duration_sec", motion["root_positions"].shape[0] / fps))
    motion_spec = meta.get("motion_spec", {})
    use_root2d = bool(motion_spec.get("use_root2d", False))
    target_speed = meta.get("target_speed_mps")

    metrics = compute_motion_metrics(
        motion,
        skeleton=skeleton,
        fps=fps,
        duration_sec=duration_sec,
        target_speed_mps=float(target_speed) if target_speed is not None else None,
        use_root2d=use_root2d,
    )

    general = qc_cfg.get("general", {})
    min_frames = int(general.get("min_frames", 30))
    if metrics["num_frames"] < min_frames:
        reasons.append(f"too_few_frames ({metrics['num_frames']:.0f} < {min_frames})")

    loco = qc_cfg.get("locomotion", {})
    if use_root2d:
        min_disp = float(loco.get("min_root_displacement_m", 0.5))
        if metrics["root_displacement_m"] < min_disp:
            reasons.append(
                f"insufficient_root_displacement ({metrics['root_displacement_m']:.3f}m < {min_disp}m)"
            )
        if target_speed is not None:
            tol = float(loco.get("speed_tolerance_ratio", 0.45))
            err = metrics.get("speed_error_ratio", 0.0)
            if err > tol:
                reasons.append(f"speed_out_of_tolerance (error_ratio={err:.2f} > {tol})")

    skate_cfg = qc_cfg.get("foot_skate", {})
    max_skate = float(skate_cfg.get("max_mean_toe_velocity_mps", 0.35))
    if metrics["foot_skate_from_height_mps"] > max_skate:
        reasons.append(
            f"foot_skate_too_high ({metrics['foot_skate_from_height_mps']:.3f} > {max_skate})"
        )

    passed = len(reasons) == 0
    return QCResult(task_dir, passed, reasons, metrics)


def run_qc_on_tree(
    input_root: Path,
    *,
    qc_config_path: str | Path | None = None,
    model_name: str = "kimodo-soma-rp-v1.1",
    approved_root: Path | None = None,
    rejected_root: Path | None = None,
    move_on_pass: bool | None = None,
) -> list[QCResult]:
    qc_cfg = load_qc_config(qc_config_path)
    general = qc_cfg.get("general", {})
    should_move = move_on_pass if move_on_pass is not None else bool(general.get("move_on_pass", False))

    model, _ = load_model(model_name, device="cpu", return_resolved_name=True)
    skeleton = model.output_skeleton
    fps = float(model.fps)

    results: list[QCResult] = []
    task_dirs = sorted(p for p in input_root.rglob("*") if (p / "motion.npz").is_file())
    for task_dir in task_dirs:
        result = evaluate_task_dir(task_dir, qc_cfg, skeleton=skeleton, fps=fps)
        results.append(result)

        meta_path = task_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["qc"] = {
            "status": "approved" if result.passed else "rejected",
            "reasons": result.reasons,
            "metrics": result.metrics,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if should_move and approved_root and rejected_root:
            rel = task_dir.relative_to(input_root)
            target_root = approved_root if result.passed else rejected_root
            move_task_dir(task_dir, target_root / rel)

    return results


def summarize_results(results: list[QCResult]) -> dict[str, int]:
    passed = sum(1 for r in results if r.passed)
    return {"total": len(results), "passed": passed, "rejected": len(results) - passed}
