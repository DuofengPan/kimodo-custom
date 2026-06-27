# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from kimodo.constraints import load_constraints_lst
from kimodo.model import load_model
from kimodo.tools import seed_everything

from gait_batch.catalog import GenerationTask, expand_catalog_jobs
from gait_batch.config_loader import load_catalog
from gait_batch.constraints import build_root2d_json
from gait_batch.export import save_constraints_json, save_task_outputs, task_is_complete
from gait_batch.paths import resolve_path


@dataclass
class GenerationSummary:
    total: int
    generated: int
    skipped: int
    failed: int


def _slice_output_at(output: dict[str, Any], index: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in output.items():
        if isinstance(v, dict):
            out[k] = _slice_output_at(v, index)
        elif hasattr(v, "shape") and len(v.shape) > 0:
            out[k] = v[index]
        else:
            out[k] = v
    return out


def _crop_output(output: dict[str, Any], num_frames: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in output.items():
        if isinstance(v, dict):
            out[k] = _crop_output(v, num_frames)
        elif hasattr(v, "shape") and len(getattr(v, "shape", ())) >= 1:
            out[k] = v[:num_frames]
        else:
            out[k] = v
    return out


def _build_cfg_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg_type = cfg.get("type", "separated")
    if cfg_type == "nocfg":
        return {"cfg_type": "nocfg"}
    if cfg_type == "regular":
        weight = float(cfg.get("text_weight", 2.0))
        return {"cfg_type": "regular", "cfg_weight": weight}
    return {
        "cfg_type": "separated",
        "cfg_weight": [float(cfg.get("text_weight", 2.0)), float(cfg.get("constraint_weight", 2.0))],
    }


def run_batch_generation(
    catalog_path: str | Path,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
    job_filter: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    batch_size: int | None = None,
) -> GenerationSummary:
    catalog = load_catalog(catalog_path)
    gen_cfg = dict(catalog.get("generation", {}))
    if batch_size is not None:
        gen_cfg["batch_size"] = batch_size
    export_cfg = catalog.get("export", {})
    output_cfg = catalog.get("output", {})

    pending_rel = output_root or output_cfg.get("pending_dir", "outputs/pending")
    pending_root = resolve_path(pending_rel)

    tasks = expand_catalog_jobs(catalog, output_root=str(pending_rel), job_filter=job_filter)
    if limit is not None:
        tasks = tasks[:limit]

    if dry_run:
        print(f"[dry-run] Would generate {len(tasks)} tasks under {pending_root}")
        for task in tasks[:10]:
            print(f"  - {task.task_id}: seed={task.seed}, frames={task.num_frames}, prompt={task.prompt!r}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return GenerationSummary(total=len(tasks), generated=0, skipped=0, failed=0)

    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model, model_name = load_model(
        gen_cfg.get("model", "kimodo-soma-rp-v1.1"),
        device=dev,
        return_resolved_name=True,
    )
    skeleton = model.output_skeleton
    fps = float(model.fps)
    batch_size = int(gen_cfg.get("batch_size", 4))
    cfg_kwargs = _build_cfg_kwargs(gen_cfg.get("cfg", {}))
    save_npz = bool(export_cfg.get("save_npz", True))
    save_bvh = bool(export_cfg.get("save_bvh", True))
    bvh_standard_tpose = bool(export_cfg.get("bvh_standard_tpose", False))
    device_t = torch.device(dev)

    pending: list[GenerationTask] = []
    skipped = 0
    for task in tasks:
        out_dir = resolve_path(task.output_dir)
        if not overwrite and task_is_complete(out_dir, require_bvh=save_bvh):
            skipped += 1
            continue
        pending.append(task)

    generated = 0
    failed = 0

    for batch_start in tqdm(range(0, len(pending), batch_size), desc="Generating batches"):
        batch = pending[batch_start : batch_start + batch_size]
        
        # 处理 multi-prompt 和普通情况
        # batch 中所有任务必须是同类型（不能混合 multi/single），或者分别处理
        # 简单处理：每个 batch 单独调用 model
        
        for task in batch:
            out_dir = resolve_path(task.output_dir)
            
            # 处理 constraints
            cjson = None
            if task.is_multi_prompt:
                # multi-prompt: 不使用 root2d constraints
                cjson = None
                batch_constraints_lst = []
            else:
                cjson = build_root2d_json(
                    task.motion_spec,
                    num_frames=task.num_frames if not task.is_multi_prompt else 0,
                    distance_m=task.root_distance_m,
                )
                if cjson:
                    batch_constraints_lst = [load_constraints_lst(cjson, model.skeleton, device=device_t)]
                else:
                    batch_constraints_lst = []

            seed_everything(task.seed)
            
            try:
                output = model(
                    task.prompt if task.is_multi_prompt else [task.prompt],
                    task.num_frames if task.is_multi_prompt else [task.num_frames],
                    constraint_lst=batch_constraints_lst,
                    num_denoising_steps=int(gen_cfg.get("diffusion_steps", 100)),
                    multi_prompt=task.is_multi_prompt,
                    num_transition_frames=int(gen_cfg.get("num_transition_frames", 5)),
                    num_samples=1,  # 单个样本生成
                    post_processing=bool(gen_cfg.get("post_processing", True)),
                    return_numpy=True,
                    **cfg_kwargs,
                )
            except Exception:
                failed += 1
                raise

            # multi-prompt 输出不需要 crop，但 single 需要
            if task.is_multi_prompt:
                sample = _slice_output_at(output, 0)
            else:
                sample = _crop_output(_slice_output_at(output, 0), task.num_frames)
                
            if cjson:
                save_constraints_json(cjson, out_dir)
            save_task_outputs(
                task,
                sample,
                out_dir,
                skeleton=skeleton,
                fps=fps,
                save_npz=save_npz,
                save_bvh=save_bvh,
                bvh_standard_tpose=bvh_standard_tpose,
                model_name=model_name,
                device=device_t,
            )
            generated += 1

    return GenerationSummary(
        total=len(tasks),
        generated=generated,
        skipped=skipped,
        failed=failed,
    )
