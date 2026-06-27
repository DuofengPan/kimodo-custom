# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from gait_batch.config_loader import load_catalog


@dataclass(frozen=True)
class GenerationTask:
    """A single clip to generate."""

    task_id: str
    gait: str
    speed_band: str
    variant_index: int
    prompt: str | list[str]  # 支持 multi-prompt
    duration_sec: float | list[float]
    num_frames: int | list[int]
    seed: int
    root_distance_m: float | None
    target_speed_mps: float | None
    motion_spec: dict[str, Any]
    output_dir: str
    is_multi_prompt: bool = False

    def to_meta(self, **extra: Any) -> dict[str, Any]:
        data = asdict(self)
        data.update(extra)
        return data


def _stable_seed(seed_base: int, task_id: str) -> int:
    digest = hashlib.sha256(task_id.encode()).hexdigest()
    return seed_base + int(digest[:8], 16) % 1_000_000


def expand_catalog_jobs(
    catalog: dict[str, Any],
    *,
    output_root: str,
    job_filter: str | None = None,
) -> list[GenerationTask]:
    """Expand catalog jobs into flat generation tasks."""
    gen_defaults = catalog.get("generation", {})
    fps = float(gen_defaults.get("fps", 30.0))
    duration_default = float(gen_defaults.get("duration_sec", 6.0))
    seed_base = int(gen_defaults.get("seed_base", 1000))

    speed_bands: dict[str, Any] = catalog.get("speed_bands", {})
    gait_types: dict[str, Any] = catalog.get("gait_types", {})
    jobs: list[dict[str, Any]] = catalog.get("jobs", [])

    tasks: list[GenerationTask] = []
    for job in jobs:
        gait_id = job["gait"]
        if job_filter and gait_id != job_filter:
            continue
        if gait_id not in gait_types:
            raise KeyError(f"Unknown gait '{gait_id}' in catalog jobs")

        gait_def = gait_types[gait_id]
        motion_spec = dict(gait_def.get("motion", {}))
        prompts = gait_def.get("prompts", {})
        job_duration = float(job.get("duration_sec", duration_default))
        num_variants = int(job.get("num_variants", 1))
        bands = job.get("speed_bands", list(prompts.keys()))

        for band in bands:
            if band not in prompts:
                raise KeyError(f"Gait '{gait_id}' has no prompt for speed band '{band}'")
            band_cfg = speed_bands.get(band, {})
            target_speed = band_cfg.get("root_speed_mps")
            root_distance = None
            if target_speed is not None and motion_spec.get("use_root2d", False):
                root_distance = float(target_speed) * job_duration

            # 处理 multi-prompt
            prompt_data = prompts[band]
            is_multi = isinstance(prompt_data, list)
            
            if is_multi:
                # multi-prompt: duration 来自 motion_spec 或默认
                durations = motion_spec.get("durations", [job_duration])
                num_frames_list = [max(1, int(round(d * fps))) for d in durations]
            else:
                durations = job_duration
                num_frames_list = max(1, int(round(job_duration * fps)))

            for variant_idx in range(num_variants):
                task_id = f"{gait_id}/{band}/v{variant_idx:03d}"
                seed = _stable_seed(seed_base, task_id)
                rel_out = f"{output_root}/{gait_id}/{band}/v{variant_idx:03d}"

                tasks.append(
                    GenerationTask(
                        task_id=task_id,
                        gait=gait_id,
                        speed_band=band,
                        variant_index=variant_idx,
                        prompt=prompt_data,
                        duration_sec=durations,
                        num_frames=num_frames_list,
                        seed=seed,
                        root_distance_m=root_distance,
                        target_speed_mps=float(target_speed) if target_speed is not None else None,
                        motion_spec=motion_spec,
                        output_dir=rel_out,
                        is_multi_prompt=is_multi,
                    )
                )
    return tasks


def iter_tasks(
    catalog_path: str,
    *,
    output_root: str | None = None,
    job_filter: str | None = None,
) -> Iterator[GenerationTask]:
    catalog = load_catalog(catalog_path)
    pending = output_root or catalog.get("output", {}).get("pending_dir", "outputs/pending")
    return iter(expand_catalog_jobs(catalog, output_root=pending, job_filter=job_filter))
