# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gait_batch.catalog import expand_catalog_jobs
from gait_batch.config_loader import load_catalog
from gait_batch.generate import run_batch_generation
from gait_batch.paths import CONFIGS_DIR, PACKAGE_ROOT, resolve_path
from gait_batch.qc.runner import run_qc_on_tree, summarize_results


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog",
        type=str,
        default=str(CONFIGS_DIR / "catalog.yaml"),
        help="Catalog YAML path (default: configs/catalog.yaml)",
    )


def cmd_list_jobs(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    pending = catalog.get("output", {}).get("pending_dir", "outputs/pending")
    tasks = expand_catalog_jobs(catalog, output_root=pending, job_filter=args.gait)
    print(f"Total tasks: {len(tasks)}")
    for task in tasks:
        print(
            f"{task.task_id}\tseed={task.seed}\tframes={task.num_frames}\t"
            f"speed={task.target_speed_mps}\tdist={task.root_distance_m}\t{task.prompt}"
        )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    summary = run_batch_generation(
        args.catalog,
        device=args.device,
        output_root=args.output,
        job_filter=args.gait,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(
        f"Done: total={summary.total}, generated={summary.generated}, "
        f"skipped={summary.skipped}, failed={summary.failed}"
    )
    return 0


def cmd_qc(args: argparse.Namespace) -> int:
    input_root = resolve_path(args.input)
    approved_root = resolve_path(args.approved) if args.approved else None
    rejected_root = resolve_path(args.rejected) if args.rejected else None

    results = run_qc_on_tree(
        input_root,
        qc_config_path=args.qc_config,
        model_name=args.model,
        approved_root=approved_root,
        rejected_root=rejected_root,
        move_on_pass=args.move_on_pass,
    )
    stats = summarize_results(results)
    print(json.dumps(stats, indent=2))

    report_path = input_root / "qc_report.json"
    report_path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"QC report written to {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gait-batch",
        description="Kimodo 步态库批量生成与 QC 工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-jobs", help="展开 catalog 并列出全部生成任务")
    _add_common_args(p_list)
    p_list.add_argument("--gait", type=str, default=None, help="仅列出指定 gait id")
    p_list.set_defaults(func=cmd_list_jobs)

    p_gen = sub.add_parser("generate", help="按 catalog 批量生成动作")
    _add_common_args(p_gen)
    p_gen.add_argument("--device", type=str, default=None, help="cuda:0 / cpu")
    p_gen.add_argument("--output", type=str, default=None, help="覆盖默认 pending 输出目录")
    p_gen.add_argument("--gait", type=str, default=None, help="仅生成指定 gait")
    p_gen.add_argument("--overwrite", action="store_true", help="覆盖已存在输出")
    p_gen.add_argument("--dry-run", action="store_true", help="只展开任务，不调用模型")
    p_gen.add_argument("--limit", type=int, default=None, help="最多生成前 N 条（调试用）")
    p_gen.add_argument("--batch-size", type=int, default=None, help="覆盖 catalog 中的 batch_size")
    p_gen.set_defaults(func=cmd_generate)

    p_qc = sub.add_parser("qc", help="对 pending 输出运行质量检查")
    p_qc.add_argument("--input", type=str, default="outputs/pending", help="待检目录")
    p_qc.add_argument("--qc-config", type=str, default=str(CONFIGS_DIR / "qc.yaml"))
    p_qc.add_argument("--model", type=str, default="kimodo-soma-rp-v1.1")
    p_qc.add_argument("--approved", type=str, default=None, help="通过后移动目标目录")
    p_qc.add_argument("--rejected", type=str, default=None, help="拒绝后移动目标目录")
    p_qc.add_argument("--move-on-pass", action="store_true", help="按 QC 结果移动目录")
    p_qc.set_defaults(func=cmd_qc)

    return parser


def main(argv: list[str] | None = None) -> int:
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
