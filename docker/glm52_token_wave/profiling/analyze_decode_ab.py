#!/usr/bin/env python3
"""Summarize steady-state decode steps from the GLM-5.2 profiling A/B.

The script discovers the five trace groups in each profiling directory,
maps them to concurrency 1/2/4/8/16, extracts pure-decode scheduler steps,
and reports wall time plus the MoE path breakdown.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


CONCURRENCIES = (1, 2, 4, 8, 16)
DEFAULT_TRACE_TOOLS = (
    "/home/nholmber/inference-testing/.agents/skills/"
    "trace-analysis/scripts"
)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _moe_category(name: str) -> str | None:
    lower = name.lower()
    if "gemm1_a4w4_port" in lower:
        return "mxmoe_g1"
    if "gemm2_a4w4_port" in lower:
        return "mxmoe_g2"
    if "mfma_moe1" in lower:
        return "mxmoe_g1"
    if "mfma_moe2" in lower:
        return "mxmoe_g2"
    if (
        "mxfp4_moe" in lower
        or "moe_sort_quant" in lower
        or "fused_mx_quant_moe_sort" in lower
        or "moe_sorting_oneshot" in lower
    ):
        return "mxmoe_sort_quant"
    if "mxfp4_" in lower and ("_g1" in lower or "stage1" in lower):
        return "fused_g1"
    if "mxfp4_" in lower and ("_g2" in lower or "stage2" in lower):
        return "fused_g2"
    return None


def _discover(directory: Path, variant: str, ranks: set[int] | None):
    by_rank: dict[int, list[Path]] = defaultdict(list)
    pattern = re.compile(r"rank(\d+)\.(\d+)\.pt\.trace\.json\.gz$")
    for path in directory.glob("dp*_rank*.pt.trace.json.gz"):
        match = pattern.search(path.name)
        if not match:
            continue
        rank = int(match.group(1))
        if ranks is not None and rank not in ranks:
            continue
        by_rank[rank].append(path)

    tasks = []
    for rank, paths in sorted(by_rank.items()):
        paths.sort(key=lambda path: path.stat().st_mtime_ns)
        if len(paths) != len(CONCURRENCIES):
            raise RuntimeError(
                f"expected {len(CONCURRENCIES)} traces for {variant} rank "
                f"{rank}, found {len(paths)}"
            )
        for concurrency, path in zip(CONCURRENCIES, paths, strict=True):
            tasks.append(
                {
                    "variant": variant,
                    "concurrency": concurrency,
                    "rank": rank,
                    "path": str(path),
                }
            )
    return tasks


def _process(task):
    tools = os.environ.get("TRACE_ANALYSIS_SCRIPTS", DEFAULT_TRACE_TOOLS)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from compare_traces import load_trace

    with contextlib.redirect_stdout(io.StringIO()):
        steps, _ = load_trace(task["path"])

    target = [
        step
        for step in steps
        if step["prefill"] == 0
        and step["decode"] == task["concurrency"]
    ]
    if not target:
        raise RuntimeError(
            f"no pure-decode steps for concurrency {task['concurrency']} "
            f"in {task['path']}"
        )

    wall = [step["wall_time_us"] for step in target]
    gpu = [step["gpu_time_us"] for step in target]
    category_total = defaultdict(float)
    category_calls = defaultdict(float)
    category_names: dict[str, set[str]] = defaultdict(set)

    for step in target:
        for name, duration, _role in step["kernels"]:
            category = _moe_category(name)
            if category is None:
                continue
            category_total[category] += duration
            category_calls[category] += 1
            category_names[category].add(name)

    count = len(target)
    categories = {
        category: {
            "avg_us": category_total[category] / count,
            "calls_per_step": category_calls[category] / count,
            "names": sorted(category_names[category]),
        }
        for category in sorted(category_total)
    }
    moe_us = sum(value["avg_us"] for value in categories.values())

    return {
        **task,
        "steps": count,
        "wall_mean_us": statistics.fmean(wall),
        "wall_median_us": statistics.median(wall),
        "wall_p10_us": _percentile(wall, 0.10),
        "wall_p90_us": _percentile(wall, 0.90),
        "gpu_mean_us": statistics.fmean(gpu),
        "moe_us": moe_us,
        "categories": categories,
    }


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["concurrency"])].append(row)

    result = {}
    for key, group in grouped.items():
        categories = defaultdict(list)
        calls = defaultdict(list)
        names: dict[str, set[str]] = defaultdict(set)
        for row in group:
            for category, value in row["categories"].items():
                categories[category].append(value["avg_us"])
                calls[category].append(value["calls_per_step"])
                names[category].update(value["names"])

        result[key] = {
            "ranks": len(group),
            "steps_min": min(row["steps"] for row in group),
            "steps_max": max(row["steps"] for row in group),
            "wall_mean_us": statistics.fmean(
                row["wall_mean_us"] for row in group
            ),
            "wall_rank_min_us": min(row["wall_mean_us"] for row in group),
            "wall_rank_max_us": max(row["wall_mean_us"] for row in group),
            "wall_median_us": statistics.fmean(
                row["wall_median_us"] for row in group
            ),
            "wall_p10_us": statistics.fmean(
                row["wall_p10_us"] for row in group
            ),
            "wall_p90_us": statistics.fmean(
                row["wall_p90_us"] for row in group
            ),
            "gpu_mean_us": statistics.fmean(
                row["gpu_mean_us"] for row in group
            ),
            "moe_us": statistics.fmean(row["moe_us"] for row in group),
            "categories": {
                category: {
                    "avg_us": statistics.fmean(values),
                    "calls_per_step": statistics.fmean(calls[category]),
                    "names": sorted(names[category]),
                }
                for category, values in sorted(categories.items())
            },
        }
    return result


def _print_report(summary):
    print(
        "| Conc | MXMOE wall (ms) | Fused wall (ms) | Delta (us) | "
        "Delta (%) | MXMOE MoE (ms) | Fused MoE (ms) |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for concurrency in CONCURRENCIES:
        base = summary[("mxmoe", concurrency)]
        fused = summary[("fused", concurrency)]
        delta = fused["wall_mean_us"] - base["wall_mean_us"]
        percent = delta / base["wall_mean_us"] * 100.0
        print(
            f"| {concurrency} | {base['wall_mean_us']/1000:.3f} | "
            f"{fused['wall_mean_us']/1000:.3f} | {delta:+.1f} | "
            f"{percent:+.2f}% | {base['moe_us']/1000:.3f} | "
            f"{fused['moe_us']/1000:.3f} |"
        )

    print("\nMoE category breakdown, averaged across selected ranks:")
    for concurrency in CONCURRENCIES:
        print(f"\nConcurrency {concurrency}:")
        for variant in ("mxmoe", "fused"):
            row = summary[(variant, concurrency)]
            parts = [
                f"{category}={value['avg_us']:.1f}us/"
                f"{value['calls_per_step']:.1f} calls"
                for category, value in row["categories"].items()
            ]
            print(f"  {variant}: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mxmoe-dir",
        type=Path,
        default=Path("/tmp/traces_glm52_mxmoe_ab_20260730"),
    )
    parser.add_argument(
        "--fused-dir",
        type=Path,
        default=Path("/tmp/traces_glm52_fused_moe_ab_20260730"),
    )
    parser.add_argument(
        "--ranks",
        default="0",
        help='comma-separated ranks or "all" (default: 0)',
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    ranks = None
    if args.ranks != "all":
        ranks = {int(item) for item in args.ranks.split(",")}

    tasks = [
        *_discover(args.mxmoe_dir, "mxmoe", ranks),
        *_discover(args.fused_dir, "fused", ranks),
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_process, tasks))

    summary = _aggregate(rows)
    _print_report(summary)

    if args.json_output:
        payload = {
            "rows": rows,
            "summary": {
                f"{variant}_c{concurrency}": value
                for (variant, concurrency), value in sorted(summary.items())
            },
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nSaved JSON to {args.json_output}")


if __name__ == "__main__":
    main()
