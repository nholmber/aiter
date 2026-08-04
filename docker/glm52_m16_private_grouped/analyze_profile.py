#!/usr/bin/env python3
"""Analyze the matched GLM-5.2 M=16 baseline/private-grouped traces."""

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


DEFAULT_TRACE_TOOLS = (
    "/home/nholmber/inference-testing/.agents/skills/"
    "trace-analysis/scripts"
)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def moe_category(name: str) -> str | None:
    lower = name.lower()
    if "gemm1_a4w4_port" in lower or "mfma_moe1" in lower:
        return "g1"
    if "gemm2_a4w4_port" in lower or "mfma_moe2" in lower:
        return "g2"
    if (
        "mxfp4_moe" in lower
        or "moe_sort_quant" in lower
        or "fused_mx_quant_moe_sort" in lower
        or "moe_sorting_oneshot" in lower
    ):
        return "aux"
    if "mxfp4_routed_compact_shared_g1" in lower:
        return "g1"
    if "mxfp4_routed_compact_shared_g2" in lower:
        return "g2"
    return None


def discover(directory: Path, variant: str, tools: str) -> list[dict]:
    pattern = re.compile(r"_rank(\d+)\.")
    tasks = []
    for path in sorted(directory.glob("dp*_rank*.pt.trace.json.gz")):
        match = pattern.search(path.name)
        if match:
            tasks.append(
                {
                    "variant": variant,
                    "rank": int(match.group(1)),
                    "path": str(path),
                    "tools": tools,
                }
            )
    if not tasks:
        raise RuntimeError(f"no worker traces found in {directory}")
    return tasks


def process(task: dict) -> dict:
    if task["tools"] not in sys.path:
        sys.path.insert(0, task["tools"])
    from compare_traces import load_trace

    with contextlib.redirect_stdout(io.StringIO()):
        steps, _ = load_trace(task["path"])
    target = [
        step
        for step in steps
        if step["prefill"] == 0 and step["decode"] == 16
    ]
    if not target:
        raise RuntimeError(f"no pure decode M=16 steps in {task['path']}")

    wall = [step["wall_time_us"] for step in target]
    gpu = [step["gpu_time_us"] for step in target]
    totals = defaultdict(float)
    calls = defaultdict(int)
    names: dict[str, set[str]] = defaultdict(set)
    for step in target:
        for name, duration, _role in step["kernels"]:
            category = moe_category(name)
            if category is None:
                continue
            totals[category] += duration
            calls[category] += 1
            names[category].add(name)

    count = len(target)
    return {
        **task,
        "steps": count,
        "wall_mean_us": statistics.fmean(wall),
        "wall_median_us": statistics.median(wall),
        "wall_p10_us": percentile(wall, 0.10),
        "wall_p90_us": percentile(wall, 0.90),
        "gpu_mean_us": statistics.fmean(gpu),
        "categories": {
            category: {
                "avg_us_per_step": totals[category] / count,
                "calls_per_step": calls[category] / count,
                "names": sorted(names[category]),
            }
            for category in sorted(totals)
        },
    }


def aggregate(rows: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    result = {}
    for variant, group in grouped.items():
        categories = defaultdict(list)
        calls = defaultdict(list)
        names: dict[str, set[str]] = defaultdict(set)
        for row in group:
            for category, value in row["categories"].items():
                categories[category].append(value["avg_us_per_step"])
                calls[category].append(value["calls_per_step"])
                names[category].update(value["names"])

        result[variant] = {
            "ranks": len(group),
            "steps_min": min(row["steps"] for row in group),
            "steps_max": max(row["steps"] for row in group),
            "wall_mean_us": statistics.fmean(
                row["wall_mean_us"] for row in group
            ),
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
            "categories": {
                category: {
                    "avg_us_per_step": statistics.fmean(values),
                    "calls_per_step": statistics.fmean(calls[category]),
                    "names": sorted(names[category]),
                }
                for category, values in sorted(categories.items())
            },
        }
    return result


def print_report(summary: dict[str, dict]) -> None:
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    print("| Metric | Baseline | Candidate | Delta |")
    print("|:--|--:|--:|--:|")
    for key, label in (
        ("wall_p10_us", "Wall p10 (us)"),
        ("wall_median_us", "Wall median (us)"),
        ("wall_p90_us", "Wall p90 (us)"),
    ):
        delta = candidate[key] - baseline[key]
        print(
            f"| {label} | {baseline[key]:.3f} | {candidate[key]:.3f} | "
            f"{delta:+.3f} ({delta / baseline[key] * 100:+.2f}%) |"
        )

    print("\nMoE time per layer:")
    print("| Component | Baseline (us) | Candidate (us) | Delta (us) |")
    print("|:--|--:|--:|--:|")
    for category in ("g1", "g2", "aux"):
        base = baseline["categories"].get(category, {}).get(
            "avg_us_per_step", 0.0
        ) / 75.0
        cand = candidate["categories"].get(category, {}).get(
            "avg_us_per_step", 0.0
        ) / 75.0
        print(f"| {category} | {base:.3f} | {cand:.3f} | {cand-base:+.3f} |")

    base_total = sum(
        value["avg_us_per_step"]
        for value in baseline["categories"].values()
    )
    cand_total = sum(
        value["avg_us_per_step"]
        for value in candidate["categories"].values()
    )
    delta = cand_total - base_total
    print(
        f"| total | {base_total / 75:.3f} | {cand_total / 75:.3f} | "
        f"{delta / 75:+.3f} |"
    )
    print(f"\nMoE delta per decode step: {delta:+.3f} us")

    print("\nKernel names:")
    for variant in ("baseline", "candidate"):
        print(f"  {variant}:")
        for category, value in summary[variant]["categories"].items():
            for name in value["names"]:
                print(f"    {category}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("/tmp/traces_glm52_m16_baseline_20260804"),
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("/tmp/traces_glm52_m16_candidate_20260804"),
    )
    parser.add_argument(
        "--trace-tools",
        default=os.environ.get("TRACE_ANALYSIS_SCRIPTS", DEFAULT_TRACE_TOOLS),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    tasks = [
        *discover(args.baseline_dir, "baseline", args.trace_tools),
        *discover(args.candidate_dir, "candidate", args.trace_tools),
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(process, tasks))
    summary = aggregate(rows)
    print_report(summary)

    if args.json_output:
        args.json_output.write_text(
            json.dumps({"rows": rows, "summary": summary}, indent=2)
        )


if __name__ == "__main__":
    main()
