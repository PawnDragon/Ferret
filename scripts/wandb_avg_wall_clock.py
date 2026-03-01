#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys

import wandb


def resolve_entity(api, explicit_entity):
    if explicit_entity:
        return explicit_entity
    env_entity = os.getenv("WANDB_ENTITY")
    if env_entity:
        return env_entity
    default_entity = getattr(api, "default_entity", None)
    if default_entity:
        return default_entity
    return None


def collect_wall_clock_values(run, metric_key):
    values = []
    for row in run.scan_history(keys=["_step", metric_key]):
        step = row.get("_step", None)
        value = row.get(metric_key, None)
        if step is None or value is None:
            continue
        if not isinstance(step, int):
            continue
        if not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            continue
        values.append(float(value))
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Compute average wall-clock per round from W&B runs."
    )
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument(
        "--run_name",
        default="",
        help="W&B run display name (optional; can be used with --run_id)",
    )
    parser.add_argument(
        "--run_id",
        default="",
        help="W&B run id (optional; can be used with --run_name)",
    )
    parser.add_argument("--entity", default="", help="W&B entity/team (optional)")
    parser.add_argument(
        "--metric",
        default="train/wall_clock_time",
        help="Metric key to aggregate (default: train/wall_clock_time)",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Optional CSV output path with columns: avg_wall_clock_per_round,total_rounds",
    )
    args = parser.parse_args()

    api = wandb.Api()
    entity = resolve_entity(api, args.entity.strip())
    if not entity:
        raise RuntimeError(
            "Cannot resolve W&B entity. Pass --entity or set WANDB_ENTITY."
        )

    project_path = f"{entity}/{args.project}"
    run_name = args.run_name.strip()
    run_id = args.run_id.strip()

    if not run_name and not run_id:
        print(
            "[error] Must provide at least one of --run_name or --run_id.",
            file=sys.stderr,
        )
        sys.exit(1)

    all_runs = api.runs(project_path)
    matched_runs = list(all_runs)
    if run_name:
        matched_runs = [run for run in matched_runs if run.name == run_name]
    if run_id:
        matched_runs = [run for run in matched_runs if run.id == run_id]

    if len(matched_runs) == 0:
        query_parts = []
        if run_name:
            query_parts.append(f"run_name={run_name}")
        if run_id:
            query_parts.append(f"run_id={run_id}")
        query_desc = ", ".join(query_parts)
        print(
            f"[error] No runs found for entity/project={project_path}, {query_desc}",
            file=sys.stderr,
        )
        sys.exit(1)

    all_round_values = []
    for run in matched_runs:
        run_values = collect_wall_clock_values(run, args.metric)
        all_round_values.extend(run_values)

    if len(all_round_values) == 0:
        print(
            f"[error] No valid values found for metric={args.metric} in matched runs.",
            file=sys.stderr,
        )
        sys.exit(1)

    total_rounds = len(all_round_values)
    avg_wall_clock_per_round = sum(all_round_values) / float(total_rounds)

    print(f"entity/project: {project_path}")
    if run_name:
        print(f"run_name: {run_name}")
    if run_id:
        print(f"run_id: {run_id}")
    print(f"matched_runs: {len(matched_runs)}")
    print(f"metric: {args.metric}")
    print(f"total_rounds: {total_rounds}")
    print(f"avg_wall_clock_per_round: {avg_wall_clock_per_round:.6f}")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["avg_wall_clock_per_round", "total_rounds"])
            writer.writerow([f"{avg_wall_clock_per_round:.6f}", total_rounds])
        print(f"[info] Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
