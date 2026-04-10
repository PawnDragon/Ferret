#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils_data.runtime_env import sanitize_openmp_env

sanitize_openmp_env(default_threads=1)

import wandb

MILLION_BYTES = 1e6


def maybe_login(wandb_key):
    key = str(wandb_key or "").strip()
    if not key:
        return
    os.environ["WANDB_API_KEY"] = key
    wandb.login(key=key, relogin=True)


def resolve_entity(api, explicit_entity):
    if explicit_entity:
        return str(explicit_entity).strip()
    env_entity = os.getenv("WANDB_ENTITY", "").strip()
    if env_entity:
        return env_entity
    default_entity = getattr(api, "default_entity", None)
    if default_entity:
        return str(default_entity).strip()
    return ""


def fetch_run(api, project_path, run_id):
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("--run-id is required")
    try:
        return api.run(f"{project_path}/{run_id}")
    except Exception as exc:
        raise RuntimeError(f"failed to fetch run_id={run_id} from {project_path}: {exc}") from exc


def _to_finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def collect_comm_by_round(run, round_key, up_key, down_key):
    round_map = {}
    for row in run.scan_history(keys=[round_key, up_key, down_key]):
        round_val = _to_finite_float(row.get(round_key, None))
        if round_val is None:
            continue
        if round_val.is_integer():
            round_val = int(round_val)

        entry = round_map.setdefault(round_val, {})
        up_val = _to_finite_float(row.get(up_key, None))
        down_val = _to_finite_float(row.get(down_key, None))
        if up_val is not None:
            entry["up"] = up_val
        if down_val is not None:
            entry["down"] = down_val

    ordered_rounds = sorted(round_map.keys())
    sum_up = 0.0
    sum_down = 0.0
    up_rounds = 0
    down_rounds = 0
    both_rounds = 0
    per_round = []

    for round_val in ordered_rounds:
        entry = round_map[round_val]
        up_val = float(entry.get("up", 0.0))
        down_val = float(entry.get("down", 0.0))
        if "up" in entry:
            up_rounds += 1
        if "down" in entry:
            down_rounds += 1
        if ("up" in entry) and ("down" in entry):
            both_rounds += 1
        sum_up += up_val
        sum_down += down_val
        per_round.append(
            {
                "round": round_val,
                "bytes_up": up_val,
                "bytes_down": down_val,
                "bytes_total": up_val + down_val,
            }
        )

    return {
        "num_rounds_seen": len(ordered_rounds),
        "num_rounds_with_up": up_rounds,
        "num_rounds_with_down": down_rounds,
        "num_rounds_with_both": both_rounds,
        "sum_bytes_up": sum_up,
        "sum_bytes_down": sum_down,
        "sum_bytes_total": sum_up + sum_down,
        "per_round": per_round,
    }


def convert_stats_to_million_bytes(stats):
    converted = dict(stats)
    converted["sum_bytes_up_million"] = float(stats["sum_bytes_up"]) / MILLION_BYTES
    converted["sum_bytes_down_million"] = float(stats["sum_bytes_down"]) / MILLION_BYTES
    converted["sum_bytes_total_million"] = float(stats["sum_bytes_total"]) / MILLION_BYTES
    converted["per_round"] = []
    for item in stats["per_round"]:
        converted["per_round"].append(
            {
                **item,
                "bytes_up_million": float(item["bytes_up"]) / MILLION_BYTES,
                "bytes_down_million": float(item["bytes_down"]) / MILLION_BYTES,
                "bytes_total_million": float(item["bytes_total"]) / MILLION_BYTES,
            }
        )
    return converted


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sum per-round communication metrics for a single W&B run."
    )
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--entity", default="", help="W&B entity/team; optional if WANDB_ENTITY is set")
    parser.add_argument("--run-id", required=True, help="W&B run id")
    parser.add_argument("--wandb-key", default="", help="W&B API key; optional if already logged in")
    parser.add_argument("--round-key", default="round", help="History key used for round index")
    parser.add_argument("--up-key", default="comm/bytes_up", help="History key for uplink bytes")
    parser.add_argument("--down-key", default="comm/bytes_down", help="History key for downlink bytes")
    parser.add_argument("--print-per-round", action="store_true", help="Print per-round communication rows")
    parser.add_argument("--output-json", default="", help="Optional JSON output path")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    maybe_login(args.wandb_key)
    api = wandb.Api()
    entity = resolve_entity(api, args.entity)
    if not entity:
        raise RuntimeError("Cannot resolve W&B entity. Pass --entity or set WANDB_ENTITY.")

    project_path = f"{entity}/{args.project}"
    run = fetch_run(api, project_path, args.run_id)
    stats = collect_comm_by_round(
        run=run,
        round_key=args.round_key,
        up_key=args.up_key,
        down_key=args.down_key,
    )
    stats = convert_stats_to_million_bytes(stats)

    summary = {
        "project": project_path,
        "run_id": str(args.run_id),
        "run_name": getattr(run, "name", ""),
        "round_key": args.round_key,
        "up_key": args.up_key,
        "down_key": args.down_key,
        "unit": "million_bytes",
        **{k: v for k, v in stats.items() if k != "per_round"},
    }

    print(f"project: {summary['project']}")
    print(f"run_id: {summary['run_id']}")
    print(f"run_name: {summary['run_name']}")
    print(f"rounds_seen: {summary['num_rounds_seen']}")
    print(f"rounds_with_up: {summary['num_rounds_with_up']}")
    print(f"rounds_with_down: {summary['num_rounds_with_down']}")
    print(f"rounds_with_both: {summary['num_rounds_with_both']}")
    print("unit: million_bytes")
    print(f"sum_bytes_up_million: {summary['sum_bytes_up_million']:.6f}")
    print(f"sum_bytes_down_million: {summary['sum_bytes_down_million']:.6f}")
    print(f"sum_bytes_total_million: {summary['sum_bytes_total_million']:.6f}")

    if args.print_per_round:
        print("\nper_round:")
        for item in stats["per_round"]:
            print(
                f"round={item['round']} "
                f"bytes_up_million={item['bytes_up_million']:.6f} "
                f"bytes_down_million={item['bytes_down_million']:.6f} "
                f"bytes_total_million={item['bytes_total_million']:.6f}"
            )

    if str(args.output_json).strip():
        output_path = os.path.expanduser(str(args.output_json))
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        payload = dict(summary)
        payload["per_round"] = stats["per_round"]
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(payload, fout, indent=2)
        print(f"\n[info] wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
