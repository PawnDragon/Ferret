#!/usr/bin/env python3
import argparse
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils_data.runtime_env import sanitize_openmp_env

sanitize_openmp_env(default_threads=1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


METHOD_SPECS = [
    ("flora", "flora_run_id", "FLoRA", "#1f77b4"),
    ("fedit", "fedit_run_id", "FedIT", "#ff7f0e"),
    ("federa", "federa_run_id", "FeDeRA", "#2ca02c"),
    ("fedexlora", "fedexlora_run_id", "FedEx-LoRA", "#9467bd"),
    ("florg", "florg_run_id", "FLoRG", "#8c564b"),
    ("fedkrso", "fedkrso_run_id", "FedKRSO", "#17becf"),
    ("fedsubmuon", "fedsubmuon_run_id", "FedSubMuon", "#1f1f1f"),
    ("fedsubmuon-GT", "fedsubmuon_gt_run_id", "FedSubMuon-GT", "#b22222"),
]

METHOD_MARKERS = {
    "flora": "o",
    "fedit": "s",
    "federa": "^",
    "fedexlora": "D",
    "florg": "v",
    "fedkrso": "P",
    "fedsubmuon": "X",
    "fedsubmuon-GT": "*",
}

METHOD_BY_KEY = {method_key: (arg_name, label, color) for method_key, arg_name, label, color in METHOD_SPECS}
METHOD_ALIASES = {
    "flora": "flora",
    "fedit": "fedit",
    "federa": "federa",
    "fedexlora": "fedexlora",
    "fedex-lora": "fedexlora",
    "florg": "florg",
    "fedkrso": "fedkrso",
    "fedsubmuon": "fedsubmuon",
    "fedsubmuon-gt": "fedsubmuon-GT",
    "fedsubmuon_gt": "fedsubmuon-GT",
}

DEFAULT_MANUAL_ROUND_COMM_BYTES = {
    "fedit": 230031360 * 2,
    "federa": 230031360 * 2,
    "flora": 230031360 * 11,
    "fedexlora": 230031360 * 12,
}

DEFAULT_MANUAL_ROUND_DOWN_COMM_BYTES = {
    "fedkrso": 164044738560,
}


def apply_publication_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#3a3a3a",
            "axes.linewidth": 0.85,
            "axes.grid": False,
            "grid.color": "#d9d9d9",
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_axis_label(key, default_label):
    key = str(key).strip().lower()
    if key == "round":
        return "Round"
    if key in {"eval/loss", "loss", "train/loss_avg"}:
        return "Validation Loss"
    return default_label


def resolve_plot_title(args, dataset_name=""):
    if args.title.strip():
        title = args.title.strip()
        if dataset_name:
            try:
                title = title.format(dataset=dataset_name)
            except Exception:
                pass
        return title
    return ""


def resolve_output_pdf_path(output_pdf):
    output_pdf = os.path.expanduser(str(output_pdf))
    root, ext = os.path.splitext(output_pdf)
    if ext.lower() != ".pdf":
        if ext == "":
            root = output_pdf
        else:
            root = root
        output_pdf = f"{root}.pdf"
    return output_pdf


def sanitize_filename_part(value):
    value = str(value).strip().replace(" ", "_")
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "dataset"


def resolve_dataset_output_pdf(output_pdf, dataset_name, multi_dataset):
    output_pdf = resolve_output_pdf_path(output_pdf)
    if not multi_dataset:
        return output_pdf
    root, ext = os.path.splitext(output_pdf)
    return f"{root}_{sanitize_filename_part(dataset_name)}{ext}"


def normalize_method_key(method_name):
    method_name = str(method_name).strip()
    alias_key = method_name.lower().replace(" ", "").replace("_", "-")
    if alias_key in METHOD_ALIASES:
        return METHOD_ALIASES[alias_key]
    if method_name in METHOD_BY_KEY:
        return method_name
    valid = ", ".join(method_key for method_key, _, _, _ in METHOD_SPECS)
    raise ValueError(f"unknown method '{method_name}'. Valid methods: {valid}")


def parse_dataset_runs_specs(raw_specs):
    groups = []
    for raw_spec in raw_specs:
        raw_spec = str(raw_spec or "").strip()
        if not raw_spec:
            continue
        if ":" not in raw_spec:
            raise ValueError(
                "--dataset-runs must use DATASET:METHOD=RUN_ID,... format, "
                f"got: {raw_spec}"
            )
        dataset_name, payload = raw_spec.split(":", 1)
        dataset_name = dataset_name.strip()
        if not dataset_name:
            raise ValueError(f"empty dataset name in --dataset-runs: {raw_spec}")
        run_ids = {}
        for part in payload.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    "--dataset-runs entries must use METHOD=RUN_ID format, "
                    f"got '{part}' in: {raw_spec}"
                )
            method_name, run_id = part.split("=", 1)
            method_key = normalize_method_key(method_name)
            run_id = run_id.strip()
            if run_id:
                run_ids[method_key] = run_id
        if not run_ids:
            raise ValueError(f"no run ids found in --dataset-runs: {raw_spec}")
        groups.append({"dataset": dataset_name, "run_ids": run_ids})
    return groups


def parse_number_expression(raw_value):
    raw_value = str(raw_value).strip().replace("_", "")
    if not raw_value:
        raise ValueError("empty numeric expression")
    product = 1.0
    for part in raw_value.split("*"):
        part = part.strip()
        if not part:
            raise ValueError(f"invalid numeric expression: {raw_value}")
        product *= float(part)
    return float(product)


def parse_manual_round_comm_specs(raw_specs, include_defaults=True):
    overrides = dict(DEFAULT_MANUAL_ROUND_COMM_BYTES) if include_defaults else {}
    for raw_spec in raw_specs:
        raw_spec = str(raw_spec or "").strip()
        if not raw_spec:
            continue
        for part in raw_spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    "--manual-round-comm entries must use METHOD=BYTES format, "
                    f"got: {part}"
                )
            method_name, raw_value = part.split("=", 1)
            method_key = normalize_method_key(method_name)
            value = parse_number_expression(raw_value)
            if value < 0:
                raise ValueError(f"manual round comm must be non-negative for {method_key}: {value}")
            overrides[method_key] = value
    return overrides


def parse_manual_round_down_comm_specs(raw_specs, include_defaults=True):
    overrides = dict(DEFAULT_MANUAL_ROUND_DOWN_COMM_BYTES) if include_defaults else {}
    for raw_spec in raw_specs:
        raw_spec = str(raw_spec or "").strip()
        if not raw_spec:
            continue
        for part in raw_spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    "--manual-round-down-comm entries must use METHOD=BYTES format, "
                    f"got: {part}"
                )
            method_name, raw_value = part.split("=", 1)
            method_key = normalize_method_key(method_name)
            value = parse_number_expression(raw_value)
            if value < 0:
                raise ValueError(f"manual round down comm must be non-negative for {method_key}: {value}")
            overrides[method_key] = value
    return overrides


def get_series_style(method_key, base_line_width):
    if method_key == "fedsubmuon":
        return {
            "linewidth": max(2.8, float(base_line_width) * 1.45),
            "alpha": 1.0,
            "linestyle": "-",
            "zorder": 4,
            "marker": METHOD_MARKERS.get(method_key, "o"),
            "markerfacecolor": "white",
            "markeredgewidth": 1.0,
            "markevery": 3,
        }
    if method_key == "fedsubmuon-GT":
        return {
            "linewidth": max(2.8, float(base_line_width) * 1.45),
            "alpha": 1.0,
            "linestyle": (0, (6, 2)),
            "zorder": 5,
            "marker": METHOD_MARKERS.get(method_key, "o"),
            "markerfacecolor": "white",
            "markeredgewidth": 1.0,
            "markevery": 3,
        }
    return {
        "linewidth": max(1.2, float(base_line_width) * 0.78),
        "alpha": 0.82,
        "linestyle": "-",
        "zorder": 2,
        "marker": METHOD_MARKERS.get(method_key, "o"),
        "markerfacecolor": "white",
        "markeredgewidth": 0.9,
        "markevery": 4,
    }


def choose_zoom_window(series):
    all_points_x = []
    for item in series:
        all_points_x.extend(float(x) for x in item["x"])
    if len(all_points_x) < 10:
        return None

    all_points_x = np.asarray(sorted(all_points_x), dtype=float)
    x_global_min = float(np.min(all_points_x))
    x_global_max = float(np.max(all_points_x))
    if x_global_max <= x_global_min:
        return None

    tail_x_min = None
    tail_points = None
    for quantile in [0.70, 0.65, 0.60, 0.55]:
        candidate = float(np.quantile(all_points_x, quantile))
        current_tail = []
        for item in series:
            for x_val, y_val in zip(item["x"], item["y"]):
                if float(x_val) >= candidate:
                    current_tail.append((float(x_val), float(y_val)))
        unique_tail_x = sorted({point[0] for point in current_tail})
        if len(current_tail) >= max(8, 2 * len(series)) and len(unique_tail_x) >= 3:
            tail_x_min = candidate
            tail_points = current_tail
            break
    if tail_x_min is None or tail_points is None:
        return None

    y_tail = np.asarray([point[1] for point in tail_points], dtype=float)
    y_min = float(np.min(y_tail))
    y_max = float(np.max(y_tail))
    y_span = float(y_max - y_min)
    y_pad = max(0.015, 0.12 * y_span) if y_span > 0 else max(0.015, abs(y_min) * 0.06)
    x_span = float(x_global_max - tail_x_min)
    x_pad = max(0.2, 0.04 * x_span)
    return {
        "xlim": (tail_x_min - x_pad, x_global_max + x_pad * 0.2),
        "ylim": (y_min - y_pad, y_max + y_pad),
    }


def choose_low_loss_zoom_window(series):
    best_points = []
    for item in series:
        if len(item["x"]) == 0 or len(item["y"]) == 0:
            continue
        y_values = np.asarray([float(y) for y in item["y"]], dtype=float)
        if y_values.size == 0:
            continue
        best_idx = int(np.argmin(y_values))
        best_points.append((float(item["x"][best_idx]), float(item["y"][best_idx])))

    if len(best_points) < 2:
        return choose_zoom_window(series)

    x_values = np.asarray([point[0] for point in best_points], dtype=float)
    y_values = np.asarray([point[1] for point in best_points], dtype=float)
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))

    y_span = float(y_max - y_min)
    positive_x = x_values[x_values > 0]
    if positive_x.size == x_values.size and x_max > x_min:
        log_min = math.log10(max(x_min, 1e-12))
        log_max = math.log10(max(x_max, 1e-12))
        log_pad = max(0.06 * float(log_max - log_min), 0.03)
        x_left = 10.0 ** (log_min - log_pad)
        x_right = 10.0 ** (log_max + log_pad)
    else:
        x_span = float(x_max - x_min)
        x_pad = max(0.04 * x_span, 0.01 * max(abs(x_max), 1.0))
        x_left = max(0.0, x_min - x_pad)
        x_right = x_max + x_pad
    y_pad = max(0.018, 0.25 * y_span) if y_span > 0 else max(0.018, abs(y_min) * 0.06)
    return {
        "xlim": (x_left, x_right),
        "ylim": (y_min - y_pad, y_max + y_pad),
    }


def parse_method_list(raw_methods):
    methods = []
    for raw_method in str(raw_methods or "").split(","):
        raw_method = raw_method.strip()
        if raw_method:
            methods.append(normalize_method_key(raw_method))
    return methods


def choose_focus_zoom_window(series, focus_methods):
    focus_method_keys = set(parse_method_list(focus_methods))
    if not focus_method_keys:
        return None

    focus_points = []
    for item in series:
        if item["method"] not in focus_method_keys:
            continue
        for x_val, y_val in zip(item["x"], item["y"]):
            focus_points.append((float(x_val), float(y_val)))
    if len(focus_points) < 2:
        return None

    x_values = np.asarray([point[0] for point in focus_points], dtype=float)
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    x_span = float(x_max - x_min)
    x_pad = max(0.05 * x_span, 0.02 * max(abs(x_max), 1.0))
    x_left = max(0.0, x_min - x_pad)
    x_right = x_max + x_pad

    local_points = []
    for item in series:
        for x_val, y_val in zip(item["x"], item["y"]):
            x_val = float(x_val)
            if x_left <= x_val <= x_right:
                local_points.append((x_val, float(y_val)))
    if len(local_points) < 2:
        local_points = focus_points

    y_values = np.asarray([point[1] for point in local_points], dtype=float)
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_span = float(y_max - y_min)
    y_pad = max(0.015, 0.14 * y_span) if y_span > 0 else max(0.015, abs(y_min) * 0.06)
    return {
        "xlim": (x_left, x_right),
        "ylim": (y_min - y_pad, y_max + y_pad),
    }


COMM_UNIT_SPECS = {
    "bytes": (1.0, "Communication Cost (Bytes)"),
    "kb": (1024.0, "Communication Cost (KB)"),
    "mb": (1024.0 ** 2, "Communication Cost (MB)"),
    "gb": (1024.0 ** 3, "Communication Cost (GB)"),
    "million": (1_000_000.0, "Communication Cost (Million Bytes)"),
}


def style_axes(ax, integer_x=True):
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#d9d9d9", linestyle="-", linewidth=0.7)
    ax.minorticks_off()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, colors="#333333")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=bool(integer_x)))


def apply_comm_xscale(ax, args):
    if args.xscale == "linear":
        return
    if args.xscale == "log":
        ax.set_xscale("log")
        return
    linthresh = max(float(args.symlog_linthresh), 1e-12)
    ax.set_xscale("symlog", linthresh=linthresh)


def resolve_entity(api, explicit_entity):
    if explicit_entity:
        return explicit_entity
    env_entity = os.getenv("WANDB_ENTITY", "").strip()
    if env_entity:
        return env_entity
    default_entity = getattr(api, "default_entity", None)
    if default_entity:
        return str(default_entity).strip()
    return ""


def maybe_login(wandb_key):
    key = str(wandb_key or "").strip()
    if not key:
        return
    os.environ["WANDB_API_KEY"] = key
    wandb.login(key=key, relogin=True)


def fetch_run(api, project_path, run_id):
    run_id = str(run_id).strip()
    if not run_id:
        return None
    try:
        return api.run(f"{project_path}/{run_id}")
    except Exception as exc:
        raise RuntimeError(f"failed to fetch run_id={run_id} from {project_path}: {exc}") from exc


def _finite_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _collect_metric_by_round(run, round_key, metric_key):
    pairs = {}
    for row in run.scan_history(keys=[round_key, metric_key]):
        round_num = _finite_float(row.get(round_key, None))
        metric_num = _finite_float(row.get(metric_key, None))
        if round_num is None or metric_num is None:
            continue
        pairs[round_num] = metric_num
    return pairs


def get_initial_comm_bytes(run, initial_comm_key, exclude_initial_comm, initial_comm_multiplier):
    if exclude_initial_comm:
        return 0.0
    key = str(initial_comm_key or "").strip()
    if not key:
        return 0.0
    try:
        value = run.summary.get(key, None)
    except Exception:
        value = None
    value = _finite_float(value)
    if value is None:
        return 0.0
    return max(0.0, value * float(initial_comm_multiplier))


def collect_loss_comm_pairs(
    run,
    method_key,
    round_key,
    loss_key,
    up_key,
    down_key,
    initial_comm_key,
    exclude_initial_comm,
    initial_comm_multiplier,
    comm_unit,
    manual_round_comm_bytes_by_method,
    manual_round_down_comm_bytes_by_method,
):
    unit_divisor, _ = COMM_UNIT_SPECS[comm_unit]
    loss_by_round = _collect_metric_by_round(run, round_key, loss_key)

    if not loss_by_round:
        return [], [], 0.0, False, None, None

    initial_comm = get_initial_comm_bytes(
        run=run,
        initial_comm_key=initial_comm_key,
        exclude_initial_comm=exclude_initial_comm,
        initial_comm_multiplier=initial_comm_multiplier,
    )

    manual_round_comm = manual_round_comm_bytes_by_method.get(method_key, None)
    if manual_round_comm is not None:
        xs = []
        ys = []
        for round_num, loss_val in sorted(loss_by_round.items(), key=lambda item: item[0]):
            cumulative_comm = initial_comm + max(float(round_num), 0.0) * float(manual_round_comm)
            xs.append(cumulative_comm / unit_divisor)
            ys.append(loss_val)
        return xs, ys, initial_comm, True, float(manual_round_comm), None

    up_by_round = _collect_metric_by_round(run, round_key, up_key)
    down_by_round = _collect_metric_by_round(run, round_key, down_key)
    manual_round_down_comm = manual_round_down_comm_bytes_by_method.get(method_key, None)
    comm_by_round = {}
    comm_rounds = set(up_by_round) | set(down_by_round)
    if manual_round_down_comm is not None:
        comm_rounds |= {round_num for round_num in loss_by_round if float(round_num) > 0.0}
    for round_num in sorted(comm_rounds):
        down_val = (
            float(manual_round_down_comm)
            if manual_round_down_comm is not None and float(round_num) > 0.0
            else down_by_round.get(round_num, 0.0)
        )
        comm_by_round[round_num] = max(0.0, up_by_round.get(round_num, 0.0)) + max(0.0, down_val)

    cumulative_comm = initial_comm
    comm_items = sorted(comm_by_round.items(), key=lambda item: item[0])
    comm_idx = 0
    xs = []
    ys = []
    for round_num, loss_val in sorted(loss_by_round.items(), key=lambda item: item[0]):
        while comm_idx < len(comm_items) and comm_items[comm_idx][0] <= round_num:
            cumulative_comm += comm_items[comm_idx][1]
            comm_idx += 1
        xs.append(cumulative_comm / unit_divisor)
        ys.append(loss_val)
    return xs, ys, initial_comm, bool(comm_by_round), None, manual_round_down_comm


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot loss-communication curves for multiple Ferret methods from W&B runs. Each method run id is optional."
    )
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--entity", default="", help="W&B entity/team; optional if WANDB_ENTITY is set")
    parser.add_argument("--wandb-key", default="", help="W&B API key; optional if already logged in")
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset name for the legacy single-group run-id arguments",
    )
    parser.add_argument(
        "--dataset-runs",
        action="append",
        default=[],
        metavar="DATASET:METHOD=RUN_ID,...",
        help=(
            "Dataset-specific run ids. Repeat this arg to save one PDF per dataset, "
            "for example: --dataset-runs dolly:flora=abc,fedit=def,fedsubmuon_gt=ghi"
        ),
    )
    parser.add_argument("--round-key", default="round", help="History key used to align loss and communication metrics")
    parser.add_argument("--loss-key", default="eval/loss", help="History key used for y-axis (default: eval/loss)")
    parser.add_argument("--up-key", default="comm/bytes_up", help="History key for per-round uplink bytes")
    parser.add_argument("--down-key", default="comm/bytes_down", help="History key for per-round downlink bytes")
    parser.add_argument("--initial-comm-key", default="comm/initial_bytes", help="Summary key for one-time initial communication bytes")
    parser.add_argument(
        "--manual-round-comm",
        action="append",
        default=[],
        metavar="METHOD=BYTES",
        help=(
            "Override total per-round communication bytes for selected methods. "
            "Can be repeated or comma-separated, e.g. --manual-round-comm fedit=230031360*2,flora=230031360*11"
        ),
    )
    parser.add_argument(
        "--no-default-manual-round-comm",
        action="store_true",
        help=(
            "Disable built-in manual per-round comm defaults for fedit/federa/flora/fedexlora "
            "and use W&B comm metrics unless --manual-round-comm is provided."
        ),
    )
    parser.add_argument(
        "--manual-round-down-comm",
        action="append",
        default=[],
        metavar="METHOD=BYTES",
        help=(
            "Override per-round downlink bytes while still reading uplink bytes from W&B. "
            "Can be repeated or comma-separated, e.g. --manual-round-down-comm fedkrso=164044738560"
        ),
    )
    parser.add_argument(
        "--no-default-manual-round-down-comm",
        action="store_true",
        help="Disable built-in manual downlink default for fedkrso and use W&B downlink unless overridden.",
    )
    parser.add_argument(
        "--exclude-initial-comm",
        action="store_true",
        help="Do not add comm/initial_bytes to the cumulative communication x-axis",
    )
    parser.add_argument(
        "--initial-comm-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to the initial communication summary value before adding it to x-axis",
    )
    parser.add_argument(
        "--comm-unit",
        choices=sorted(COMM_UNIT_SPECS),
        default="million",
        help="Unit used for communication x-axis",
    )
    parser.add_argument(
        "--xscale",
        choices=["linear", "symlog", "log"],
        default="linear",
        help="Communication x-axis scale. Default is linear for a uniform x-axis.",
    )
    parser.add_argument(
        "--symlog-linthresh",
        type=float,
        default=1.0,
        help="Linear threshold for symlog x-axis, expressed in the selected --comm-unit.",
    )
    parser.add_argument(
        "--output-pdf",
        default="wandb_plots/loss_comm_comparison.pdf",
        help="Output PDF path",
    )
    parser.add_argument("--title", default="", help="Optional plot title")
    parser.add_argument("--fig-width", type=float, default=7.0, help="Figure width in inches")
    parser.add_argument("--fig-height", type=float, default=4.2, help="Figure height in inches")
    parser.add_argument("--line-width", type=float, default=2.0, help="Line width")
    parser.add_argument("--marker-size", type=float, default=3.0, help="Marker size")
    parser.add_argument(
        "--show-inset",
        dest="show_inset",
        action="store_true",
        default=False,
        help="Show a zoom inset focused on low-communication target methods. Disabled by default.",
    )
    parser.add_argument(
        "--hide-inset",
        dest="show_inset",
        action="store_false",
        help="Disable the zoom inset.",
    )
    parser.add_argument(
        "--inset-focus-methods",
        default="fedsubmuon,fedsubmuon-GT",
        help="Comma-separated methods used to choose the inset zoom window.",
    )
    parser.add_argument(
        "--inset-mode",
        choices=["tail", "focus"],
        default="tail",
        help="Inset window selection mode. tail zooms the lowest-loss region across methods; focus zooms selected methods.",
    )
    parser.add_argument("--flora-run-id", default="", help="Optional W&B run id for flora")
    parser.add_argument("--fedit-run-id", default="", help="Optional W&B run id for fedit")
    parser.add_argument("--federa-run-id", default="", help="Optional W&B run id for federa")
    parser.add_argument("--fedexlora-run-id", default="", help="Optional W&B run id for fedexlora")
    parser.add_argument("--florg-run-id", default="", help="Optional W&B run id for florg")
    parser.add_argument("--fedkrso-run-id", default="", help="Optional W&B run id for fedkrso")
    parser.add_argument("--fedsubmuon-run-id", default="", help="Optional W&B run id for fedsubmuon")
    parser.add_argument("--fedsubmuon-gt-run-id", dest="fedsubmuon_gt_run_id", default="", help="Optional W&B run id for fedsubmuon-GT")
    return parser


def build_run_groups(args):
    groups = parse_dataset_runs_specs(args.dataset_runs)
    legacy_run_ids = {}
    for method_key, arg_name, _, _ in METHOD_SPECS:
        run_id = getattr(args, arg_name, "")
        if str(run_id).strip():
            legacy_run_ids[method_key] = str(run_id).strip()
    if legacy_run_ids:
        groups.append(
            {
                "dataset": args.dataset.strip() or "default",
                "run_ids": legacy_run_ids,
            }
        )
    return groups


def collect_series_for_group(
    api,
    project_path,
    args,
    group,
    manual_round_comm_bytes_by_method,
    manual_round_down_comm_bytes_by_method,
):
    series = []
    for method_key, arg_name, label, color in METHOD_SPECS:
        run_id = group["run_ids"].get(method_key, "")
        if not str(run_id).strip():
            # Missing ids are allowed; just skip that method.
            continue
        run = fetch_run(api, project_path, run_id)
        (
            xs,
            ys,
            initial_comm_bytes,
            has_comm_metrics,
            manual_round_comm_bytes,
            manual_round_down_comm_bytes,
        ) = collect_loss_comm_pairs(
            run=run,
            method_key=method_key,
            round_key=args.round_key,
            loss_key=args.loss_key,
            up_key=args.up_key,
            down_key=args.down_key,
            initial_comm_key=args.initial_comm_key,
            exclude_initial_comm=args.exclude_initial_comm,
            initial_comm_multiplier=args.initial_comm_multiplier,
            comm_unit=args.comm_unit,
            manual_round_comm_bytes_by_method=manual_round_comm_bytes_by_method,
            manual_round_down_comm_bytes_by_method=manual_round_down_comm_bytes_by_method,
        )
        if len(xs) == 0:
            print(
                f"[warn] skip {label}: no valid points for round_key={args.round_key}, loss_key={args.loss_key}",
                file=sys.stderr,
            )
            continue
        if not has_comm_metrics:
            print(
                f"[warn] {label}: no finite {args.up_key}/{args.down_key} values found; x-axis uses only initial comm if present",
                file=sys.stderr,
            )
        series.append(
            {
                "method": method_key,
                "label": label,
                "color": color,
                "run_id": str(run.id),
                "run_name": str(run.name),
                "x": xs,
                "y": ys,
                "initial_comm_bytes": initial_comm_bytes,
                "manual_round_comm_bytes": manual_round_comm_bytes,
                "manual_round_down_comm_bytes": manual_round_down_comm_bytes,
            }
        )
    return series


def save_series_plot(series, args, output_pdf, dataset_name):
    if len(series) == 0:
        raise RuntimeError(
            f"No valid runs/points found for dataset={dataset_name}. "
            "Check run ids, entity/project, and metric keys."
        )
    output_dir = os.path.dirname(output_pdf)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(float(args.fig_width), float(args.fig_height)))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.76, bottom=0.14, top=0.90)
    style_axes(ax, integer_x=False)
    apply_comm_xscale(ax, args)
    for item in series:
        line_style = get_series_style(item["method"], args.line_width)
        ax.plot(
            item["x"],
            item["y"],
            label=item["label"],
            color=item["color"],
            linewidth=float(line_style["linewidth"]),
            linestyle=line_style["linestyle"],
            alpha=float(line_style["alpha"]),
            zorder=int(line_style["zorder"]),
            marker=line_style["marker"],
            markersize=float(args.marker_size),
            markerfacecolor=line_style["markerfacecolor"],
            markeredgecolor=item["color"],
            markeredgewidth=float(line_style["markeredgewidth"]),
            markevery=int(line_style["markevery"]),
            solid_capstyle="round",
        )

    _, comm_axis_label = COMM_UNIT_SPECS[args.comm_unit]
    ax.set_xlabel(comm_axis_label)
    ax.set_ylabel(resolve_axis_label(args.loss_key, str(args.loss_key)))
    plot_title = resolve_plot_title(args, dataset_name=dataset_name)
    if plot_title:
        ax.set_title(plot_title, pad=8)

    zoom_window = None
    if args.show_inset:
        if args.inset_mode == "focus":
            zoom_window = choose_focus_zoom_window(series, args.inset_focus_methods)
            if zoom_window is None:
                zoom_window = choose_low_loss_zoom_window(series)
        else:
            zoom_window = choose_low_loss_zoom_window(series)
    if zoom_window is not None:
        inset_ax = ax.inset_axes([0.66, 0.72, 0.32, 0.22])
        inset_ax.set_zorder(10)
        inset_ax.patch.set_facecolor("white")
        inset_ax.patch.set_alpha(0.96)
        style_axes(inset_ax, integer_x=False)
        apply_comm_xscale(inset_ax, args)
        for item in series:
            line_style = get_series_style(item["method"], args.line_width)
            inset_ax.plot(
                item["x"],
                item["y"],
                color=item["color"],
                linewidth=float(line_style["linewidth"]),
                linestyle=line_style["linestyle"],
                alpha=float(line_style["alpha"]),
                zorder=int(line_style["zorder"]),
                marker=line_style["marker"],
                markersize=max(2.0, float(args.marker_size) * 0.8),
                markerfacecolor=line_style["markerfacecolor"],
                markeredgecolor=item["color"],
                markeredgewidth=float(line_style["markeredgewidth"]),
                markevery=int(line_style["markevery"]),
                solid_capstyle="round",
            )
        inset_ax.set_xlim(*zoom_window["xlim"])
        inset_ax.set_ylim(*zoom_window["ylim"])
        inset_ax.tick_params(axis="both", which="major", labelsize=10, length=3)
        try:
            ax.indicate_inset_zoom(inset_ax, edgecolor="#7f7f7f", alpha=0.9)
        except Exception:
            pass

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        borderaxespad=0.0,
        handlelength=3.0,
    )
    fig.savefig(output_pdf, format="pdf", bbox_extra_artists=(legend,), bbox_inches="tight")
    plt.close(fig)
    return output_pdf


def main():
    parser = build_parser()
    args = parser.parse_args()
    apply_publication_style()
    maybe_login(args.wandb_key)
    api = wandb.Api()
    entity = resolve_entity(api, args.entity.strip())
    if not entity:
        raise RuntimeError("Cannot resolve W&B entity. Pass --entity or set WANDB_ENTITY.")

    project_path = f"{entity}/{args.project}"
    run_groups = build_run_groups(args)
    if len(run_groups) == 0:
        raise RuntimeError("No run ids provided. Pass method run ids or one/more --dataset-runs entries.")
    manual_round_comm_bytes_by_method = parse_manual_round_comm_specs(
        args.manual_round_comm,
        include_defaults=not args.no_default_manual_round_comm,
    )
    manual_round_down_comm_bytes_by_method = parse_manual_round_down_comm_specs(
        args.manual_round_down_comm,
        include_defaults=not args.no_default_manual_round_down_comm,
    )

    print(f"[info] entity/project: {project_path}")
    print(
        f"[info] round_key={args.round_key}, loss_key={args.loss_key}, "
        f"up_key={args.up_key}, down_key={args.down_key}, comm_unit={args.comm_unit}, "
        f"xscale={args.xscale}"
    )
    if not args.exclude_initial_comm:
        print(
            f"[info] initial_comm_key={args.initial_comm_key}, "
            f"initial_comm_multiplier={args.initial_comm_multiplier}"
        )
    if manual_round_comm_bytes_by_method:
        pretty_manual_comm = ", ".join(
            f"{method}={value:.0f}" for method, value in sorted(manual_round_comm_bytes_by_method.items())
        )
        print(f"[info] manual per-round comm overrides(bytes): {pretty_manual_comm}")
    if manual_round_down_comm_bytes_by_method:
        pretty_manual_down_comm = ", ".join(
            f"{method}={value:.0f}" for method, value in sorted(manual_round_down_comm_bytes_by_method.items())
        )
        print(f"[info] manual per-round downlink overrides(bytes): {pretty_manual_down_comm}")

    multi_dataset = len(run_groups) > 1
    for group in run_groups:
        dataset_name = group["dataset"]
        series = collect_series_for_group(
            api,
            project_path,
            args,
            group,
            manual_round_comm_bytes_by_method=manual_round_comm_bytes_by_method,
            manual_round_down_comm_bytes_by_method=manual_round_down_comm_bytes_by_method,
        )
        output_pdf = resolve_dataset_output_pdf(args.output_pdf, dataset_name, multi_dataset=multi_dataset)
        saved_path = save_series_plot(series, args, output_pdf, dataset_name)
        print(f"[info] Saved PDF for dataset={dataset_name} to {saved_path}")
        for item in series:
            print(
                f"[info] {dataset_name}/{item['label']}: run_id={item['run_id']}, "
                f"run_name={item['run_name']}, points={len(item['x'])}, "
                f"comm_min={item['x'][0]:.4g}, comm_max={item['x'][-1]:.4g}, "
                f"initial_comm_bytes={item['initial_comm_bytes']:.4g}, "
                f"manual_round_comm_bytes={item['manual_round_comm_bytes']}, "
                f"manual_round_down_comm_bytes={item['manual_round_down_comm_bytes']}"
            )


if __name__ == "__main__":
    main()
