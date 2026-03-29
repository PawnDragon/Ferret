import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from evaluations import rouge_score
from optimizers.ferret_optimizer import FerretFramework
from optimizers.lora_utils import (
    build_lora_model,
    load_classifier_state,
    load_lora_A_state,
    load_lora_B_state,
    load_lora_state,
)
from optimizers.florg_utils import (
    build_florg_model,
    load_florg_A_state,
    sample_florg_delta_norm,
)
from utils_data.load_data import get_loaders
from utils_data.gsm8k_metrics import (
    compute_gsm8k_metrics,
    extract_gsm8k_gold_final_answer,
    extract_gsm8k_pred_final_answer,
)
from utils_data.math_metrics import (
    compute_math_metrics,
    extract_math_gold_final_answer,
    extract_math_pred_final_answer,
)
from utils_data.model_loader import resolve_model_source, resolve_torch_dtype


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def resolve_runtime_device(requested_device):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cuda_count = torch.cuda.device_count()
    if cuda_count <= 0:
        print("[warn] No CUDA GPU visible, using CPU")
        return torch.device("cpu"), "cpu"

    dev = int(requested_device)
    if dev < 0 or dev >= cuda_count:
        print(
            f"[warn] --device {dev} is invalid for {cuda_count} visible GPU(s); fallback to 0"
        )
        dev = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
    return torch.device("cuda:0"), "cuda"


def load_checkpoint_into_model(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")

    ckpt_type = "state_dict"
    x_global = None
    m_global = None
    v_global = None
    seeds = None
    u_global = None
    v_basis_global = None
    saved_algo = None
    global_lora_state = None
    global_lora_A_state = None
    global_lora_B_state = None
    global_classifier_state = None
    global_deltaW_state = None
    common_hparams = {}
    lora_hparams = {}
    global_florg_A_state = None
    global_florg_seed_state = None
    global_florg_basis_state = None
    florg_hparams = {}
    global_multisub_b_state = None
    global_multisub_c_state = None
    global_multisub_metadata = None
    global_multisub_scores = None
    global_multisub_selected_keys = None
    global_struct_x_state = None
    global_struct_metadata = None
    global_struct_scores = None
    global_struct_selected_keys = None

    if isinstance(payload, dict) and "backbone_state_dict" in payload:
        model.load_state_dict(payload["backbone_state_dict"], strict=True)
        x_global = payload.get("x_global", None)
        m_global = payload.get("m_global", None)
        v_global = payload.get("v_global", None)
        seeds = payload.get("seeds", None)
        u_global = payload.get("u_global", None)
        v_basis_global = payload.get("v_basis_global", None)
        global_lora_state = payload.get("global_lora_state", None)
        global_lora_A_state = payload.get("global_lora_A_state", None)
        global_lora_B_state = payload.get("global_lora_B_state", None)
        global_classifier_state = payload.get("global_classifier_state", None)
        global_deltaW_state = payload.get("global_deltaW_state", None)
        global_florg_A_state = payload.get("global_florg_A_state", None)
        global_florg_seed_state = payload.get("global_florg_seed_state", None)
        global_florg_basis_state = payload.get("global_florg_basis_state", None)
        global_multisub_b_state = payload.get("global_multisub_b_state", None)
        global_multisub_c_state = payload.get("global_multisub_c_state", None)
        global_multisub_metadata = payload.get("global_multisub_metadata", None)
        global_multisub_scores = payload.get("global_multisub_scores", None)
        global_multisub_selected_keys = payload.get("global_multisub_selected_keys", None)
        global_struct_x_state = payload.get("global_struct_x_state", None)
        global_struct_metadata = payload.get("global_struct_metadata", None)
        global_struct_scores = payload.get("global_struct_scores", None)
        global_struct_selected_keys = payload.get("global_struct_selected_keys", None)
        lora_hparams = (
            payload.get("lora_hparams", {})
            if isinstance(payload.get("lora_hparams", {}), dict)
            else {}
        )
        florg_hparams = (
            payload.get("florg_hparams", {})
            if isinstance(payload.get("florg_hparams", {}), dict)
            else {}
        )
        hparams = payload.get("hparams", None)
        if isinstance(hparams, dict):
            common_hparams = dict(hparams)
            saved_algo = hparams.get("algo", None)
        if saved_algo is None:
            saved_algo = payload.get("algo", None)

        if (
            saved_algo in ["fedit", "federa", "flora"]
            or global_lora_state is not None
            or global_deltaW_state is not None
        ):
            ckpt_type = "lora_best"
        elif (
            saved_algo == "fedexlora"
            or (
                global_lora_A_state is not None
                and global_lora_B_state is not None
            )
        ):
            ckpt_type = "fedexlora_best"
        elif saved_algo == "florg" or global_florg_A_state is not None:
            ckpt_type = "florg_best"
        elif (
            saved_algo == "fedmultisubmuon"
            or (
                global_multisub_b_state is not None
                and global_multisub_c_state is not None
                and global_multisub_metadata is not None
            )
        ):
            ckpt_type = "fedmultisubmuon_best"
        elif (
            saved_algo == "fedstructmuon"
            or (global_struct_x_state is not None and global_struct_metadata is not None)
        ):
            ckpt_type = "fedstructmuon_best"
        elif saved_algo == "fedavg":
            ckpt_type = "fedavg_best"
        elif saved_algo == "ferret":
            ckpt_type = "ferret_best"
        elif saved_algo == "fedsubmuonv2":
            ckpt_type = "fedsubmuonv2_best"
        elif saved_algo == "fedsubmuon_gt" or (
            x_global is not None
            and seeds is not None
            and u_global is not None
            and v_basis_global is not None
        ):
            ckpt_type = "fedsubmuon_gt_best"
        elif saved_algo in ["fedsubmuon", "fedsubadam", "fedsubsgd"] or (
            x_global is not None and seeds is not None
        ):
            ckpt_type = "fedsubmuon_best"
        else:
            ckpt_type = "backbone_best"
    elif isinstance(payload, dict):
        model.load_state_dict(payload, strict=True)
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(payload)}")

    return (
        ckpt_type,
        x_global,
        m_global,
        v_global,
        seeds,
        u_global,
        v_basis_global,
        saved_algo,
        global_lora_state,
        global_lora_A_state,
        global_lora_B_state,
        global_classifier_state,
        global_deltaW_state,
        common_hparams,
        lora_hparams,
        global_florg_A_state,
        global_florg_seed_state,
        global_florg_basis_state,
        florg_hparams,
        global_multisub_b_state,
        global_multisub_c_state,
        global_multisub_metadata,
        global_multisub_scores,
        global_multisub_selected_keys,
        global_struct_x_state,
        global_struct_metadata,
        global_struct_scores,
        global_struct_selected_keys,
    )


def to_left_padded_inputs(input_ids, attention_mask, pad_token_id):
    """
    Convert right-padded batch tensors to left-padded layout for decoder-only generation.
    """
    bs, seq_len = input_ids.shape
    left_input_ids = torch.full_like(input_ids, pad_token_id)
    left_attention_mask = torch.zeros_like(attention_mask)
    for i in range(bs):
        valid_tokens = input_ids[i][attention_mask[i].bool()]
        n = valid_tokens.numel()
        if n > 0:
            left_input_ids[i, seq_len - n :] = valid_tokens
            left_attention_mask[i, seq_len - n :] = 1
    return left_input_ids, left_attention_mask


def sanitize_greedy_generation_config(model):
    """
    Normalize generation_config to greedy defaults to avoid repeated warnings
    about sampling-only flags when do_sample=False.
    """
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return
    try:
        gen_cfg.do_sample = False
        gen_cfg.num_beams = 1
        if hasattr(gen_cfg, "temperature"):
            gen_cfg.temperature = 1.0
        if hasattr(gen_cfg, "top_p"):
            gen_cfg.top_p = 1.0
        if hasattr(gen_cfg, "top_k"):
            gen_cfg.top_k = 50
    except Exception:
        return


def maybe_resolve_gsm8k_eval_metric(args, eval_metric_explicit=False):
    if args.dataset == "gsm8k" and args.eval_metric == "rouge" and (not eval_metric_explicit):
        args.eval_metric = "gsm8k_acc"
        print("[info] dataset=gsm8k and --eval_metric not set; defaulting eval metric to gsm8k_acc")
    if args.dataset == "math" and args.eval_metric == "rouge" and (not eval_metric_explicit):
        args.eval_metric = "math_acc"
        print("[info] dataset=math and --eval_metric not set; defaulting eval metric to math_acc")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--algo",
        type=str,
        default="auto",
        choices=[
            "auto",
            "ferret",
            "fedsubmuon",
            "fedsubmuonv2",
            "fedsubmuon_gt",
            "fedsubadam",
            "fedsubsgd",
            "fedmultisubmuon",
            "fedstructmuon",
            "fedit",
            "federa",
            "flora",
            "fedexlora",
            "florg",
            "fedavg",
        ],
    )

    # Data/eval args to keep dolly processing aligned with main.py
    parser.add_argument(
        "--dataset", type=str, default="dolly", choices=["dolly", "instruct", "gsm8k", "math"]
    )
    parser.add_argument("--zerotask", type=int, default=7)
    parser.add_argument("--dataset_subsample", type=float, default=1.0)
    parser.add_argument("--iid", type=str, default="dir0.5")
    parser.add_argument(
        "--ni_root",
        type=str,
        default="./data/NI",
        help="root directory for Natural Instructions dataset",
    )
    parser.add_argument(
        "--gsm8k_root",
        type=str,
        default="./data/gsm8k",
        help="root directory for local GSM8K dataset files",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/math",
        help="root directory for local MATH dataset parquet files",
    )
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--use_prompts", default=True)
    parser.add_argument(
        "--eval_metric", type=str, default="rouge", choices=["loss", "rouge", "gsm8k_acc", "math_acc"]
    )

    # Runtime/model args used by framework/server-style behavior
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank_r", type=int, default=8)
    parser.add_argument("--rank_left", type=int, default=None)
    parser.add_argument("--rank_right", type=int, default=None)
    parser.add_argument("--svd_rank", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--optimizer",
        type=lambda x: x.lower(),
        default="adamw",
        choices=["adamw", "sgd"],
    )
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--ns_steps", type=int, default=5)
    parser.add_argument("--gt_topk", type=int, default=0)
    parser.add_argument("--gt_rank1_approx", default=False, action="store_true")
    parser.add_argument("--gt_target_rel_step", type=float, default=0.0)
    parser.add_argument(
        "--basis_init_mode",
        type=lambda x: str(x).lower(),
        default="random",
        choices=["random", "svd_left", "svd_right", "svd_both"],
    )
    parser.add_argument(
        "--gt_update_mode",
        type=lambda x: str(x).lower(),
        default="both",
        choices=["both", "left", "right", "alternate_lr", "alternate_rl"],
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_accum", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=-100.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--federa_svd_dtype", type=str, default="fp32", choices=["fp32", "fp64"])
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--florg_rank_r", type=int, default=16)
    parser.add_argument("--florg_seed_base", type=int, default=95317)
    parser.add_argument("--multisub_num_subspaces", type=int, default=4)
    parser.add_argument("--multisub_topk", type=int, default=64)
    parser.add_argument("--multisub_seed_base", type=int, default=0)
    parser.add_argument("--multisub_score_interval", type=int, default=1)
    parser.add_argument("--multisub_score_beta1", type=float, default=0.9)
    parser.add_argument("--multisub_score_beta2", type=float, default=0.999)
    parser.add_argument("--struct_num_subspaces", type=int, default=4)
    parser.add_argument("--struct_topk", type=int, default=64)
    parser.add_argument("--struct_seed_base", type=int, default=0)
    parser.add_argument("--struct_score_interval", type=int, default=1)

    parser.add_argument("--save_json", type=str, default="")
    return parser


def run_evaluate(args, eval_metric_explicit=False):
    device, _ = resolve_runtime_device(args.device)
    setup_seed(args.seed)
    maybe_resolve_gsm8k_eval_metric(args, eval_metric_explicit=eval_metric_explicit)
    if args.dataset != "gsm8k" and args.eval_metric == "gsm8k_acc":
        raise ValueError("--eval_metric gsm8k_acc is only valid for --dataset gsm8k")
    if args.dataset != "math" and args.eval_metric == "math_acc":
        raise ValueError("--eval_metric math_acc is only valid for --dataset math")
    if getattr(args, "rank_left", None) is None:
        args.rank_left = int(args.rank_r)
    if getattr(args, "rank_right", None) is None:
        args.rank_right = int(args.rank_r)
    if int(args.rank_left) <= 0:
        args.rank_left = int(args.rank_r)
    if int(args.rank_right) <= 0:
        args.rank_right = int(args.rank_r)

    # Keep exactly the same final-eval path as main.py.
    setup_seed(args.seed)
    _, eval_loader, tokenizer = get_loaders(args, only_eval=True)
    if args.dataset == "dolly":
        print(
            f"[info] Eval dataset=dolly | zerotask={args.zerotask}, eval_samples={len(eval_loader.dataset)}"
        )
    else:
        print(
            f"[info] Eval dataset={args.dataset} | eval_samples={len(eval_loader.dataset)}"
        )

    model_source = resolve_model_source(args.model)
    model_dtype = resolve_torch_dtype(getattr(args, "model_dtype", "bf16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map="cpu",
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ckpt_type = "none"
    x_global = None
    m_global = None
    v_global = None
    seeds = None
    u_global = None
    v_basis_global = None
    saved_algo = None
    global_lora_state = None
    global_lora_A_state = None
    global_lora_B_state = None
    global_classifier_state = None
    global_deltaW_state = None
    common_hparams = {}
    lora_hparams = {}
    global_florg_A_state = None
    global_florg_seed_state = None
    global_florg_basis_state = None
    florg_hparams = {}
    global_multisub_b_state = None
    global_multisub_c_state = None
    global_multisub_metadata = None
    global_multisub_scores = None
    global_multisub_selected_keys = None
    global_struct_x_state = None
    global_struct_metadata = None
    global_struct_scores = None
    global_struct_selected_keys = None
    if args.checkpoint:
        (
            ckpt_type,
            x_global,
            m_global,
            v_global,
            seeds,
            u_global,
            v_basis_global,
            saved_algo,
            global_lora_state,
            global_lora_A_state,
            global_lora_B_state,
            global_classifier_state,
            global_deltaW_state,
            common_hparams,
            lora_hparams,
            global_florg_A_state,
            global_florg_seed_state,
            global_florg_basis_state,
            florg_hparams,
            global_multisub_b_state,
            global_multisub_c_state,
            global_multisub_metadata,
            global_multisub_scores,
            global_multisub_selected_keys,
            global_struct_x_state,
            global_struct_metadata,
            global_struct_scores,
            global_struct_selected_keys,
        ) = load_checkpoint_into_model(model, args.checkpoint)
        print(f"[info] Loaded checkpoint: {args.checkpoint} ({ckpt_type})")

    if isinstance(common_hparams, dict) and len(common_hparams) > 0:
        if "lora_target_modules" in common_hparams:
            args.lora_target_modules = common_hparams["lora_target_modules"]
        if "basis_init_mode" in common_hparams:
            args.basis_init_mode = str(common_hparams["basis_init_mode"]).lower()
        if "gt_update_mode" in common_hparams:
            args.gt_update_mode = str(common_hparams["gt_update_mode"]).lower()
        if "gt_rank1_approx" in common_hparams:
            args.gt_rank1_approx = bool(common_hparams["gt_rank1_approx"])
        if "gt_target_rel_step" in common_hparams:
            args.gt_target_rel_step = float(common_hparams["gt_target_rel_step"])
        if "rank_left" in common_hparams:
            args.rank_left = int(common_hparams["rank_left"])
        elif "rank_r" in common_hparams:
            args.rank_left = int(common_hparams["rank_r"])
        if "rank_right" in common_hparams:
            args.rank_right = int(common_hparams["rank_right"])
        elif "rank_r" in common_hparams:
            args.rank_right = int(common_hparams["rank_r"])

    eval_algo = args.algo
    if eval_algo == "auto":
        if saved_algo in [
            "ferret",
            "fedsubmuon",
            "fedsubmuonv2",
            "fedsubmuon_gt",
            "fedsubadam",
            "fedsubsgd",
            "fedmultisubmuon",
            "fedstructmuon",
            "fedit",
            "federa",
            "flora",
            "fedexlora",
            "florg",
            "fedavg",
        ]:
            eval_algo = saved_algo
        else:
            if global_lora_state is not None:
                eval_algo = "fedit"
            elif (
                global_lora_A_state is not None
                and global_lora_B_state is not None
            ):
                eval_algo = "fedexlora"
            elif global_florg_A_state is not None:
                eval_algo = "florg"
            elif (
                global_multisub_b_state is not None
                and global_multisub_c_state is not None
                and global_multisub_metadata is not None
            ):
                eval_algo = "fedmultisubmuon"
            elif (
                global_struct_x_state is not None
                and global_struct_metadata is not None
            ):
                eval_algo = "fedstructmuon"
            elif global_deltaW_state is not None:
                eval_algo = "flora"
            else:
                eval_algo = (
                    "fedsubmuon_gt"
                    if (x_global is not None and seeds is not None and u_global is not None and v_basis_global is not None)
                    else ("fedsubmuon" if (x_global is not None and seeds is not None) else "ferret")
                )

    # Align LoRA eval settings with saved checkpoint when available.
    if isinstance(lora_hparams, dict) and len(lora_hparams) > 0:
        if "lora_r" in lora_hparams:
            args.lora_r = int(lora_hparams["lora_r"])
        if "lora_alpha" in lora_hparams:
            args.lora_alpha = float(lora_hparams["lora_alpha"])
        if "lora_dropout" in lora_hparams:
            args.lora_dropout = float(lora_hparams["lora_dropout"])
        if "lora_target_modules" in lora_hparams:
            args.lora_target_modules = lora_hparams["lora_target_modules"]
        if "lora_bias" in lora_hparams:
            args.lora_bias = lora_hparams["lora_bias"]
    if isinstance(florg_hparams, dict) and len(florg_hparams) > 0:
        if "florg_rank_r" in florg_hparams:
            args.florg_rank_r = int(florg_hparams["florg_rank_r"])
        if "florg_seed_base" in florg_hparams:
            args.florg_seed_base = int(florg_hparams["florg_seed_base"])
        if "lora_target_modules" in florg_hparams:
            args.lora_target_modules = florg_hparams["lora_target_modules"]

    model = model.to(device)
    model.eval()
    eval_model = model

    framework = None
    if eval_algo in ["fedsubmuon", "fedsubmuonv2", "fedsubmuon_gt", "fedsubadam", "fedsubsgd"]:
        if x_global is None or seeds is None:
            raise ValueError("FedSub eval requires checkpoint with x_global and seeds")
        if eval_algo == "fedsubmuon" and m_global is None:
            raise ValueError("FedSubMuon eval requires checkpoint with m_global")
        if eval_algo == "fedsubmuon_gt":
            if not isinstance(u_global, dict) or not isinstance(v_basis_global, dict):
                raise ValueError("FedSubMuon-GT eval requires checkpoint with u_global and v_basis_global")
        if len(x_global) == 0:
            raise ValueError("FedSub checkpoint contains empty x_global")
        first_key = next(iter(x_global.keys()))
        ckpt_rank = int(x_global[first_key].shape[0])
        if int(args.rank_r) != ckpt_rank:
            print(
                f"[info] rank_r mismatch (args={args.rank_r}, ckpt={ckpt_rank}); override args.rank_r with ckpt rank"
            )
            args.rank_r = ckpt_rank
        args.algo = eval_algo
        framework = FerretFramework(model, args=args, lr=args.lr, candidate_seeds=[])
        uv_state = None
        if eval_algo == "fedsubmuon_gt":
            uv_state = {"u": u_global, "v": v_basis_global}
        framework.set_submuon_state(
            x_global,
            m_global if eval_algo == "fedsubmuon" else None,
            seeds,
            trainable=False,
            v_state=None,
            uv_state=uv_state,
        )
    elif eval_algo == "fedmultisubmuon":
        if (
            global_multisub_b_state is None
            or global_multisub_c_state is None
            or global_multisub_metadata is None
        ):
            raise ValueError(
                "FedMultiSubMuon eval requires checkpoint with global_multisub_b_state, "
                "global_multisub_c_state and global_multisub_metadata"
            )
        args.algo = eval_algo
        framework = FerretFramework(model, args=args, lr=args.lr, candidate_seeds=[])
        framework.set_multisub_state(
            {
                "b_global": global_multisub_b_state,
                "c_global": global_multisub_c_state,
                "metadata": global_multisub_metadata,
                "selected_keys": list(global_multisub_b_state.keys()),
                "score_state": global_multisub_scores if isinstance(global_multisub_scores, dict) else {},
            },
            trainable=False,
        )
    elif eval_algo == "fedstructmuon":
        if global_struct_x_state is None or global_struct_metadata is None:
            raise ValueError(
                "FedStructMuon eval requires checkpoint with global_struct_x_state and global_struct_metadata"
            )
        args.algo = eval_algo
        framework = FerretFramework(model, args=args, lr=args.lr, candidate_seeds=[])
        framework.set_struct_state(
            {
                "x_global": global_struct_x_state,
                "metadata": global_struct_metadata,
                "selected_keys": list(global_struct_x_state.keys()),
                "score_state": global_struct_scores if isinstance(global_struct_scores, dict) else {},
            },
            trainable=False,
        )
    elif eval_algo in ["fedit", "federa"]:
        if global_lora_state is None:
            raise ValueError("LoRA eval requires checkpoint with global_lora_state")
        eval_model = build_lora_model(model, args)
        load_lora_state(eval_model, global_lora_state)
        eval_model = eval_model.to(device)
        eval_model.eval()
    elif eval_algo == "fedexlora":
        if global_lora_A_state is None:
            raise ValueError("FedEx-LoRA eval requires checkpoint with global_lora_A_state")
        if global_lora_B_state is None:
            raise ValueError("FedEx-LoRA eval requires checkpoint with global_lora_B_state")
        eval_model = build_lora_model(model, args)
        load_lora_A_state(eval_model, global_lora_A_state)
        load_lora_B_state(eval_model, global_lora_B_state)
        if global_classifier_state is not None:
            load_classifier_state(eval_model, global_classifier_state)
        eval_model = eval_model.to(device)
        eval_model.eval()
    elif eval_algo == "florg":
        if global_florg_A_state is None:
            raise ValueError("FLoRG eval requires checkpoint with global_florg_A_state")
        if global_florg_seed_state is None:
            raise ValueError("FLoRG eval requires checkpoint with global_florg_seed_state")
        eval_model = build_florg_model(
            model,
            args,
            seed_state=global_florg_seed_state,
            basis_state=global_florg_basis_state,
        )
        load_florg_A_state(eval_model, global_florg_A_state)
        eval_model = eval_model.to(device)
        eval_model.eval()
        debug_layer_name, debug_delta_norm = sample_florg_delta_norm(eval_model)
        print(f"[debug][florg][eval] layer={debug_layer_name} ||deltaW||={debug_delta_norm:.6e}")
    elif eval_algo == "flora":
        # Flora checkpoints are evaluated from the saved backbone state.
        # Current training already applies global delta to backbone each round.
        eval_model = model
    elif eval_algo == "fedavg":
        # FedAvg checkpoints store the full global backbone state.
        eval_model = model

    result = None
    gsm8k_metrics = {}
    math_metrics = {}
    if args.eval_metric == "loss":
        loss_total = 0.0
        num_eval = 0
        pbar = tqdm(range(len(eval_loader)))
        with torch.inference_mode():
            for batch in eval_loader:
                batch = {
                    "input_ids": batch["input_ids"].to(device),
                    "labels": batch["labels"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                outputs = eval_model(**batch)
                loss = outputs.loss
                pbar.update(1)
                if torch.isnan(loss):
                    continue
                loss_total += loss
                num_eval += len(batch["input_ids"])
                if num_eval == 0:
                    num_eval = 1e-10
                pbar.set_description(f"eval loss: {loss_total / num_eval}")
        result = float((loss_total / num_eval).item())
        print(f"[result] eval_loss={result}")
    elif args.dataset == "gsm8k":
        sanitize_greedy_generation_config(eval_model)
        pred_texts = []
        ref_texts = []
        num_eval = 0
        running_correct = 0
        pbar = tqdm(range(len(eval_loader)))
        with torch.inference_mode():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                label_ids = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bs = input_ids.size(0)
                for i in range(bs):
                    valid_input = input_ids[i][attention_mask[i].bool()].unsqueeze(0)
                    valid_mask = torch.ones_like(valid_input, device=device)
                    output_ids = eval_model.generate(
                        input_ids=valid_input,
                        attention_mask=valid_mask,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=256,
                    )
                    prompt_len = int(valid_mask[0].sum().item())
                    pred_ids = output_ids[0][prompt_len:]
                    ref_ids = label_ids[i]
                    if ref_ids.numel() > 0:
                        ref_ids = ref_ids[ref_ids >= 0]
                    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
                    pred_texts.append(pred_text)
                    ref_texts.append(ref_text)

                    pred_final, pred_invalid = extract_gsm8k_pred_final_answer(pred_text)
                    gold_final = extract_gsm8k_gold_final_answer(ref_text)
                    if (not pred_invalid) and (pred_final is not None) and (gold_final is not None) and (pred_final == gold_final):
                        running_correct += 1
                num_eval += bs
                pbar.update(1)
                denom = float(max(num_eval, 1))
                pbar.set_description(f"eval gsm8k_acc: {running_correct / denom:.6f}")

        gsm8k_metrics = compute_gsm8k_metrics(pred_texts, ref_texts)
        if args.eval_metric == "gsm8k_acc":
            result = float(gsm8k_metrics["gsm8k_acc"])
            print(
                f"[result] gsm8k_acc={gsm8k_metrics['gsm8k_acc']:.6f}, "
                f"gsm8k_rougeL={gsm8k_metrics['gsm8k_rougeL']:.6f}, "
                f"gsm8k_invalid_rate={gsm8k_metrics['gsm8k_invalid_rate']:.6f}"
            )
        elif args.eval_metric == "rouge":
            result = float(gsm8k_metrics["gsm8k_rougeL"])
            print(
                f"[result] gsm8k_rougeL={gsm8k_metrics['gsm8k_rougeL']:.6f}, "
                f"gsm8k_acc={gsm8k_metrics['gsm8k_acc']:.6f}, "
                f"gsm8k_invalid_rate={gsm8k_metrics['gsm8k_invalid_rate']:.6f}"
            )
        else:
            raise ValueError(f"unsupported eval_metric={args.eval_metric} for dataset=gsm8k")
    elif args.dataset == "math":
        sanitize_greedy_generation_config(eval_model)
        pred_texts = []
        ref_texts = []
        gold_finals = []
        subjects = []
        levels = []
        num_eval = 0
        running_correct = 0
        pbar = tqdm(range(len(eval_loader)))
        with torch.inference_mode():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                label_ids = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bs = input_ids.size(0)
                meta_ref_solution = batch.get("meta_ref_solution", None)
                meta_final_answer = batch.get("meta_final_answer", None)
                meta_subject = batch.get("meta_subject", None)
                meta_level = batch.get("meta_level", None)
                for i in range(bs):
                    valid_input = input_ids[i][attention_mask[i].bool()].unsqueeze(0)
                    valid_mask = torch.ones_like(valid_input, device=device)
                    output_ids = eval_model.generate(
                        input_ids=valid_input,
                        attention_mask=valid_mask,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=512,
                    )
                    prompt_len = int(valid_mask[0].sum().item())
                    pred_ids = output_ids[0][prompt_len:]
                    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

                    ref_text = None
                    if isinstance(meta_ref_solution, list) and i < len(meta_ref_solution):
                        ref_text = meta_ref_solution[i]
                    if ref_text is None:
                        ref_ids = label_ids[i]
                        if ref_ids.numel() > 0:
                            ref_ids = ref_ids[ref_ids >= 0]
                        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

                    gold_final = None
                    if isinstance(meta_final_answer, list) and i < len(meta_final_answer):
                        gold_final = meta_final_answer[i]
                    subject = "unknown"
                    if isinstance(meta_subject, list) and i < len(meta_subject) and meta_subject[i] is not None:
                        subject = str(meta_subject[i])
                    level = "unknown"
                    if isinstance(meta_level, list) and i < len(meta_level) and meta_level[i] is not None:
                        level = str(meta_level[i])

                    pred_texts.append(pred_text)
                    ref_texts.append(ref_text)
                    gold_finals.append(gold_final)
                    subjects.append(subject)
                    levels.append(level)

                    pred_final, pred_invalid = extract_math_pred_final_answer(pred_text)
                    gold_final_norm = extract_math_gold_final_answer(
                        gold_final,
                        fallback_solution=ref_text,
                    )
                    if (not pred_invalid) and (pred_final is not None) and (gold_final_norm is not None) and (pred_final == gold_final_norm):
                        running_correct += 1
                num_eval += bs
                pbar.update(1)
                denom = float(max(num_eval, 1))
                pbar.set_description(f"eval math_acc: {running_correct / denom:.6f}")

        math_metrics = compute_math_metrics(
            pred_texts=pred_texts,
            ref_texts=ref_texts,
            gold_finals=gold_finals,
            subjects=subjects,
            levels=levels,
        )
        if args.eval_metric == "math_acc":
            result = float(math_metrics["math_acc"])
            print(
                f"[result] math_acc={math_metrics['math_acc']:.6f}, "
                f"math_rougeL={math_metrics['math_rougeL']:.6f}, "
                f"math_invalid_rate={math_metrics['math_invalid_rate']:.6f}"
            )
        elif args.eval_metric == "rouge":
            result = float(math_metrics["math_rougeL"])
            print(
                f"[result] math_rougeL={math_metrics['math_rougeL']:.6f}, "
                f"math_acc={math_metrics['math_acc']:.6f}, "
                f"math_invalid_rate={math_metrics['math_invalid_rate']:.6f}"
            )
        else:
            raise ValueError(f"unsupported eval_metric={args.eval_metric} for dataset=math")
    else:
        metric_total = 0.0
        num_eval = 0
        pbar = tqdm(range(len(eval_loader)))
        with torch.inference_mode():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                label_ids = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bs = input_ids.size(0)
                for i in range(bs):
                    valid_input = input_ids[i][attention_mask[i].bool()].unsqueeze(0)
                    valid_mask = torch.ones_like(valid_input, device=device)
                    output_ids = eval_model.generate(
                        input_ids=valid_input,
                        attention_mask=valid_mask,
                        pad_token_id=tokenizer.pad_token_id,
                        max_new_tokens=128,
                        num_beams=1,
                    )
                    prompt_len = int(valid_mask[0].sum().item())
                    pred_ids = output_ids[0][prompt_len:]
                    ref_ids = label_ids[i]
                    if ref_ids.numel() > 0:
                        ref_ids = ref_ids[ref_ids >= 0]
                    metric_total += rouge_score(pred_ids, ref_ids, tokenizer)
                pbar.update(1)
                num_eval += bs
                if num_eval == 0:
                    num_eval = 1e-10
                pbar.set_description(f"eval rouge: {metric_total / num_eval}")
        result = float(metric_total / num_eval)
        print(f"[result] eval_rouge={result}")

    if framework is not None:
        if eval_algo == "fedmultisubmuon":
            framework.clear_multisub_state()
        elif eval_algo == "fedstructmuon":
            framework.clear_struct_state()
        else:
            framework.clear_submuon_state()

    metrics = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "algo": eval_algo,
        "eval_metric": args.eval_metric,
        "zerotask": args.zerotask,
        "result": result,
        "eval_samples": len(eval_loader.dataset),
        "ckpt_type": ckpt_type,
    }
    if args.dataset == "gsm8k":
        metrics.update(gsm8k_metrics)
    if args.dataset == "math":
        metrics.update(math_metrics)
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[info] Saved eval json to {args.save_json}")
    return metrics


def _to_plain_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, argparse.Namespace):
        return vars(obj).copy()
    return vars(obj).copy()


def run_evaluate_from_checkpoint(
    checkpoint_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str = None,
    data_args=None,
    eval_args=None,
    device=0,
    **kwargs,
):
    """
    Load model from checkpoint_path and run evaluation using exactly this module's logic.
    tokenizer_name_or_path is accepted for compatibility; tokenizer follows model_name_or_path
    in the current data pipeline.
    """
    parser = build_parser()
    args = parser.parse_args(["--model", str(model_name_or_path)])

    data_args_dict = _to_plain_dict(data_args)
    eval_args_dict = _to_plain_dict(eval_args)
    eval_metric_explicit = (
        ("eval_metric" in data_args_dict)
        or ("eval_metric" in eval_args_dict)
        or ("eval_metric" in kwargs)
    )

    for key, value in data_args_dict.items():
        setattr(args, key, value)
    for key, value in eval_args_dict.items():
        setattr(args, key, value)
    for key, value in kwargs.items():
        setattr(args, key, value)

    args.model = model_name_or_path
    args.checkpoint = checkpoint_path
    args.device = int(device)
    if (
        tokenizer_name_or_path is not None
        and tokenizer_name_or_path != model_name_or_path
    ):
        print(
            "[warn] tokenizer_name_or_path is ignored; tokenizer follows model_name_or_path in current pipeline"
        )

    return run_evaluate(args, eval_metric_explicit=eval_metric_explicit)


def main():
    parser = build_parser()
    args = parser.parse_args()
    eval_metric_explicit = any(
        token == "--eval_metric" or token.startswith("--eval_metric=")
        for token in sys.argv[1:]
    )
    run_evaluate(args, eval_metric_explicit=eval_metric_explicit)


if __name__ == "__main__":
    main()
