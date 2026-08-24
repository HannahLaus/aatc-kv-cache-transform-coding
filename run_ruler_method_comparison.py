#!/usr/bin/env python3
"""
Method comparison on RULER: baseline vs KIVI vs KVQuant vs PALU vs AATC.

Shows how different KV cache quantization methods affect long-context
understanding across context lengths — the setting where the gap between a
uniform budget and AATC's allocated one is largest (paper Fig. 1).

Methods:
  baseline     — FP16 base model, no compression
  kivi         — FP16 base model + KIVI uniform KV cache quantization
  kvquant      — FP16 base model + KVQuant non-uniform quantization
  palu         — PALU-decomposed model (rank ratio < 1), full-precision latents
  palu_uniform — PALU baseline: rank ratio < 1 (e.g. 0.7) + *uniform* latent
                 quantization with Hadamard rotation
  aatc         — Attention-Aware Transform Coding (this paper): rank ratio 1.0,
                 i.e. an exactly invertible whitening transform with no rank
                 truncation, plus the *per-channel* bit allocation from
                 run_bit_allocation.py. All loss is confined to the allocation
                 step. Requires --bitwidth_file.

The palu_uniform / aatc distinction is the paper's central one: PALU compresses
by *deleting* low-energy subspaces (ratio 0.7) and then spends a flat bit budget
on what is left; AATC keeps the full rank (ratio 1.0) and instead spends a
*non-uniform* budget derived by reverse water-filling on the attention-aware
distortion.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_ruler_method_comparison.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --transform_model /path/to/transform/model \\
        --bitwidth_file /path/to/bit_alloc.pt \\
        --ruler_repo_path /path/to/RULER \\
        --context_lengths 4096 8192 16384 \\
        --kivi_bits 2 --lt_bits 2 \\
        --num_samples 500 --max_new_tokens 50
"""

import sys
import types

# Suppress broken flash_attn CUDA module from vllm conflicts
try:
    import flash_attn_2_cuda
except (ImportError, AttributeError, OSError, RuntimeError) as e:
    if any(k in str(e).lower() for k in ("undefined symbol", "flash_attn")):
        dummy = types.ModuleType("flash_attn_2_cuda")
        dummy.flash_attn_varlen_func = lambda *a, **kw: None
        dummy.flash_attn_func = lambda *a, **kw: None
        sys.modules["flash_attn_2_cuda"] = dummy

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import palu.model  # noqa: F401 — registers PaluLlama / PaluMistral / PaluQwen2

from utils import load_model_and_tokenizer
from palu.quant_utils import configure_latent_quantizer
from run_generation_with_quantization import configure_model_quantization
from palu.model.modules.svd_linear import QuantizedCache
from palu.model.modules.quantized_cache_wrapper import QuantizedDynamicCache
from palu.model.svd_llama.adaptive_quantization_llama import LlamaAdaptiveQuantizationManager
from palu.model.svd_qwen.adaptive_quantization_qwen import Qwen2AdaptiveQuantizationManager

def _get_sliding_window_manager_cls(model):
    if getattr(getattr(model, "config", None), "model_type", None) == "paluqwen2":
        return Qwen2AdaptiveQuantizationManager
    return LlamaAdaptiveQuantizationManager
from run_ruler import run_ruler_evaluation, RULER_REPO_PATH, RULER_DIRECT_AVAILABLE
from kivi_cache import KIVIQuantizedCache
from kvquant_cache import KVQuantizedCache, load_quantizer


ALL_METHODS = ["baseline", "kivi", "kvquant", "palu_uniform", "aatc"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_score(data: dict) -> Optional[float]:
    for key in ("score", "accuracy", "f1"):
        if key in data and data[key] is not None:
            return float(data[key])
    return None


def _task_scores(results: dict) -> Dict[str, float]:
    out = {}
    for task, data in results.items():
        if isinstance(data, dict):
            s = _get_score(data)
            if s is not None:
                out[task] = round(s, 1)
        elif isinstance(data, (int, float)):
            out[task] = round(float(data), 1)
    return out


def _mean(scores: dict) -> Optional[float]:
    vals = list(scores.values())
    return round(sum(vals) / len(vals), 1) if vals else None


# Task → category mapping (mirrors RULER_TASK_CATEGORIES in run_ruler.py)
TASK_CATEGORIES = {
    "niah_single_1":  "retrieval",
    "niah_single_2":  "retrieval",
    "niah_single_3":  "retrieval",
    "niah_multikey_1": "retrieval",
    "niah_multikey_2": "retrieval",
    "niah_multikey_3": "retrieval",
    "niah_multivalue": "retrieval",
    "niah_multiquery": "retrieval",
    "vt":   "multi_hop",
    "cwe":  "aggregation",
    "fwe":  "aggregation",
    "qa_1": "qa",
    "qa_2": "qa",
}
CATEGORY_ORDER = ["retrieval", "multi_hop", "aggregation", "qa"]


def _category_means(scores: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Return mean score per category for a single (method, context_length) result."""
    buckets: Dict[str, List[float]] = {c: [] for c in CATEGORY_ORDER}
    for task, s in scores.items():
        cat = TASK_CATEGORIES.get(task)
        if cat:
            buckets[cat].append(s)
    return {
        cat: (round(sum(v) / len(v), 1) if v else None)
        for cat, v in buckets.items()
    }


def _overall_mean(all_results: dict, method: str, context_lengths: List[int]) -> Optional[float]:
    """Mean across all tasks and all context lengths for one method."""
    vals = []
    for cl in context_lengths:
        vals.extend(all_results.get(method, {}).get(cl, {}).values())
    return round(sum(vals) / len(vals), 1) if vals else None


def _print_table(
    all_results: Dict[str, Dict[int, Dict[str, float]]],
    methods: List[str],
    context_lengths: List[int],
):
    """Print summary tables: mean per context length, per category, and overall."""
    W = 20 + 8 * len(context_lengths) + 10  # table width

    # ── 1. Mean score per context length + overall ────────────────────────────
    print("\n" + "=" * W)
    print("RULER Method Comparison — Mean Score (%)")
    print("=" * W)
    header = f"{'Method':<20}" + "".join(f"  {cl//1024}k".rjust(8) for cl in context_lengths) + "  overall"
    print(header)
    print("-" * W)
    for m in methods:
        row = f"{m:<20}"
        for cl in context_lengths:
            scores = all_results.get(m, {}).get(cl, {})
            mean = _mean(scores)
            row += f"  {mean:6.1f}" if mean is not None else "     N/A"
        overall = _overall_mean(all_results, m, context_lengths)
        row += f"  {overall:7.1f}" if overall is not None else "      N/A"
        print(row)
    print("=" * W)

    # ── 2. Per-category mean, averaged across all context lengths ─────────────
    print(f"\n{'Method':<20}" + "".join(f"  {c[:10]:>10}" for c in CATEGORY_ORDER))
    print("-" * (20 + 12 * len(CATEGORY_ORDER)))
    for m in methods:
        # Collect all task scores across all context lengths, then bucket by category
        all_task_scores: Dict[str, List[float]] = {}
        for cl in context_lengths:
            for task, s in all_results.get(m, {}).get(cl, {}).items():
                all_task_scores.setdefault(task, []).append(s)
        # Average each task across context lengths first, then mean within category
        avg_task = {task: sum(v) / len(v) for task, v in all_task_scores.items()}
        cat_means = _category_means(avg_task)
        row = f"{m:<20}"
        for cat in CATEGORY_ORDER:
            s = cat_means.get(cat)
            row += f"  {s:10.1f}" if s is not None else "       N/A"
        print(row)

    # ── 3. Per-task breakdown per context length ──────────────────────────────
    for cl in context_lengths:
        print(f"\n--- {cl // 1024}k context ---")
        tasks: List[str] = []
        for m in methods:
            for t in all_results.get(m, {}).get(cl, {}):
                if t not in tasks:
                    tasks.append(t)
        if not tasks:
            print("  (no results)")
            continue
        header2 = f"{'Task':<28}{'Category':<14}" + "".join(f"  {m[:12]:>12}" for m in methods)
        print(header2)
        print("-" * (42 + 14 * len(methods)))
        for task in sorted(tasks):
            cat = TASK_CATEGORIES.get(task, "")
            row2 = f"{task:<28}{cat:<14}"
            for m in methods:
                s = all_results.get(m, {}).get(cl, {}).get(task)
                row2 += f"  {s:12.1f}" if s is not None else "         N/A"
            print(row2)


# ── Per-method model loading ───────────────────────────────────────────────────

def load_model_for_method(method: str, args) -> tuple:
    """Load and configure model+tokenizer for the given method."""

    if method in ("baseline", "kivi", "kvquant"):
        logger.info(f"  Loading base model ({args.model}) ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        model.eval()
        return model, tokenizer

    # palu_uniform / aatc — need the compressed checkpoint
    if args.transform_model is None:
        raise ValueError(f"--transform_model is required for method '{method}'.")

    logger.info(f"  Loading PALU model ({args.transform_model}) ...")
    model, tokenizer = load_model_and_tokenizer(args.transform_model)

    if method == "palu_uniform":
        logger.info(f"  Configuring uniform quantization: {args.lt_bits}-bit, "
                    f"hadamard={args.lt_hadamard}")
        configure_latent_quantizer(model, n_bits=args.lt_bits, hadamard=args.lt_hadamard)

    elif method == "aatc":
        if args.bitwidth_file is None:
            raise ValueError("--bitwidth_file is required for method 'aatc'.")
        logger.info(f"  Configuring per-dim quantization from {args.bitwidth_file}")
        configure_model_quantization(
            model,
            args.bitwidth_file,
            scaling_type=args.scaling_type,
            channelwise_scaling_file=args.channelwise_scaling_file,
            channel_group_size=args.channel_group_size,
        )

    model.eval()
    return model, tokenizer


def make_kivi_hook(args) -> callable:
    """Return a per-sample hook that creates a fresh KIVIQuantizedCache."""
    num_sink = args.attention_sink_tokens

    def _hook():
        return KIVIQuantizedCache(
            nbits=args.kivi_bits,
            group_size=args.kivi_group_size,
            residual_length=args.kivi_residual_length,
            num_sink_tokens=num_sink,
        )

    return _hook


def make_kvquant_hook(args) -> callable:
    """Return a per-sample hook that creates a fresh KVQuantizedCache."""
    quantizer = load_quantizer(args.kvquant_quantizer_file)

    def _hook():
        return KVQuantizedCache(
            nbits=args.kvquant_bits,
            quantizer=quantizer,
            include_sparse=not args.kvquant_no_sparse,
            sparsity_threshold=args.kvquant_sparsity_threshold,
            num_sink_tokens=args.kvquant_sink_tokens,
            residual_length=args.kvquant_residual_length,
            group_size=args.kvquant_group_size,
            pre_rope=not args.kvquant_post_rope,
            cap_outliers=args.kvquant_cap_outliers,
        )

    return _hook


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RULER method comparison: baseline / KIVI / palu_uniform / aatc"
    )

    # Methods
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS,
                        help="Methods to compare (default: all four)")

    # Model paths
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID for baseline / kivi")
    parser.add_argument("--transform_model", "--palu_model", dest="transform_model",
                        default=None,
                        help="Path to the decomposed checkpoint from compress.py. For aatc this is the "
                             "rank-ratio-1.0 (invertible transform, no truncation) checkpoint; for "
                             "palu/palu_uniform it is the truncated one (e.g. ratio 0.7).")
    parser.add_argument("--tokenizer_path", default=None,
                        help="Local path to tokenizer files for RULER sample generation. "
                             "Required when --model is a HuggingFace Hub ID and the model is not "
                             "cached locally (e.g. --tokenizer_path /path/to/Llama-3.1-8B-Instruct).")

    # KIVI
    parser.add_argument("--kivi_bits", type=int, default=2,
                        help="KV cache quantization bitwidth for KIVI (default: 2)")
    parser.add_argument("--kivi_group_size", type=int, default=32)
    parser.add_argument("--kivi_residual_length", type=int, default=128,
                        help="Recent tokens kept at full precision in KIVI (default: 128)")

    # KVQuant (pure PyTorch, no CUDA kernel)
    parser.add_argument("--kvquant_bits", type=int, default=4, choices=[2, 3, 4, 5])
    parser.add_argument("--kvquant_quantizer_file", default=None,
                        help="Calibrated NUQ quantizer .pt from run_kvquant_calibration.py "
                             "(omit for NormalFloat fallback)")
    parser.add_argument("--kvquant_sparsity_threshold", type=float, default=0.99)
    parser.add_argument("--kvquant_no_sparse", action="store_true")
    parser.add_argument("--kvquant_sink_tokens", type=int, default=5)
    parser.add_argument("--kvquant_residual_length", type=int, default=0)
    parser.add_argument("--kvquant_group_size", type=int, default=32)
    parser.add_argument("--kvquant_post_rope", action="store_true",
                        help="Ablation: quantize keys post-RoPE instead of pre-RoPE")
    parser.add_argument("--kvquant_cap_outliers", type=float, default=-1.0,
                        help="Keys: fixed outlier fraction per token (-1 = off)")

    # Uniform latent quantization (palu_uniform)
    parser.add_argument("--lt_bits", type=int, default=2,
                        help="Latent bitwidth for palu_uniform (default: 2)")
    parser.add_argument("--lt_hadamard", action="store_true", default=True,
                        help="Apply Hadamard rotation before quantization (default: on)")
    parser.add_argument("--no_lt_hadamard", dest="lt_hadamard", action="store_false")

    # Per-dim quantization (aatc)
    parser.add_argument("--bitwidth_file", default=None,
                        help="Per-dim bitwidth allocation file (.pt)")
    parser.add_argument("--scaling_type", default="tokenwise",
                        choices=["tokenwise", "channelwise", "channel_group", "factored"])
    parser.add_argument("--channelwise_scaling_file", default=None,
                        help="Per-channel calibration file (.pt). Required for channelwise / "
                             "channel_group / factored scaling.")
    parser.add_argument("--channel_group_size", type=int, default=64)

    # Sliding window (palu_uniform / aatc)
    parser.add_argument("--recent_tokens", type=int, default=0,
                        help="Keep last N tokens at full 16-bit precision for aatc "
                             "(0 = disabled). Uses the sliding-window manager; cache is reset "
                             "between RULER samples.")
    parser.add_argument("--attention_sink_tokens", type=int, default=0,
                        help="Keep first N tokens (attention sinks) at full 16-bit precision "
                             "for aatc (0 = disabled).")
    parser.add_argument("--sliding_step_size", type=int, default=None,
                        help="Tokens requantized per sliding step (default: recent_tokens // 2). "
                             "E.g. --recent_tokens 128 --sliding_step_size 16 keeps 112-128 tokens "
                             "at full precision, compressing 16 at a time.")

    # Context lengths
    parser.add_argument("--context_lengths", nargs="+", type=int,
                        default=[4096, 8192, 16384],
                        help="Context lengths to evaluate (default: 4096 8192 16384)")

    # RULER
    parser.add_argument("--ruler_repo_path", default=None,
                        help="Path to the RULER repository")
    parser.add_argument("--root_dir", default=None,
                        help="Root dir for RULER intermediate outputs")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Specific RULER tasks (default: all)")
    parser.add_argument("--task_category", default=None,
                        choices=["retrieval", "multi_hop", "aggregation", "qa", "all"])
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Samples per task (default 500; use 10-50 for quick tests)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per task (quick testing only)")
    parser.add_argument("--max_new_tokens", type=int, default=50)

    # Output
    parser.add_argument("--output_dir", default="results/ruler_method_comparison",
                        help="Directory to save results JSON files")

    args = parser.parse_args()

    # Validate
    ruler_repo = args.ruler_repo_path or RULER_REPO_PATH
    if not ruler_repo:
        parser.error("RULER repository not found. Use --ruler_repo_path or set RULER_REPO_PATH.")

    needs_palu = [m for m in args.methods if m in ("palu_uniform", "aatc")]
    if needs_palu and args.transform_model is None:
        parser.error(f"--transform_model is required for methods: {needs_palu}")

    needs_bitwidth = [m for m in args.methods if m == "aatc"]
    if needs_bitwidth and args.bitwidth_file is None:
        parser.error("--bitwidth_file is required for method 'aatc'.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = args.root_dir or str(output_dir / "ruler_outputs")

    logger.info("=" * 76)
    logger.info("RULER Method Comparison")
    logger.info(f"  Methods         : {args.methods}")
    logger.info(f"  Context lengths : {args.context_lengths}")
    logger.info(f"  KIVI bits       : {args.kivi_bits} (residual_length={args.kivi_residual_length})")
    logger.info(f"  Uniform bits    : {args.lt_bits} (hadamard={args.lt_hadamard})")
    logger.info(f"  Bitwidth file   : {args.bitwidth_file}")
    logger.info("=" * 76)

    # all_results[method][context_length] = {task: score, ...}
    all_results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for method in args.methods:
        logger.info("=" * 76)
        logger.info(f"Method: {method}")

        model, tokenizer = load_model_for_method(method, args)

        extra_kwargs = None

        # For aatc with a sliding window: install the sliding-window manager
        # so recent/sink tokens are kept at full precision. The manager's cache
        # must be reset before each RULER sample (passed as pre_generate_hook).
        perdim_manager = None
        pre_hook = None
        if method == "kivi":
            pre_hook = make_kivi_hook(args)
        elif method == "kvquant":
            pre_hook = make_kvquant_hook(args)
        elif method in ("aatc", "palu_uniform") and (args.recent_tokens > 0 or args.attention_sink_tokens > 0):
            logger.info(f"  Installing sliding-window manager: "
                        f"recent_tokens={args.recent_tokens}, "
                        f"attention_sink_tokens={args.attention_sink_tokens}")
            perdim_manager = _get_sliding_window_manager_cls(model)(
                model,
                quantized_cache=QuantizedCache(),
                recent_tokens=args.recent_tokens,
                attention_sink_tokens=args.attention_sink_tokens,
                sliding_step_size=args.sliding_step_size,
            )

            def _make_qdc_hook(_manager=perdim_manager, _model=model):
                """Reset cache, create a fresh QuantizedDynamicCache, and return it.

                run_ruler_evaluation passes the returned value as past_key_values to
                model.generate() so that the sliding step's requantized latents are
                reconstructed on-the-fly during attention (without this, the sliding
                step only updates QuantizedCache but the standard DynamicCache keeps
                old 16-bit reconstructions).
                """
                _manager.reset_cache()
                qdc = QuantizedDynamicCache(
                    quantized_cache=_manager.quantized_cache,
                    model=_model,
                )
                _manager.past_key_values = qdc
                return qdc

            pre_hook = _make_qdc_hook

        all_results[method] = {}

        for ctx_len in args.context_lengths:
            logger.info(f"  Context length: {ctx_len}")
            run_root = os.path.join(root_dir, f"{method}_{ctx_len}")

            try:
                # RULER generates samples of exactly ctx_len tokens.
                # generate_response has an internal buffer of 100 tokens on top
                # of max_new_tokens, which would silently cut that many tokens
                # from the left of the context.  Pass generate_max_length =
                # ctx_len + max_new_tokens + 100 so the model sees the full
                # ctx_len tokens of context while still having room to generate.
                results = run_ruler_evaluation(
                    model=model,
                    tokenizer=tokenizer,
                    model_path=args.transform_model if method in ("palu_uniform", "aatc") else args.model,
                    ruler_repo_path=ruler_repo,
                    root_dir=run_root,
                    tasks=args.tasks,
                    task_category=args.task_category,
                    max_length=ctx_len,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    limit=args.limit,
                    num_samples=args.num_samples,
                    extra_generate_kwargs=extra_kwargs,
                    generate_max_length=ctx_len + args.max_new_tokens + 100,
                    pre_generate_hook=pre_hook,
                    tokenizer_path=args.tokenizer_path,
                )
                task_scores = _task_scores(results)
                all_results[method][ctx_len] = task_scores

                run_file = output_dir / f"{method}_{ctx_len}_{timestamp}.json"
                with open(run_file, "w") as f:
                    json.dump({"method": method, "context_length": ctx_len,
                               "scores": task_scores, "raw": results}, f, indent=2)
                logger.info(f"    Mean: {_mean(task_scores)} — saved to {run_file}")

            except Exception as e:
                logger.error(f"  Failed {method} @ {ctx_len}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                all_results[method][ctx_len] = {}

        # Clean up sliding-window manager hooks before freeing the model
        if perdim_manager is not None:
            perdim_manager.remove_hooks()
            perdim_manager = None

        # Free GPU memory before loading next method's model
        del model
        torch.cuda.empty_cache()

    # Save combined summary
    summary_file = output_dir / f"summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nFull summary saved to {summary_file}")

    _print_table(all_results, args.methods, args.context_lengths)


if __name__ == "__main__":
    main()
