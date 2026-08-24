#!/usr/bin/env python3
"""
LongBench method comparison script.

Evaluates KV cache compression methods on LongBench tasks, producing a
side-by-side summary table.

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

LongBench v1 tasks (THUDM/LongBench):
  triviaqa, qasper, trec, samsum, lcc, repobench-p, qmsum
  Metrics: F1 (qa), ROUGE-L (summarisation), classification acc, code similarity

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_longbench_comparison.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --transform_model /path/to/transform/model \\
        --methods baseline aatc \\
        --bitwidth_file /path/to/bit_alloc.pt \\
        --output_dir results/longbench

    # Limit samples for quick testing
    CUDA_VISIBLE_DEVICES=0 python run_longbench_comparison.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --methods baseline \\
        --limit 50
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["WANDB_DISABLED"] = "true"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import palu.model  # noqa: F401

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

from longbench_utils import (
    scorer as v1_scorer, DATASET2PROMPT, DATASET2MAXLEN, MODEL2MAXLEN,
)
from kivi_cache import KIVIQuantizedCache
from kvquant_cache import KVQuantizedCache, load_quantizer


# ── Constants ─────────────────────────────────────────────────────────────────

ALL_METHODS = ["baseline", "kivi", "kvquant", "palu", "palu_uniform", "aatc"]

V1_TASKS = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum"]

# Default max context length for models not in model2maxlen
DEFAULT_MAX_LENGTH = 131072  # 128k — Llama 3.1 supports this



# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_for_method(method: str, args) -> tuple:
    """Load and configure model + tokenizer for the given method."""
    if method in ("baseline", "kivi", "kvquant"):
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        model.eval()
        return model, tokenizer

    if args.transform_model is None:
        raise ValueError(f"--transform_model is required for method '{method}'.")

    model, tokenizer = load_model_and_tokenizer(args.transform_model)

    if method == "palu":
        pass  # full-precision latents, no quantization

    elif method == "palu_uniform":
        configure_latent_quantizer(model, n_bits=args.lt_bits, hadamard=args.lt_hadamard)

    elif method == "aatc":
        if args.bitwidth_file is None:
            raise ValueError("--bitwidth_file is required for method 'aatc'.")
        configure_model_quantization(
            model,
            args.bitwidth_file,
            scaling_type=args.scaling_type,
            channelwise_scaling_file=args.channelwise_scaling_file,
            channel_group_size=args.channel_group_size,
        )

    model.eval()
    return model, tokenizer


def patch_kivi(model, args):
    """Wrap model.generate to inject a fresh KIVIQuantizedCache per call."""
    num_sink = args.attention_sink_tokens
    orig = model.generate
    def _kivi_generate(*a, **kw):
        if "past_key_values" not in kw and "cache_implementation" not in kw:
            kw["past_key_values"] = KIVIQuantizedCache(
                nbits=args.kivi_bits,
                group_size=args.kivi_group_size,
                residual_length=args.kivi_residual_length,
                num_sink_tokens=num_sink,
            )
        return orig(*a, **kw)
    model.generate = _kivi_generate
    return orig


def patch_kvquant(model, args):
    """Wrap model.generate to inject a fresh KVQuantizedCache per call."""
    quantizer = load_quantizer(args.kvquant_quantizer_file)
    orig = model.generate
    def _kvquant_generate(*a, **kw):
        if "past_key_values" not in kw and "cache_implementation" not in kw:
            kw["past_key_values"] = KVQuantizedCache(
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
        return orig(*a, **kw)
    model.generate = _kvquant_generate
    return orig


def restore_generate(model, orig):
    model.generate = orig


def _get_max_length(model, model_name: str, args, max_gen: int = 0) -> int:
    """Max *prompt* length in tokens, leaving room for `max_gen` new tokens.

    Priority:
      1. --max_length, if given explicitly.
      2. LongBench's model2maxlen table. It only covers Llama-2-era models, and
         its values already bake in generation headroom (e.g. 3500 for a 4k
         model), so they are used as-is.
      3. The model's own config (`max_position_embeddings`) minus `max_gen`.
         This is the path Llama-3.x / Qwen / Mistral take — the table above has
         no entry for them, and the old code silently fell through to a fixed
         128k that left no room for the answer.
      4. DEFAULT_MAX_LENGTH minus `max_gen`, if the config has no usable value.
    """
    if args.max_length is not None:
        return args.max_length

    model_key = model_name.split("/")[-1].split("_")[0]
    if model_key in MODEL2MAXLEN:
        return MODEL2MAXLEN[model_key]

    ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not isinstance(ctx, int) or ctx <= 0:
        ctx = DEFAULT_MAX_LENGTH
    return max(ctx - max_gen, 1)


# ── Generation helpers ────────────────────────────────────────────────────────

def _truncate_prompt(tokenizer, prompt: str, max_length: int,
                     strategy: str = "middle") -> str:
    """Truncate prompt to max_length tokens.

    strategy="middle" — keep first + last half (good for v1 tasks where
                        relevant info can be anywhere).
    strategy="end"    — keep last max_length tokens (good when the
                        question/choices are appended after the context, so
                        keeping the tail preserves the most relevant context).
    """
    tokens = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokens) <= max_length:
        return prompt
    if strategy == "end":
        return tokenizer.decode(tokens[-max_length:], skip_special_tokens=True)
    # middle
    half = max_length // 2
    return (tokenizer.decode(tokens[:half], skip_special_tokens=True) +
            tokenizer.decode(tokens[-half:], skip_special_tokens=True))


def _build_chat(tokenizer, prompt: str, model_name: str) -> str:
    """Apply chat template for models that need it."""
    name_lower = model_name.lower()
    if "vicuna" in name_lower or "longchat" in name_lower:
        try:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
        except ImportError:
            pass
    if tokenizer.chat_template is not None and (
        "instruct" in name_lower or "chat" in name_lower
        or "llama-3" in name_lower or "qwen3" in name_lower
    ):
        extra = {"enable_thinking": False} if "qwen3" in name_lower else {}
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **extra,
        )
    return prompt


def _generate_with_qdc(model, tokenizer, prompt: str, max_new_tokens: int,
                       dataset: str, manager: LlamaAdaptiveQuantizationManager) -> str:
    """Generate one response, passing a fresh QuantizedDynamicCache as past_key_values.

    Required when a sliding-window/sink manager is active: without it the
    standard DynamicCache holds old 16-bit reconstructions and the manager's
    requantization steps never affect attention quality.
    """
    manager.reset_cache()
    qdc = QuantizedDynamicCache(quantized_cache=manager.quantized_cache, model=model)
    manager.past_key_values = qdc

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = inputs.input_ids.shape[-1]

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
        past_key_values=qdc,
    )
    if dataset == "samsum":
        generate_kwargs["min_length"] = context_length + 1
        generate_kwargs["eos_token_id"] = [
            tokenizer.eos_token_id,
            tokenizer.encode("\n", add_special_tokens=False)[-1],
        ]

    with torch.no_grad():
        out = model.generate(**inputs, **generate_kwargs)[0]

    return tokenizer.decode(out[context_length:], skip_special_tokens=True)


def _generate_one(model, tokenizer, prompt: str, max_new_tokens: int,
                  dataset: str, device) -> str:
    """Generate a single response."""
    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = inputs.input_ids.shape[-1]

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    # samsum: prevent endless newline repetition
    if dataset == "samsum":
        generate_kwargs["min_length"] = context_length + 1
        generate_kwargs["eos_token_id"] = [
            tokenizer.eos_token_id,
            tokenizer.encode("\n", add_special_tokens=False)[-1],
        ]

    with torch.no_grad():
        out = model.generate(**inputs, **generate_kwargs)[0]

    return tokenizer.decode(out[context_length:], skip_special_tokens=True)


# ── LongBench v1 ─────────────────────────────────────────────────────────────

def run_v1_task(model, tokenizer, task: str, args, model_name: str,
                manager: Optional[LlamaAdaptiveQuantizationManager] = None) -> float:
    """Run a single LongBench v1 task. Returns the score (0–100)."""
    max_gen = DATASET2MAXLEN[task]
    max_length = _get_max_length(model, model_name, args, max_gen)
    prompt_format = DATASET2PROMPT[task]
    device = next(model.parameters()).device

    data = load_dataset("THUDM/LongBench", task, split="test")
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))

    predictions, answers, all_classes_last = [], [], None
    for item in tqdm(data, desc=f"    {task}"):
        prompt = prompt_format.format(**item)
        # Tasks that work better without chat wrapping
        if task not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = _build_chat(tokenizer, prompt, model_name)
        prompt = _truncate_prompt(tokenizer, prompt, max_length)
        if manager is not None:
            pred = _generate_with_qdc(model, tokenizer, prompt, max_gen, task, manager)
        else:
            pred = _generate_one(model, tokenizer, prompt, max_gen, task, device)
        predictions.append(pred)
        answers.append(item["answers"])
        all_classes_last = item.get("all_classes")

    score = v1_scorer(task, predictions, answers, all_classes_last)
    return score


def run_v1(model, tokenizer, tasks: List[str], args, model_name: str,
           manager: Optional[LlamaAdaptiveQuantizationManager] = None) -> Dict[str, float]:
    """Run all requested LongBench v1 tasks. Returns {task: score}."""
    scores = {}
    for task in tasks:
        t0 = time.time()
        score = run_v1_task(model, tokenizer, task, args, model_name, manager=manager)
        elapsed = (time.time() - t0) / 60
        print(f"    {task}: {score:.1f}  ({elapsed:.1f} min)")
        scores[task] = score
    return scores


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(
    all_scores: Dict[str, Dict[str, Optional[float]]],
    tasks: List[str],
    methods: List[str],
):
    col_w = 13
    task_labels = tasks
    header = f"{'Method':<20}" + "".join(f"{t[:col_w-1]:>{col_w}}" for t in task_labels)
    sep = "=" * len(header)
    print()
    print(sep)
    print("LongBench Comparison")
    print(sep)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:<20}"
        for t in task_labels:
            s = all_scores.get(m, {}).get(t)
            row += f"{s:>{col_w}.1f}" if s is not None else f"{'N/A':>{col_w}}"
        print(row)
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LongBench method comparison"
    )

    # Methods
    parser.add_argument("--methods", nargs="+", default=["baseline"],
                        choices=ALL_METHODS)

    # Benchmarks
    parser.add_argument("--v1_tasks", nargs="+", default=V1_TASKS,
                        choices=V1_TASKS,
                        help=f"LongBench v1 tasks to run (default: all 7). "
                             f"Choices: {', '.join(V1_TASKS)}")

    # Model paths
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--transform_model", "--palu_model", dest="transform_model",
                        default=None,
                        help="Path to the decomposed checkpoint from compress.py. For aatc this is the "
                             "rank-ratio-1.0 (invertible transform, no truncation) checkpoint; for "
                             "palu/palu_uniform it is the truncated one (e.g. ratio 0.7).")

    # KIVI
    parser.add_argument("--kivi_bits", type=int, default=2)
    parser.add_argument("--kivi_group_size", type=int, default=32)
    parser.add_argument("--kivi_residual_length", type=int, default=128)

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
                        help="Ablation: quantize keys post-RoPE instead of the faithful "
                             "pre-RoPE (needs a post-RoPE-calibrated quantizer)")
    parser.add_argument("--kvquant_cap_outliers", type=float, default=-1.0,
                        help="Keys: keep a fixed fraction of outliers per token "
                             "(e.g. 0.01) instead of threshold-based (-1 = off)")

    # Uniform latent quantization
    parser.add_argument("--lt_bits", type=int, default=2)
    parser.add_argument("--lt_hadamard", action="store_true", default=True)
    parser.add_argument("--no_lt_hadamard", dest="lt_hadamard", action="store_false")

    # Per-dim quantization
    parser.add_argument("--bitwidth_file", default=None)
    parser.add_argument("--scaling_type", default="tokenwise",
                        choices=["tokenwise", "channelwise", "channel_group", "factored"])
    parser.add_argument("--channelwise_scaling_file", default=None)
    parser.add_argument("--channel_group_size", type=int, default=64)

    # Sliding window (palu_uniform / aatc)
    parser.add_argument("--recent_tokens", type=int, default=0,
                        help="Keep last N tokens at full precision for aatc (0 = disabled).")
    parser.add_argument("--attention_sink_tokens", type=int, default=0,
                        help="Keep first N tokens at full precision for aatc (0 = disabled).")
    parser.add_argument("--sliding_step_size", type=int, default=None,
                        help="Tokens requantized per sliding step (default: recent_tokens // 2).")

    # Context length
    parser.add_argument("--max_length", type=int, default=None,
                        help="Max prompt tokens for v1 tasks. Default: LongBench's "
                             "model2maxlen entry if the model has one, else the model's "
                             "own max_position_embeddings minus the task's generation "
                             "budget.")

    # Misc
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per task (quick testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/longbench")

    args = parser.parse_args()

    # Validate
    needs_palu = [m for m in args.methods if m == "aatc" or m.startswith("palu")]
    if needs_palu and args.transform_model is None:
        parser.error(f"--transform_model is required for: {needs_palu}")
    if "aatc" in args.methods and args.bitwidth_file is None:
        parser.error("--bitwidth_file is required for aatc.")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_task_labels = list(args.v1_tasks)

    print("=" * 72)
    print("LongBench Method Comparison")
    print(f"  Methods    : {args.methods}")
    print(f"  Tasks      : {args.v1_tasks}")
    if args.limit:
        print(f"  Limit      : {args.limit} samples per task")
    print("=" * 72)

    all_scores: Dict[str, Dict[str, Optional[float]]] = {}

    for method in args.methods:
        print(f"\n{'='*72}")
        print(f"Method: {method}")

        model, tokenizer = load_model_for_method(method, args)
        model_name = args.model.split("/")[-1] if method in ("baseline", "kivi", "kvquant") \
            else args.transform_model.split("/")[-1]

        orig_generate = None
        if method == "kivi":
            orig_generate = patch_kivi(model, args)
        elif method == "kvquant":
            orig_generate = patch_kvquant(model, args)

        perdim_manager = None
        if method in ("aatc", "palu_uniform") and (args.recent_tokens > 0 or args.attention_sink_tokens > 0):
            print(f"  Installing sliding-window manager: "
                  f"recent_tokens={args.recent_tokens}, "
                  f"attention_sink_tokens={args.attention_sink_tokens}")
            perdim_manager = _get_sliding_window_manager_cls(model)(
                model,
                quantized_cache=QuantizedCache(),
                recent_tokens=args.recent_tokens,
                attention_sink_tokens=args.attention_sink_tokens,
                sliding_step_size=args.sliding_step_size,
            )

        method_scores: Dict[str, Optional[float]] = {}

        print(f"  LongBench ({len(args.v1_tasks)} tasks) ...")
        v1_scores = run_v1(model, tokenizer, args.v1_tasks, args, model_name,
                           manager=perdim_manager)
        method_scores.update(v1_scores)

        if method in ("kivi", "kvquant") and orig_generate is not None:
            restore_generate(model, orig_generate)
        if perdim_manager is not None:
            perdim_manager.remove_hooks()

        all_scores[method] = method_scores

        # Save per-method results
        out_path = output_dir / f"{method}.json"
        with open(out_path, "w") as f:
            json.dump({
                "method": method,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "scores": method_scores,
            }, f, indent=2)
        print(f"  Saved → {out_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary(all_scores, all_task_labels, args.methods)

    # Save combined summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "methods": args.methods,
            "tasks": all_task_labels,
            "scores": all_scores,
        }, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
