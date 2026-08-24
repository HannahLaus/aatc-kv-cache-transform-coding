#!/usr/bin/env python3
"""
GSM8K + MMLU-Pro evaluator for KV cache quantization methods.

Evaluates reasoning (GSM8K) and knowledge (MMLU-Pro) benchmarks across
compression methods, using lm-evaluation-harness as the backend.

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

Benchmarks:
  gsm8k        — 8-shot chain-of-thought, grade school math (flexible-extract accuracy)
  mmlu_pro     — 5-shot multiple choice across 14 subjects (loglikelihood accuracy)
  math_500     — 0-shot CoT, 500 competition math problems, boxed-answer exact match
  mmlu_pro_gen — MMLU-Pro evaluated via generation (model.generate); activates QDC
                 so aatc's recent-tokens window behaves identically to GSM8K

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_gsm8k_mmlu.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --transform_model /path/to/transform/model \\
        --methods baseline aatc \\
        --bitwidth_file /path/to/bit_alloc.pt \\
        --benchmarks gsm8k mmlu_pro \\
        --output_dir results/gsm8k_mmlu

    # Quick test with limited samples
    CUDA_VISIBLE_DEVICES=0 python run_gsm8k_mmlu.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --methods baseline \\
        --benchmarks gsm8k \\
        --limit 100
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

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
from kivi_cache import KIVIQuantizedCache
from kvquant_cache import KVQuantizedCache, load_quantizer


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

ALL_METHODS = ["baseline", "kivi", "kvquant", "palu", "palu_uniform", "aatc"]

MMLU_PRO_SUBJECTS = [
    "math", "physics", "chemistry", "biology", "computer_science",
    "law", "economics", "engineering", "psychology", "history",
    "health", "philosophy", "business", "other",
]

# lm-eval task names and their default few-shot settings (for documentation)
BENCHMARK_TASKS = {
    "gsm8k":        "gsm8k",        # 8-shot CoT, flexible-extract accuracy
    "mmlu_pro":     "mmlu_pro",     # 5-shot loglikelihood, accuracy across 14 subjects
    "math_500":     "math_500",     # 0-shot CoT, boxed-answer exact match
    "mmlu_pro_gen": "mmlu_pro_gen", # MMLU-Pro via generation (activates QDC for palu*)
}

# Metrics to extract per task (primary metric first)
TASK_METRICS = {
    "gsm8k":    ["exact_match,flexible-extract", "exact_match,strict-match", "acc"],
    "mmlu_pro": ["acc,none", "acc", "acc_norm"],
    "math_500": ["exact_match,none", "exact_match,flexible-extract",
                 "exact_match,strict-match", "acc"],
}


def _get_task_index() -> set:
    """Return the set of task names registered in this lm-eval installation."""
    try:
        from lm_eval.tasks import TaskManager
        tm = TaskManager()
        return set(tm.task_index.keys()) if hasattr(tm, "task_index") else set()
    except Exception:
        return set()


def _resolve_subject_tasks(prefix: str, subjects: List[str]) -> List[str]:
    """Resolve lm-eval task names for a benchmark's subjects.

    Tries '<prefix>_<subject>' and '<prefix>/<subject>' against the installed
    task index.  Raises a clear error listing available tasks if none match.
    """
    index = _get_task_index()

    resolved = []
    for subject in subjects:
        candidates = [f"{prefix}_{subject}", f"{prefix}/{subject}"]
        matched = next((c for c in candidates if c in index), None)
        if matched:
            resolved.append(matched)
        elif index:
            available = sorted(k for k in index if prefix in k)
            raise ValueError(
                f"Could not find subject '{subject}' for benchmark '{prefix}'.\n"
                f"Tried: {candidates}\n"
                f"Available '{prefix}' tasks in this lm-eval installation:\n"
                + "\n".join(f"  {t}" for t in available[:40])
            )
        else:
            # task_index unavailable — pass through and let lm-eval error
            resolved.append(f"{prefix}_{subject}")

    return resolved


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_for_method(method: str, args) -> tuple:
    """Load and configure model + tokenizer for the given method.

    For KIVI, returns (model, tokenizer, kivi_kwargs) where kivi_kwargs should
    be injected into model.generate via a wrapper (see patch_kivi below).
    For all other methods, applies quantization in-place and returns
    (model, tokenizer, None).
    """
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

    if args.transform_model is None:
        raise ValueError(f"--transform_model is required for method '{method}'.")

    logger.info(f"  Loading PALU model ({args.transform_model}) ...")
    model, tokenizer = load_model_and_tokenizer(args.transform_model)

    if method == "palu":
        pass  # no quantization — full-precision latents

    elif method == "palu_uniform":
        logger.info(f"  Uniform latent quantization: {args.lt_bits}-bit, "
                    f"hadamard={args.lt_hadamard}")
        configure_latent_quantizer(model, n_bits=args.lt_bits, hadamard=args.lt_hadamard)

    elif method == "aatc":
        if args.bitwidth_file is None:
            raise ValueError("--bitwidth_file is required for method 'aatc'.")
        logger.info(f"  Per-dim quantization from {args.bitwidth_file}")
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
    residual_length = (
        args.kivi_residual_length if args.kivi_residual_length is not None
        else (args.recent_tokens if args.recent_tokens > 0 else 128)
    )
    num_sink = args.attention_sink_tokens
    orig_generate = model.generate
    _call_count = [0]

    def _kivi_generate(*a, **kw):
        _call_count[0] += 1
        print(f"  [KIVI] model.generate call #{_call_count[0]}", flush=True)
        if "past_key_values" not in kw and "cache_implementation" not in kw:
            kw["past_key_values"] = KIVIQuantizedCache(
                nbits=args.kivi_bits,
                group_size=args.kivi_group_size,
                residual_length=residual_length,
                num_sink_tokens=num_sink,
            )
        return orig_generate(*a, **kw)

    model.generate = _kivi_generate
    return orig_generate, _call_count


def patch_palu_qdc(model, manager):
    """Wrap model.generate to inject a fresh QuantizedDynamicCache per call.

    Mirrors patch_kivi: resets the manager's quantized cache and creates a new
    QDC on every model.generate() call so each lm-eval sample (or MATH-500 item)
    starts from a clean state with the sliding window/recent-token window active.
    MMLU-Pro is loglikelihood-scored (no incremental cache), so the patch is no-op there.
    """
    orig_generate = model.generate

    def _qdc_generate(*a, **kw):
        if "past_key_values" not in kw and "cache_implementation" not in kw:
            manager.reset_cache()
            qdc = QuantizedDynamicCache(quantized_cache=manager.quantized_cache, model=model)
            manager.past_key_values = qdc
            kw["past_key_values"] = qdc
        return orig_generate(*a, **kw)

    model.generate = _qdc_generate
    return orig_generate


def patch_kvquant(model, args):
    """Wrap model.generate to inject a fresh KVQuantizedCache per call.

    Unlike KIVI (patched only around the lm-eval block), this patch is kept
    active for the whole method iteration so KVQuant also covers the
    generate-based tasks (MATH-500, MMLU-Pro-gen).
    """
    quantizer = load_quantizer(args.kvquant_quantizer_file)
    orig_generate = model.generate

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
        return orig_generate(*a, **kw)

    model.generate = _kvquant_generate
    return orig_generate


def restore_generate(model, orig_generate):
    model.generate = orig_generate


# ── lm-eval runner ────────────────────────────────────────────────────────────

def run_lm_eval(model, tokenizer, tasks: List[str], args, num_fewshot: Optional[int] = None) -> dict:
    """Run lm-evaluation-harness on the given tasks and return results dict."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    model.seqlen = args.max_length

    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        add_bos_token=False,
        batch_size=args.batch_size,
    )

    # Resolve TaskManager (API differs across lm-eval versions)
    task_manager = None
    try:
        from lm_eval.tasks import TaskManager
        task_manager = TaskManager()
    except (ImportError, AttributeError):
        pass

    with torch.no_grad():
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=tasks,
            task_manager=task_manager,
            num_fewshot=num_fewshot,  # None = use each task's built-in default
            log_samples=False,
            limit=args.limit,
        )

    print(make_table(results))
    return results["results"]


# ── Metric extraction ─────────────────────────────────────────────────────────

def _extract_primary_metric(task_results: dict) -> Optional[float]:
    """Extract the primary accuracy-like metric from one task's result dict.

    lm-eval versions differ in how they store metrics:
      - Newer: flat keys like 'acc,none' or 'exact_match,flexible-extract'
      - Older: nested dicts like {'acc': {'none': 0.68}}

    We prefer flexible-extract > strict-match > acc > acc_norm and skip
    stderr entries.
    """
    # Priority order for metric name prefixes
    preferred = ["exact_match,flexible-extract", "exact_match,strict-match",
                 "acc,none", "acc", "acc_norm,none", "acc_norm"]

    # Check flat keys first (newer lm-eval)
    for key in preferred:
        if key in task_results:
            val = task_results[key]
            if isinstance(val, (int, float)):
                return float(val)

    # Fallback: scan all keys, skip stderr, pick first numeric acc-like value
    for key, val in task_results.items():
        if "stderr" in key:
            continue
        if any(m in key for m in ("acc", "exact_match")):
            if isinstance(val, (int, float)):
                return float(val)
            # Older nested format: {'acc': {'none': 0.68}}
            if isinstance(val, dict):
                for v in val.values():
                    if isinstance(v, (int, float)):
                        return float(v)

    return None


def _collect_scores(raw_results: dict, benchmark: str) -> Optional[float]:
    """Given lm-eval results dict, find and return the primary metric for benchmark.

    Handles both group tasks (e.g. 'mmlu_pro' returning an aggregate) and
    individual subject tasks (e.g. 'mmlu_pro_math').  When the group
    key is absent, averages over all matching subtask entries.
    """
    lm_task = BENCHMARK_TASKS.get(benchmark, benchmark)

    # Direct key — group aggregate or exact subject task name
    if lm_task in raw_results:
        return _extract_primary_metric(raw_results[lm_task])

    # Collect subtask entries whose key starts with lm_task
    subtask_scores = []
    for key, val in raw_results.items():
        if key == lm_task or key.startswith(lm_task + "_") or key.startswith(lm_task + "/"):
            if isinstance(val, dict):
                s = _extract_primary_metric(val)
                if s is not None:
                    subtask_scores.append(s)

    if subtask_scores:
        return sum(subtask_scores) / len(subtask_scores)

    return None


# ── MATH-500 custom evaluation ───────────────────────────────────────────────

MATH500_SYSTEM = (
    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"
)

# Simple 0-shot CoT prompt — default, works well for smaller models (e.g. 8B)
MATH500_USER_SIMPLE = (
    "Solve the following math problem step by step. "
    "Put your final answer within $\\boxed{{}}$.\n\n"
    "Problem: {problem}\n\nSolution:"
)

# Structured prompt from the KVTC paper (arXiv 2511.01815) — tuned for Llama 3.3 70B
MATH500_USER_STRUCTURED = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "- For simple problems (2 steps or fewer):\n"
    "Provide a concise solution with minimal explanation.\n\n"
    "- For complex problems (3 steps or more):\n"
    "Use this step-by-step format:\n\n"
    "## Step 1: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "...\n\n"
    "Regardless of the approach, always conclude with:\n\n"
    "Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n\n"
    "Where [answer] is just the final number or expression that solves the problem.\n\n"
    "Problem: {problem}"
)

MATH500_DATASET = "HuggingFaceH4/MATH-500"


def _extract_boxed(text: str) -> str:
    """Extract the last \\boxed{{...}} answer from generated text.

    Mirrors the paper's extraction: split on \\boxed{{ and take the part
    before the closing }}$.
    """
    if r"\boxed{" not in text:
        return ""
    answer = text.split(r"\boxed{")[-1].split("}$")[0].strip()
    return answer


def _normalize_answer(answer: str) -> str:
    """Normalise a LaTeX answer string — matches the paper's answer_normalize."""
    answer = answer.replace(r"\left", "")
    answer = answer.replace(r"\right", "")
    answer = answer.replace(r"\begin{align}", "")
    answer = answer.replace(r"\end{align}", "")
    answer = answer.replace(r"\begin{equation}", "")
    answer = answer.replace(r"\end{equation}", "")
    answer = answer.replace(" ", "")
    answer = answer.replace(r"\$", "")
    if answer.startswith(r"\text"):
        answer = answer.replace(r"\text{", "").replace(r"}", "")
    if answer.startswith(r"x\in"):
        answer = answer.replace(r"x\in", "")
    if answer.startswith(r"y="):
        answer = answer.replace(r"y=", "")
    return answer


def _answers_equal(gold: str, model_answer: str) -> bool:
    """Compare gold and predicted answers using math_verify with normalisation fallback.

    Matches the paper's compare_answers function exactly.
    """
    try:
        from math_verify import parse, verify

        gold_parsed = parse(gold)
        model_parsed = parse(model_answer)
        res = verify(gold_parsed, model_parsed)
        if not res:
            gold_n = _normalize_answer(gold)
            model_n = _normalize_answer(model_answer)
            gold_parsed2 = parse(gold_n)
            model_parsed2 = parse(model_n)
            if not verify(gold_parsed2, model_parsed2):
                res = verify(gold, model_answer)
            else:
                res = True
        return bool(res)
    except ImportError:
        # math_verify not installed — fall back to normalised string match
        logger.warning("math_verify not installed; using string match. "
                       "Install with: pip install math-verify")
        return _normalize_answer(gold) == _normalize_answer(model_answer)


def load_math500(limit: Optional[int] = None) -> List[dict]:
    """Load MATH-500 from HuggingFace (HuggingFaceH4/MATH-500, test split)."""
    from datasets import load_dataset
    ds = load_dataset(MATH500_DATASET, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [{"problem": ex["problem"], "answer": ex["answer"],
             "subject": ex.get("subject", ""), "level": ex.get("level", "")}
            for ex in ds]


def _generate_with_qdc(model, inputs, prompt_len: int, max_new_tokens: int,
                       eos_token_id: int,
                       manager: LlamaAdaptiveQuantizationManager) -> "torch.Tensor":
    """Generate one response using a fresh QuantizedDynamicCache.

    Required when a sliding-window/sink manager is active: without passing
    past_key_values=qdc the standard DynamicCache holds stale 16-bit
    reconstructions and the manager's requantization never affects attention.
    Returns the generated token ids (excluding the prompt).
    """
    manager.reset_cache()
    qdc = QuantizedDynamicCache(quantized_cache=manager.quantized_cache, model=model)
    manager.past_key_values = qdc

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=eos_token_id,
            past_key_values=qdc,
        )
    return out[0][prompt_len:]


def run_math500(model, tokenizer, args,
                manager: Optional[LlamaAdaptiveQuantizationManager] = None) -> dict:
    """Evaluate model on MATH-500 with 0-shot CoT and greedy decoding.

    Returns a results dict with 'accuracy', 'n', and per-subject breakdowns.
    """
    problems = load_math500(limit=args.limit)
    prompt_style = "structured (paper)" if args.math500_structured_prompt else "simple (default)"
    logger.info(f"  MATH-500: {len(problems)} problems, 0-shot CoT, "
                f"prompt={prompt_style}, max_new_tokens={args.math500_max_new_tokens}")

    device = next(model.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    correct = 0
    subject_correct: Dict[str, int] = {}
    subject_total: Dict[str, int] = {}
    records = []

    for ex in tqdm(problems, desc="  MATH-500"):
        template = MATH500_USER_STRUCTURED if args.math500_structured_prompt else MATH500_USER_SIMPLE
        user_msg = template.format(problem=ex["problem"])
        messages = [
            {"role": "system", "content": MATH500_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]
        if tokenizer.chat_template is not None:
            _is_qwen3 = "qwen3" in getattr(tokenizer, "name_or_path", "").lower()
            _ct_extra = {"enable_thinking": False} if _is_qwen3 else {}
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **_ct_extra,
            )
        else:
            formatted = f"{MATH500_SYSTEM}\n\n{user_msg}"

        inputs = tokenizer(formatted, return_tensors="pt",
                           truncation=True, max_length=2048).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        if manager is not None:
            gen_ids = _generate_with_qdc(model, inputs, prompt_len,
                                         args.math500_max_new_tokens,
                                         tokenizer.eos_token_id, manager)
        else:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.math500_max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids = out[0][prompt_len:]

        generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_boxed = _extract_boxed(generated)
        is_correct = _answers_equal(ex["answer"], pred_boxed)

        correct += int(is_correct)
        subj = ex.get("subject", "unknown")
        subject_correct[subj] = subject_correct.get(subj, 0) + int(is_correct)
        subject_total[subj] = subject_total.get(subj, 0) + 1

        records.append({
            "problem": ex["problem"],
            "gold": ex["answer"],
            "pred_boxed": pred_boxed,
            "generated": generated,
            "correct": is_correct,
            "subject": subj,
            "level": ex.get("level", ""),
        })

    n = len(problems)
    accuracy = correct / n if n > 0 else 0.0
    logger.info(f"  MATH-500 accuracy: {accuracy * 100:.1f}% ({correct}/{n})")

    by_subject = {
        subj: {"correct": subject_correct[subj], "total": subject_total[subj],
               "acc": subject_correct[subj] / subject_total[subj]}
        for subj in subject_total
    }
    return {"accuracy": accuracy, "correct": correct, "n": n,
            "by_subject": by_subject, "records": records}


# ── MMLU-Pro generation-based evaluation ──────────────────────────────────────

MMLU_PRO_GEN_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_ANSWER_CHOICES = list("ABCDEFGHIJ")
MMLU_PRO_N_SHOT = 5
MMLU_PRO_GEN_MAX_NEW_TOKENS = 32


def _format_mmlu_pro_question(question: str, options: List[str]) -> str:
    lines = [f"Question: {question}"]
    for i, opt in enumerate(options):
        lines.append(f"{MMLU_PRO_ANSWER_CHOICES[i]}. {opt}")
    return "\n".join(lines)


def _extract_mmlu_pro_answer(text: str) -> str:
    import re
    m = re.search(r'[Aa]nswer[:\s]+([A-J])', text)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-J])\b', text)
    if m:
        return m.group(1)
    for ch in text.strip():
        if ch.upper() in set(MMLU_PRO_ANSWER_CHOICES):
            return ch.upper()
    return ""


def load_mmlu_pro_gen(subjects: Optional[List[str]] = None,
                      limit: Optional[int] = None):
    """Load MMLU-Pro test + validation splits."""
    from datasets import load_dataset
    test_ds = load_dataset(MMLU_PRO_GEN_DATASET, split="test")
    val_ds  = load_dataset(MMLU_PRO_GEN_DATASET, split="validation")
    if subjects:
        test_ds = test_ds.filter(lambda x: x["category"] in subjects)
        val_ds  = val_ds.filter(lambda x: x["category"] in subjects)
    if limit:
        test_ds = test_ds.select(range(min(limit, len(test_ds))))
    return list(test_ds), list(val_ds)


def run_mmlu_pro_gen(model, tokenizer, args,
                     manager: Optional[LlamaAdaptiveQuantizationManager] = None) -> dict:
    """Evaluate MMLU-Pro via model.generate() (5-shot) instead of loglikelihood.

    Because model.generate() is used, the QuantizedDynamicCache patch is active
    for aatc, so recent-tokens and attention-sink windows behave exactly
    as during GSM8K evaluation — unlike the loglikelihood path which never calls
    model.generate() and therefore never activates the quantized cache.
    """
    test_samples, val_samples = load_mmlu_pro_gen(
        subjects=args.mmlu_pro_subjects,
        limit=args.limit,
    )
    n = len(test_samples)
    logger.info(
        f"  MMLU-Pro (generation): {n} samples, {MMLU_PRO_N_SHOT}-shot, "
        f"max_new_tokens={MMLU_PRO_GEN_MAX_NEW_TOKENS}"
    )

    device = next(model.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    correct = 0
    by_category: Dict[str, Dict] = {}
    records = []

    for ex in tqdm(test_samples, desc="  MMLU-Pro gen"):
        category = ex["category"]
        gold = ex["answer"]  # letter "A"–"J"

        system_msg = (
            f"The following are multiple choice questions (with answers) about "
            f"{category}. Answer with just the letter of the correct answer "
            f"(A, B, C, D, E, F, G, H, I, or J)."
        )
        fewshot = [e for e in val_samples if e["category"] == category][:MMLU_PRO_N_SHOT]
        test_q = _format_mmlu_pro_question(ex["question"], ex["options"])

        if tokenizer.chat_template is not None:
            messages = [{"role": "system", "content": system_msg}]
            for fs in fewshot:
                messages.append({"role": "user",
                                  "content": _format_mmlu_pro_question(
                                      fs["question"], fs["options"])})
                messages.append({"role": "assistant", "content": fs["answer"]})
            messages.append({"role": "user", "content": test_q})
            _is_qwen3 = "qwen3" in getattr(tokenizer, "name_or_path", "").lower()
            _ct_extra = {"enable_thinking": False} if _is_qwen3 else {}
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **_ct_extra,
            )
        else:
            parts = [system_msg, ""]
            for fs in fewshot:
                parts.append(_format_mmlu_pro_question(fs["question"], fs["options"]))
                parts.append(f"Answer: {fs['answer']}")
                parts.append("")
            parts.append(test_q)
            parts.append("Answer:")
            formatted = "\n".join(parts)

        inputs = tokenizer(
            formatted, return_tensors="pt",
            truncation=True, max_length=args.max_length,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        if manager is not None:
            gen_ids = _generate_with_qdc(
                model, inputs, prompt_len,
                MMLU_PRO_GEN_MAX_NEW_TOKENS, tokenizer.eos_token_id, manager,
            )
        else:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MMLU_PRO_GEN_MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids = out[0][prompt_len:]

        generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred = _extract_mmlu_pro_answer(generated)
        is_correct = (pred == gold)

        correct += int(is_correct)
        cat_stats = by_category.setdefault(category, {"correct": 0, "total": 0})
        cat_stats["correct"] += int(is_correct)
        cat_stats["total"] += 1

        records.append({
            "question": ex["question"],
            "gold": gold,
            "pred": pred,
            "generated": generated,
            "correct": is_correct,
            "category": category,
        })

    accuracy = correct / n if n > 0 else 0.0
    logger.info(
        f"  MMLU-Pro (generation) accuracy: {accuracy * 100:.1f}% ({correct}/{n})"
    )
    for cat, stats in sorted(by_category.items()):
        stats["acc"] = stats["correct"] / stats["total"]
        logger.info(
            f"    {cat}: {stats['acc'] * 100:.1f}% "
            f"({stats['correct']}/{stats['total']})"
        )

    return {"accuracy": accuracy, "correct": correct, "n": n,
            "by_category": by_category, "records": records}


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(
    all_scores: Dict[str, Dict[str, Optional[float]]],
    benchmarks: List[str],
    methods: List[str],
):
    """Print a method × benchmark table of primary metrics."""
    col_w = 14
    header = f"{'Method':<22}" + "".join(f"{b:>{col_w}}" for b in benchmarks)
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("GSM8K / MMLU-Pro / MMLU-Pro-Gen / MATH-500 — Primary metric (%)")
    print("=" * len(header))
    print(header)
    print(sep)
    for m in methods:
        row = f"{m:<22}"
        for b in benchmarks:
            score = all_scores.get(m, {}).get(b)
            if score is None:
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{score * 100:>{col_w - 1}.1f}%"
        print(row)
    print("=" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GSM8K + MMLU-Pro evaluator for KV cache quantization methods"
    )

    # Methods
    parser.add_argument("--methods", nargs="+", default=["baseline"],
                        choices=ALL_METHODS,
                        help="Methods to evaluate (default: baseline)")

    # Benchmarks
    parser.add_argument("--benchmarks", nargs="+",
                        default=["gsm8k", "mmlu_pro"],
                        choices=list(BENCHMARK_TASKS.keys()),
                        help="Benchmarks to run (default: gsm8k mmlu_pro).")
    parser.add_argument("--math500_max_new_tokens", type=int, default=2048,
                        help="Max new tokens for MATH-500 generation (default: 2048)")
    parser.add_argument("--math500_structured_prompt", action="store_true",
                        help="Use the structured step-by-step prompt from arXiv:2511.01815 "
                             "(tuned for Llama 3.3 70B). Default: simple 0-shot CoT prompt.")
    parser.add_argument("--mmlu_pro_subjects", nargs="+", default=None,
                        choices=MMLU_PRO_SUBJECTS, metavar="SUBJECT",
                        help="Run only specific MMLU-Pro subjects instead of all 14. "
                             f"Choices: {', '.join(MMLU_PRO_SUBJECTS)}. "
                             "Example: --mmlu_pro_subjects math physics")

    # Model paths
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID for baseline / kivi")
    parser.add_argument("--transform_model", "--palu_model", dest="transform_model",
                        default=None,
                        help="Path to the decomposed checkpoint from compress.py. For aatc this is the "
                             "rank-ratio-1.0 (invertible transform, no truncation) checkpoint; for "
                             "palu/palu_uniform it is the truncated one (e.g. ratio 0.7).")

    # KIVI
    parser.add_argument("--kivi_bits", type=int, default=2,
                        help="KV cache quantization bitwidth for KIVI (default: 2)")
    parser.add_argument("--kivi_group_size", type=int, default=32)
    parser.add_argument("--kivi_residual_length", type=int, default=None,
                        help="Recent tokens kept at full precision in KIVI "
                             "(default: same as --recent_tokens, or 128 if that is also unset)")

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
    parser.add_argument("--channelwise_scaling_file", default=None)
    parser.add_argument("--channel_group_size", type=int, default=64)

    # Sliding-window / attention-sink (palu_uniform / aatc)
    parser.add_argument("--recent_tokens", type=int, default=0,
                        help="Recent tokens kept at full bitwidth (default: 0 = off)")
    parser.add_argument("--attention_sink_tokens", type=int, default=0,
                        help="Attention-sink tokens kept at full bitwidth (default: 0 = off)")
    parser.add_argument("--sliding_step_size", type=int, default=None,
                        help="Tokens requantized per sliding step (default: recent_tokens // 2).")

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for lm-eval (default: 8; reduce if OOM)")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length (default: 4096)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per task for quick testing (default: full dataset)")

    # Output
    parser.add_argument("--output_dir", default="results/gsm8k_mmlu",
                        help="Directory to save per-method JSON results")

    args = parser.parse_args()

    # Validate
    needs_palu = [m for m in args.methods if m == "aatc" or m.startswith("palu")]
    if needs_palu and args.transform_model is None:
        parser.error(f"--transform_model is required for methods: {needs_palu}")
    needs_bitwidth = [m for m in args.methods if m == "aatc"]
    if needs_bitwidth and args.bitwidth_file is None:
        parser.error("--bitwidth_file is required for method 'aatc'.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split benchmarks: math_500 and mmlu_pro_gen use custom loops; rest goes to lm-eval.
    run_math500_flag     = "math_500"     in args.benchmarks
    run_mmlu_pro_gen_flag = "mmlu_pro_gen" in args.benchmarks
    lm_benchmarks = [b for b in args.benchmarks
                     if b not in ("math_500", "mmlu_pro_gen")]

    # Build the lm-eval task list. When *_subjects is given, replace the whole
    # group task with individual per-subject tasks.
    lm_tasks: List[str] = []
    # benchmark_labels drives column headers in the summary table.
    benchmark_labels: List[str] = []
    for b in lm_benchmarks:
        if b == "mmlu_pro" and args.mmlu_pro_subjects:
            subject_tasks = _resolve_subject_tasks("mmlu_pro", args.mmlu_pro_subjects)
            lm_tasks.extend(subject_tasks)
            benchmark_labels.extend(subject_tasks)
        else:
            lm_tasks.append(BENCHMARK_TASKS[b])
            benchmark_labels.append(b)
    if run_math500_flag:
        benchmark_labels.append("math_500")
    if run_mmlu_pro_gen_flag:
        benchmark_labels.append("mmlu_pro_gen")

    logger.info("=" * 72)
    logger.info("GSM8K / MMLU-Pro / MATH-500 / MMLU-Pro-Gen Evaluation")
    logger.info(f"  Methods    : {args.methods}")
    logger.info(f"  Benchmarks : {args.benchmarks}")
    if args.mmlu_pro_subjects:
        logger.info(f"  MMLU-Pro subjects: {args.mmlu_pro_subjects}")
    if lm_tasks:
        logger.info(f"  lm-eval tasks: {lm_tasks}")
    if run_math500_flag:
        logger.info(f"  MATH-500: 0-shot CoT, max_new_tokens={args.math500_max_new_tokens}")
    if run_mmlu_pro_gen_flag:
        logger.info(f"  MMLU-Pro (generation): {MMLU_PRO_N_SHOT}-shot, "
                    f"max_new_tokens={MMLU_PRO_GEN_MAX_NEW_TOKENS}")
    if args.limit:
        logger.info(f"  Limit      : {args.limit} samples per task (testing mode)")
    logger.info("=" * 72)

    # all_scores[method][benchmark] = float (primary metric, 0–1 scale)
    all_scores: Dict[str, Dict[str, Optional[float]]] = {}

    for method in args.methods:
        logger.info(f"\n{'='*72}")
        logger.info(f"Method: {method}")

        model, tokenizer = load_model_for_method(method, args)

        # KVQuant: patch generate for the whole iteration (covers GSM8K, MATH-500,
        # MMLU-Pro-gen). MMLU-Pro is loglikelihood-scored, so the patch is a no-op there.
        kvquant_orig_generate = patch_kvquant(model, args) if method == "kvquant" else None
        if method == "kvquant":
            _q = "calibrated NUQ" if args.kvquant_quantizer_file else "NormalFloat (uncalibrated)"
            print(f"  KVQuant: {args.kvquant_bits}-bit, {_q}, "
                  f"sparse={not args.kvquant_no_sparse}, sink={args.kvquant_sink_tokens}", flush=True)

        perdim_manager = None
        if method in ("aatc", "palu_uniform") and (args.recent_tokens > 0 or args.attention_sink_tokens > 0):
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

        method_scores: Dict[str, Optional[float]] = {}

        # ── lm-eval benchmarks (GSM8K, MMLU, MMLU-Pro) ───────────────────────
        if lm_tasks:
            # Patch model.generate to inject a fresh cache per call.
            # KIVI: KIVIQuantizedCache; palu sliding window: QuantizedDynamicCache.
            # MMLU-Pro is loglikelihood-scored (no generate call), so patches are no-ops there.
            orig_generate = None
            kivi_call_count = None
            if method == "kivi":
                orig_generate, kivi_call_count = patch_kivi(model, args)
                _resolved_res = (
                    args.kivi_residual_length if args.kivi_residual_length is not None
                    else (args.recent_tokens if args.recent_tokens > 0 else 128)
                )
                print(f"  KIVI: {args.kivi_bits}-bit, residual_length={_resolved_res} "
                      f"(sink={args.attention_sink_tokens})", flush=True)
            elif perdim_manager is not None:
                orig_generate = patch_palu_qdc(model, perdim_manager)
                logger.info(
                    f"  [sliding window] Patched model.generate with QuantizedDynamicCache "
                    f"(recent_tokens={args.recent_tokens}, "
                    f"attention_sink_tokens={args.attention_sink_tokens}). "
                    f"Active for generative tasks (GSM8K, MATH-500); "
                    f"MMLU uses loglikelihood scoring (no KV cache needed)."
                )

            # GSM8K must be run with num_fewshot=8 (standard); other tasks use
            # their built-in defaults (MMLU-Pro = 5-shot).
            gsm8k_tasks  = [t for t in lm_tasks if t == "gsm8k"]
            other_tasks  = [t for t in lm_tasks if t != "gsm8k"]

            raw_results: dict = {}
            if gsm8k_tasks:
                calls_before = kivi_call_count[0] if method == "kivi" else None
                logger.info("  Running GSM8K with num_fewshot=8 ...")
                raw_results.update(run_lm_eval(model, tokenizer, gsm8k_tasks, args,
                                               num_fewshot=8))
                if calls_before is not None:
                    print(f"  [KIVI] model.generate calls during GSM8K: "
                          f"{kivi_call_count[0] - calls_before}", flush=True)
            if other_tasks:
                calls_before = kivi_call_count[0] if method == "kivi" else None
                logger.info("  Running remaining lm-eval tasks with default few-shot ...")
                raw_results.update(run_lm_eval(model, tokenizer, other_tasks, args))
                if calls_before is not None:
                    print(f"  [KIVI] model.generate calls during {other_tasks}: "
                          f"{kivi_call_count[0] - calls_before}", flush=True)

            if orig_generate is not None:
                restore_generate(model, orig_generate)

            for label in benchmark_labels:
                if label == "math_500":
                    continue
                score = _collect_scores(raw_results, label)
                method_scores[label] = score
                display = f"{score * 100:.1f}%" if score is not None else "N/A"
                logger.info(f"  {label}: {display}")

            # Save raw lm-eval results for this method
            out_path = output_dir / f"{method}_lmeval.json"
            with open(out_path, "w") as f:
                json.dump(raw_results, f, indent=2)
            logger.info(f"  Saved lm-eval results → {out_path}")

        # ── MATH-500 custom evaluation ────────────────────────────────────────
        if run_math500_flag:
            math_results = run_math500(model, tokenizer, args, manager=perdim_manager)
            method_scores["math_500"] = math_results["accuracy"]

            out_path = output_dir / f"{method}_math500.json"
            # Omit full generated texts from the summary JSON to keep it small
            summary_results = {k: v for k, v in math_results.items() if k != "records"}
            with open(out_path, "w") as f:
                json.dump(summary_results, f, indent=2)
            logger.info(f"  Saved MATH-500 results → {out_path}")

        # ── MMLU-Pro generation-based evaluation ─────────────────────────────
        if run_mmlu_pro_gen_flag:
            mpg_results = run_mmlu_pro_gen(model, tokenizer, args, manager=perdim_manager)
            method_scores["mmlu_pro_gen"] = mpg_results["accuracy"]

            out_path = output_dir / f"{method}_mmlu_pro_gen.json"
            summary_results = {k: v for k, v in mpg_results.items() if k != "records"}
            with open(out_path, "w") as f:
                json.dump(summary_results, f, indent=2)
            logger.info(f"  Saved MMLU-Pro-gen results → {out_path}")

        all_scores[method] = method_scores

        if kvquant_orig_generate is not None:
            restore_generate(model, kvquant_orig_generate)

        # Free GPU memory between methods
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    print_summary(all_scores, benchmark_labels, args.methods)

    # Save combined summary
    summary_path = output_dir / "summary.json"
    summary = {
        "methods": args.methods,
        "benchmarks": benchmark_labels,
        "scores": {
            m: {b: (round(s * 100, 2) if s is not None else None)
                for b, s in scores.items()}
            for m, scores in all_scores.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
