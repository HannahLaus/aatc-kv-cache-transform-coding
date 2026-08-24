#!/usr/bin/env python3
"""Stage 2 of AATC — rate-distortion calibration.

Collects the per-channel statistics that the attention-aware distortion of
Theorem 1 needs: the transformed-coefficient variances sigma^2 and the
attention-aware importance weights w (the q_c^2 term for keys, W_Oc for
values). Run it on the rank-ratio-1.0 checkpoint from compress.py; the output
.pt feeds run_bit_allocation.py.

Paper default: 256 sequences of 2048 tokens, split evenly between FineWeb
(sample-10BT) and OpenR1-Math-220k.
"""

import argparse
import torch
from loguru import logger
from tqdm import tqdm
from utils import load_model_and_tokenizer
from rd_calibration import run_rd_calibration


def main():
    parser = argparse.ArgumentParser(
        description="Run RD calibration on a compressed model to collect statistics. "
        "If OOM: use --low_memory or --seqlen 1024 --nsamples 128 --max_query_capture_batches 8 --gradient_checkpointing"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the compressed model (local path or HuggingFace model ID)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run calibration on"
    )
    
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Use conservative settings to avoid OOM (recommended on Slurm): seqlen=1024, nsamples=128, max_query_capture_batches=8, gradient_checkpointing. On Slurm you may also need e.g. #SBATCH --mem=48G."
    )
    
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples (default 128 to reduce OOM risk; use 256 for more accuracy)."
    )
    
    parser.add_argument(
        "--seqlen",
        type=int,
        default=1024,
        help="Sequence length for calibration (default 1024 to reduce OOM; use 2048 for more accuracy)."
    )
    parser.add_argument(
        "--max_query_capture_batches",
        type=int,
        default=16,
        help="Max batches to keep for key calibration (limits RAM; default 16 to avoid OOM)."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce GPU memory (slower)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save statistics. If None, uses default cache location"
    )
    
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cached statistics if available"
    )
    
    parser.add_argument(
        "--flash2",
        action="store_true",
        help="Use flash attention 2"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information"
    )
    
    parser.add_argument(
        "--high_accuracy",
        action="store_true",
        help="Use higher seqlen/nsamples/capture for better stats (more memory). Overrides defaults to seqlen=2048, nsamples=256, max_query_capture_batches=32."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        help=(
            "Calibration dataset. Supported: wikitext2, c4, fineweb, openr1math, "
            "fineweb_openr1math (50/50 mixture, recommended for broad coverage), "
            "or LongBench datasets: lcc, triviaqa, qasper, trec, samsum, repobench-p, "
            "qmsum, multi_news, lsht (default: wikitext2)"
        )
    )

    args = parser.parse_args()
    
    if args.low_memory:
        args.seqlen = 1024
        args.nsamples = 128
        args.max_query_capture_batches = 8
        args.gradient_checkpointing = True
        logger.info("Low-memory mode: seqlen=1024, nsamples=128, max_query_capture_batches=8, gradient_checkpointing=True")
    if args.high_accuracy and not args.low_memory:
        args.seqlen = 2048
        args.nsamples = 256
        args.max_query_capture_batches = 32
        logger.info("High-accuracy mode: seqlen=2048, nsamples=256, max_query_capture_batches=32")
    
    # Setup logger
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level="INFO" if not args.verbose else "DEBUG"
    )
    
    # Load the compressed model
    logger.info(f"Loading compressed model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, use_flash_attn2=args.flash2)
    model.to(torch.device(args.device))
    
    # Check if model has HeadwiseLowRankModule layers
    has_compressed_layers = False
    for name, module in model.named_modules():
        if module.__class__.__name__ == "HeadwiseLowRankModule":
            has_compressed_layers = True
            logger.info(f"Found compressed layer: {name}")
            break
    
    if not has_compressed_layers:
        logger.warning(
            "No HeadwiseLowRankModule layers found in the model. "
            "This model may not be compressed. RD calibration requires compressed models."
        )
    
    # Run RD calibration (returns sigma2, w, w_o, sc, sc_k, w_k_distortion)
    logger.info("Starting RD calibration...")
    sigma2, w, w_o, sc, sc_k, w_k_distortion = run_rd_calibration(
        model,
        tokenizer,
        args.device,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        cache_file=args.output_path,
        use_cache=args.use_cache,
        max_query_capture_batches=args.max_query_capture_batches,
        gradient_checkpointing=args.gradient_checkpointing,
        dataset=args.dataset,
    )
    
    logger.info(f"RD calibration completed!")
    logger.info(f"Collected statistics for {len(sigma2)} modules")
    if w_o:
        logger.info(f"W_O weights extracted for {len(w_o)} value layers")
    if sc:
        logger.info(f"Scale statistics (sc) loaded for {len(sc)} layers")
    if sc_k:
        logger.info(f"Key channel-wise scaling (sc_k) for {len(sc_k)} layers")
    if w_k_distortion:
        logger.info(f"Key distortion weights (w_k_distortion) for {len(w_k_distortion)} layers")
    
    # Print summary
    if args.verbose:
        logger.info("\nStatistics summary:")
        for name in sorted(sigma2.keys()):
            logger.info(f"  {name}:")
            logger.info(f"    sigma2 shape: {sigma2[name].shape if sigma2[name] is not None else 'None'}")
            logger.info(f"    w shape: {w[name].shape if w[name] is not None else 'None'}")
        if w_o:
            logger.info("\nW_O (output projection) summary:")
            for name in sorted(w_o.keys())[:5]:
                logger.info(f"  {name}: shape {w_o[name].shape if w_o[name] is not None else 'None'}")
            if len(w_o) > 5:
                logger.info(f"  ... and {len(w_o) - 5} more")
        if sc_k:
            logger.info("\nKey calibration (sc_k) summary:")
            for name in sorted(sc_k.keys())[:3]:
                t = sc_k[name]
                logger.info(f"  {name}: shape {t.shape if torch.is_tensor(t) else 'N/A'}")
            if len(sc_k) > 3:
                logger.info(f"  ... and {len(sc_k) - 3} more")
        if w_k_distortion:
            logger.info("\nKey distortion weights (w_k_distortion) summary:")
            for name in sorted(w_k_distortion.keys())[:3]:
                t = w_k_distortion[name]
                logger.info(f"  {name}: shape {t.shape if torch.is_tensor(t) else 'N/A'}")
            if len(w_k_distortion) > 3:
                logger.info(f"  ... and {len(w_k_distortion) - 3} more")


if __name__ == "__main__":
    main()



