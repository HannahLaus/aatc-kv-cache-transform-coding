#!/usr/bin/env python3
"""Stage 3 of AATC — reverse water-filling bit allocation.

Turns the RD statistics from run_rd_calibration.py into a per-channel bitwidth
file. This is the step that makes the method AATC rather than PALU: instead of
spending a flat budget on a truncated basis, it distributes a fixed budget
non-uniformly over the full-rank transformed channels, minimizing the
attention-aware distortion of Theorem 1 (Eq. 2, the classical Huang-Schultheiss
/ reverse water-filling solution).

Paper defaults: --min_bits 0 --max_bits 16, allocated globally across layers,
separately for keys and values, with sigma^2 normalized per layer by its own
95th percentile and the first three layers protected.
"""

import argparse
import re
import torch
from loguru import logger
from tqdm import tqdm
from rd_calibration import load_rd_statistics
from bit_allocation import (
    allocate_bits,
    allocate_bits_per_layer,
    print_bit_allocation_summary,
    get_total_dimensions,
    calculate_current_bit_usage,
    suggest_bit_budget,
    separate_kv_layers
)


def _layer_idx(name):
    m = re.search(r'\.layers\.(\d+)\.', name)
    return int(m.group(1)) if m else 999


def _natural_protected_budget(sigma2_dict, w_dict, prot_names, total_bits, budget_split, eps=1e-12):
    """
    Compute the importance-based natural budget share for the protected layer group.
    Used to implement floor semantics: protected layers get at least protect_bits_per_dim,
    but more if global importance allocates them a larger share.
    Returns the natural budget for protected layers under the given budget_split.
    Falls back to proportional (dims / total_dims * total_bits) for splits that don't use importance.
    """
    total_dims = sum(
        sum(v.numel() for v in heads.values() if v is not None and torch.is_tensor(v))
        for heads in sigma2_dict.values()
    )
    if total_dims == 0:
        return 0.0

    if budget_split in ('importance', 'fisher', 'log_fisher'):
        layer_imp = {}
        for name, heads in sigma2_dict.items():
            w_heads = w_dict.get(name, {})
            imp = 0.0
            for hidx, s2 in heads.items():
                if s2 is None or not torch.is_tensor(s2):
                    continue
                w_h = w_heads.get(hidx)
                if w_h is None or not torch.is_tensor(w_h):
                    imp += float(s2.clamp(min=eps).sum().item())
                else:
                    if budget_split == 'importance':
                        imp += float((s2.clamp(min=eps) * w_h.clamp(min=eps)).sum().item())
                    else:
                        raw = float(w_h.clamp(min=eps).sum().item())
                        imp += (math.log(raw + 1.0) if budget_split == 'log_fisher' else raw)
            layer_imp[name] = imp
        total_imp = sum(layer_imp.values())
        if total_imp <= 0:
            total_imp = 1.0
        prot_imp = sum(layer_imp.get(n, 0.0) for n in prot_names)
        return total_bits * (prot_imp / total_imp)
    else:
        # proportional / equal: share proportional to dims
        prot_dims = sum(
            sum(v.numel() for v in sigma2_dict[n].values() if v is not None and torch.is_tensor(v))
            for n in prot_names if n in sigma2_dict
        )
        return total_bits * (prot_dims / total_dims)


def _split_protected(sigma2_dict, w_dict, w_o_dict, n_protect):
    """Split layers into protected (layer index < n_protect) and free."""
    prot_s2, prot_w, prot_wo = {}, {}, {}
    free_s2, free_w, free_wo = {}, {}, {}
    for name in sigma2_dict:
        if _layer_idx(name) < n_protect:
            prot_s2[name] = sigma2_dict[name]
            if name in w_dict: prot_w[name] = w_dict[name]
            if w_o_dict and name in w_o_dict: prot_wo[name] = w_o_dict[name]
        else:
            free_s2[name] = sigma2_dict[name]
            if name in w_dict: free_w[name] = w_dict[name]
            if w_o_dict and name in w_o_dict: free_wo[name] = w_o_dict[name]
    return (prot_s2, prot_w, prot_wo), (free_s2, free_w, free_wo)


def main():
    parser = argparse.ArgumentParser(
        description="Run bit allocation for KV cache quantization"
    )
    
    parser.add_argument(
        "--stats_path",
        type=str,
        required=True,
        help="Path to the saved RD statistics file (.pt file)"
    )
    
    parser.add_argument(
        "--total_bits",
        type=float,
        default=None,
        help="Total bit budget for allocation (applies to both K and V if separate budgets not specified). "
             "If not provided, will calculate based on current usage."
    )
    
    parser.add_argument(
        "--total_bits_k",
        type=float,
        default=None,
        help="Total bit budget for keys (k_proj) only. If not provided, uses --total_bits or --compression_ratio."
    )
    
    parser.add_argument(
        "--total_bits_v",
        type=float,
        default=None,
        help="Total bit budget for values (v_proj) only. If not provided, uses --total_bits or --compression_ratio."
    )
    
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=None,
        help="Compression ratio (e.g., 0.5 for 50%% of original bits). "
             "Used if --total_bits is not provided. Default: 0.5. "
             "Applies to both K and V if separate ratios not specified."
    )
    
    parser.add_argument(
        "--compression_ratio_k",
        type=float,
        default=None,
        help="Compression ratio for keys (k_proj) only. If not provided, uses --compression_ratio."
    )
    
    parser.add_argument(
        "--compression_ratio_v",
        type=float,
        default=None,
        help="Compression ratio for values (v_proj) only. If not provided, uses --compression_ratio."
    )
    
    parser.add_argument(
        "--bits_per_dimension",
        type=int,
        default=16,
        help="Current bits per dimension (default: 16 for float16)"
    )
    
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length for bit calculation (default: 2048)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for bit calculation (default: 1)"
    )
    
    parser.add_argument(
        "--show_suggestions",
        action="store_true",
        help="Show bit budget suggestions based on current usage"
    )
    
    parser.add_argument(
        "--per_layer",
        action="store_true",
        help="Allocate bits per layer instead of globally (better budget utilization)"
    )
    
    parser.add_argument(
        "--budget_split",
        type=str,
        default="proportional",
        choices=["proportional", "equal", "importance", "fisher", "log_fisher"],
        help="How to split budget across layers when using --per_layer. "
             "'proportional': by dimension count (default). "
             "'equal': same bits per layer. "
             "'fisher': distribute budget by Σ_c w_c (Fisher weights only, no σ²). "
             "'log_fisher': distribute budget by log(Σ_c w_c + 1) — dampens extreme w outliers, "
             "giving a moderate tilt toward high-importance layers without blowing up the budget. "
             "'importance': distribute by Σ(w·σ²) — legacy, dominated by activation scale."
    )
    
    parser.add_argument(
        "--min_bits",
        type=int,
        default=0,
        help="Minimum bits per dimension (default: 0)"
    )
    
    parser.add_argument(
        "--max_bits",
        type=int,
        default=8,
        help="Maximum bits per dimension (default: 8)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the bit allocation results (.pt file). "
             "If not provided, auto-generates based on stats_path and total_bits"
    )
    
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small epsilon for numerical stability (default: 1e-12)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information"
    )
    
    parser.add_argument(
        "--use_variance_only",
        action="store_true",
        help="Use only variance (sigma^2) for bit allocation and ignore importance weights (w). "
             "By default, uses both variance and importance weights. "
             "Applies to both K and V unless --use_variance_only_k / --use_variance_only_v override it."
    )
    parser.add_argument(
        "--use_variance_only_k",
        dest="use_variance_only_k",
        action="store_const",
        const=True,
        default=None,
        help="Force variance-only allocation for KEYS (k_proj) only, ignoring the key "
             "importance weights (w_k_distortion / q_c^2). If unset, inherits --use_variance_only. "
             "Use --use_weights_k to force the opposite (weighted) for keys."
    )
    parser.add_argument(
        "--use_weights_k",
        dest="use_variance_only_k",
        action="store_const",
        const=False,
        help="Force weighted allocation for KEYS (use w_k_distortion / q_c^2), overriding "
             "--use_variance_only for keys only."
    )
    parser.add_argument(
        "--use_variance_only_v",
        dest="use_variance_only_v",
        action="store_const",
        const=True,
        default=None,
        help="Force variance-only allocation for VALUES (v_proj) only, ignoring the value "
             "importance weights (w_o / W_o). If unset, inherits --use_variance_only. "
             "Use --use_weights_v to force the opposite (weighted) for values."
    )
    parser.add_argument(
        "--use_weights_v",
        dest="use_variance_only_v",
        action="store_const",
        const=False,
        help="Force weighted allocation for VALUES (use w_o / W_o), overriding "
             "--use_variance_only for values only."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for DP bit allocation when using --per_layer (e.g. cuda:0). "
             "Stats are moved layer-by-layer to limit GPU memory. Default: cpu."
    )
    parser.add_argument(
        "--protect_first_n_layers",
        type=int,
        default=0,
        help="Number of initial layers to protect from global water-filling. "
             "Protected layers each receive a budget determined by --protect_bits_per_dim "
             "(default: proportional share = total_bits / n_layers) allocated via local "
             "per-layer water-filling. The remaining budget is optimized globally over the rest. "
             "E.g. --protect_first_n_layers 3 fixes layers 0, 1, 2."
    )
    parser.add_argument(
        "--protect_bits_per_dim",
        type=float,
        default=None,
        help="Bits per dimension for protected layers (both K and V). Overridden by "
             "--protect_bits_per_dim_k / --protect_bits_per_dim_v if those are set."
    )
    parser.add_argument(
        "--protect_bits_per_dim_k",
        type=float,
        default=None,
        help="Bits per dimension for protected K layers. Overrides --protect_bits_per_dim for K."
    )
    parser.add_argument(
        "--protect_bits_per_dim_v",
        type=float,
        default=None,
        help="Bits per dimension for protected V layers. Overrides --protect_bits_per_dim for V."
    )
    parser.add_argument(
        "--protect_budget_split",
        type=str,
        default=None,
        choices=["proportional", "importance", "fisher", "log_fisher"],
        help="How to compute the natural budget for protected layers. "
             "If None (default), protected layers receive exactly --protect_bits_per_dim bits/dim (fixed). "
             "'importance': budget proportional to sum(w*sigma2) — protected layers can exceed "
             "--protect_bits_per_dim when importance warrants it (--protect_bits_per_dim becomes a floor). "
             "'log_fisher': like importance but log-dampened to avoid extreme tilts. "
             "'proportional': proportional to dimension count (same as None when all layers have equal dims)."
    )
    parser.add_argument(
        "--w_normalization",
        type=str,
        default="global",
        choices=["global", "per_layer"],
        help="How to normalize importance weights w before scoring (K layers). "
             "'global': normalize by global 95th percentile (default). "
             "'per_layer': normalize by each layer's own 95th percentile — removes "
             "gradient-attenuation bias that starves early layers at low bitwidths."
    )
    parser.add_argument(
        "--w_o_normalization",
        type=str,
        default="global",
        choices=["global", "per_layer"],
        help="How to normalize output-projection importance w_o before scoring (V layers). "
             "'global': normalize by global 95th percentile (default). "
             "'per_layer': normalize w_o by each layer's own 95th percentile — removes "
             "the cross-layer w_o scale bias (w_o grows toward later layers) so it does "
             "not dominate inter-layer V budget allocation."
    )
    parser.add_argument(
        "--sigma2_normalization",
        type=str,
        default="global",
        choices=["global", "per_layer"],
        help="How to normalize variance sigma2 before scoring. "
             "'global': normalize by global 95th percentile (default). "
             "'per_layer': normalize by each layer's own 95th percentile — removes "
             "activation-scale bias where early layers have smaller activations and "
             "thus smaller sigma2, causing them to be starved independently of w."
    )
    parser.add_argument(
        "--min_avg_bits_per_layer",
        type=float,
        default=0.0,
        help="Minimum average bits per dimension per layer when using --per_layer. "
             "Layers whose water-filling budget falls below this floor are raised to "
             "min_avg_bits_per_layer * n_dims; the deficit is redistributed among "
             "other layers. E.g. --min_avg_bits_per_layer 1.0 ensures every layer "
             "gets at least 1 bit/dim on average. Default: 0.0 (no floor)."
    )

    args = parser.parse_args()

    # Resolve per-projection variance-only mode (fall back to the global flag)
    var_only_k = args.use_variance_only_k if args.use_variance_only_k is not None else args.use_variance_only
    var_only_v = args.use_variance_only_v if args.use_variance_only_v is not None else args.use_variance_only

    # Setup logger
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level="INFO" if not args.verbose else "DEBUG"
    )
    
    # Load statistics
    logger.info(f"Loading RD statistics from {args.stats_path}...")
    try:
        result = load_rd_statistics(args.stats_path)
        # Handle backward compatibility: 2, 3, 4, 5, or 6 return values
        if len(result) == 2:
            sigma2, w = result
            w_o = w_k_distortion = None
        elif len(result) == 3:
            sigma2, w, w_o = result
            w_k_distortion = None
        elif len(result) == 4:
            sigma2, w, w_o, w_k_distortion = result
        elif len(result) == 5:
            sigma2, w, w_o, _, w_k_distortion = result  # sc ignored (not used for bit allocation)
        elif len(result) == 6:
            sigma2, w, w_o, _, _, w_k_distortion = result  # sc, sc_k ignored
        else:
            raise ValueError(f"load_rd_statistics returned unexpected number of values: {len(result)}")
    except Exception as e:
        logger.error(f"Failed to load statistics: {e}")
        return
    
    logger.info(f"Loaded statistics for {len(sigma2)} layers")
    if w_o is not None:
        logger.info(f"Loaded W_O weights for {len(w_o)} value layers")
    else:
        logger.info("No W_O weights found (will use w_dict for values)")
    if w_k_distortion is not None:
        logger.info(f"Loaded key distortion weights (w_k_distortion) for {len(w_k_distortion)} layers")
    
    logger.info(
        f"Bit-allocation mode:  KEYS = {'variance-only' if var_only_k else 'weighted (w_k_distortion / q_c^2)'},  "
        f"VALUES = {'variance-only' if var_only_v else 'weighted (w_o / W_o)'}"
    )
    
    # Separate K and V layers
    sigma2_k, w_k, sigma2_v, w_v, w_o_v = separate_kv_layers(sigma2, w, w_o_dict=w_o)

    # For keys: use new formula weights (w_k_distortion) when available, else gradient-based w_k
    if w_k_distortion is not None:
        w_k_for_alloc = {}
        for layer in sigma2_k:
            if layer in w_k_distortion:
                t = w_k_distortion[layer]
                if isinstance(t, dict):
                    w_k_for_alloc[layer] = t
                else:
                    head_ids = sigma2_k[layer].keys() if isinstance(sigma2_k[layer], dict) else [0]
                    w_k_for_alloc[layer] = {h: t for h in head_ids}
            else:
                w_k_for_alloc[layer] = w_k.get(layer, {})
        logger.info(f"Using key distortion weights (w_k_distortion) for bit allocation ({len(w_k_distortion)} layers)")
    else:
        w_k_for_alloc = w_k
        logger.info("Using gradient-based w for key bit allocation (no w_k_distortion in stats)")

    
    logger.info(f"Found {len(sigma2_k)} key (k_proj) layers and {len(sigma2_v)} value (v_proj) layers")
    
    # Calculate current bit usage for K and V separately
    if len(sigma2_k) > 0:
        current_usage_k = calculate_current_bit_usage(
            sigma2_k,
            bits_per_dimension=args.bits_per_dimension,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )
    else:
        current_usage_k = None
        logger.warning("No k_proj layers found!")
    
    if len(sigma2_v) > 0:
        current_usage_v = calculate_current_bit_usage(
            sigma2_v,
            bits_per_dimension=args.bits_per_dimension,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )
    else:
        current_usage_v = None
        logger.warning("No v_proj layers found!")
    
    # Display current usage
    logger.info("\n" + "=" * 80)
    logger.info("Current Bit Usage (Unquantized)")
    logger.info("=" * 80)
    logger.info(f"Note: Bit allocation is PER DIMENSION (not per token)")
    logger.info(f"      For KV cache: multiply by seq_len={args.seq_len} to get total cache size")
    if current_usage_k:
        logger.info(f"\nKEYS (k_proj):")
        logger.info(f"  Dimensions: {current_usage_k['total_dimensions']}")
        logger.info(f"  Layers: {current_usage_k['num_layers']}")
        logger.info(f"  Bits per dimension: {current_usage_k['bits_per_dimension']}")
        logger.info(f"  Total bits (for allocation): {current_usage_k['total_bits_for_allocation']:.2e}")
        logger.info(f"  KV cache size (seq_len={args.seq_len}): {current_usage_k['total_bits_kv_cache']:.2e} bits ({current_usage_k['total_bits_kv_cache'] / (8 * 1024 * 1024):.2f} MB)")
    if current_usage_v:
        logger.info(f"\nVALUES (v_proj):")
        logger.info(f"  Dimensions: {current_usage_v['total_dimensions']}")
        logger.info(f"  Layers: {current_usage_v['num_layers']}")
        logger.info(f"  Bits per dimension: {current_usage_v['bits_per_dimension']}")
        logger.info(f"  Total bits (for allocation): {current_usage_v['total_bits_for_allocation']:.2e}")
        logger.info(f"  KV cache size (seq_len={args.seq_len}): {current_usage_v['total_bits_kv_cache']:.2e} bits ({current_usage_v['total_bits_kv_cache'] / (8 * 1024 * 1024):.2f} MB)")
    if current_usage_k and current_usage_v:
        total_alloc = current_usage_k['total_bits_for_allocation'] + current_usage_v['total_bits_for_allocation']
        total_cache = current_usage_k['total_bits_kv_cache'] + current_usage_v['total_bits_kv_cache']
        logger.info(f"\nTOTAL:")
        logger.info(f"  Total bits (for allocation): {total_alloc:.2e}")
        logger.info(f"  KV cache size: {total_cache:.2e} bits ({total_cache / (8 * 1024 * 1024):.2f} MB)")
    logger.info("=" * 80)
    
    # Determine bit budgets for K and V
    def determine_budget(sigma2_dict, current_usage, total_bits_arg, compression_ratio_arg, name):
        if total_bits_arg is not None:
            return total_bits_arg
        elif compression_ratio_arg is not None:
            # Calculate budget: bits_per_dim * compression_ratio * num_dimensions
            if current_usage:
                bits_per_dim = current_usage['bits_per_dimension'] * compression_ratio_arg
                return bits_per_dim * current_usage['total_dimensions']
            else:
                return 0
        elif args.total_bits is not None:
            # Split total_bits proportionally if separate budgets not specified
            if current_usage_k and current_usage_v:
                total_current = current_usage_k['total_bits_for_allocation'] + current_usage_v['total_bits_for_allocation']
                if total_current > 0:
                    if name == 'K':
                        return args.total_bits * (current_usage_k['total_bits_for_allocation'] / total_current)
                    else:
                        return args.total_bits * (current_usage_v['total_bits_for_allocation'] / total_current)
                else:
                    # If both are 0, split equally
                    logger.warning(f"Both K and V have 0 bits for allocation, splitting equally")
                    return args.total_bits / 2
            else:
                return args.total_bits / 2  # Equal split if only one type exists
        else:
            # Use default compression ratio
            default_ratio = args.compression_ratio if args.compression_ratio is not None else 0.5
            if current_usage:
                bits_per_dim = current_usage['bits_per_dimension'] * default_ratio
                return bits_per_dim * current_usage['total_dimensions']
            else:
                return 0
    
    logger.info(f"\nDEBUG: Budget calculation inputs:")
    logger.info(f"  args.total_bits_k: {args.total_bits_k}")
    logger.info(f"  args.total_bits_v: {args.total_bits_v}")
    logger.info(f"  args.compression_ratio_k: {args.compression_ratio_k}")
    logger.info(f"  args.compression_ratio_v: {args.compression_ratio_v}")
    logger.info(f"  args.total_bits: {args.total_bits}")
    logger.info(f"  args.compression_ratio: {args.compression_ratio}")
    if current_usage_k:
        logger.info(f"  K: total_dims={current_usage_k['total_dimensions']}, bits_per_dim={current_usage_k['bits_per_dimension']}")
    if current_usage_v:
        logger.info(f"  V: total_dims={current_usage_v['total_dimensions']}, bits_per_dim={current_usage_v['bits_per_dimension']}")
    
    total_bits_k = determine_budget(sigma2_k, current_usage_k, args.total_bits_k, args.compression_ratio_k, 'K')
    total_bits_v = determine_budget(sigma2_v, current_usage_v, args.total_bits_v, args.compression_ratio_v, 'V')

    logger.info(f"\nBit Budgets (for allocation):")
    logger.info(f"  Keys (k_proj):   {total_bits_k:.2e} bits")
    logger.info(f"  Values (v_proj): {total_bits_v:.2e} bits")
    logger.info(f"  Total:           {total_bits_k + total_bits_v:.2e} bits")
    
    if current_usage_k and current_usage_v:
        logger.info(f"\nDimension comparison:")
        logger.info(f"  K dimensions: {current_usage_k['total_dimensions']}")
        logger.info(f"  V dimensions: {current_usage_v['total_dimensions']}")
        logger.info(f"  Ratio (V/K): {current_usage_v['total_dimensions'] / current_usage_k['total_dimensions']:.2f}x")
        logger.info(f"  (Values have more dimensions, so they get proportionally more bits)")
        logger.info(f"\nNote: Budgets are proportional to dimensions.")
        logger.info(f"  If you want equal budgets, use --total_bits_k and --total_bits_v explicitly.")
        logger.info(f"  Or use --compression_ratio_k and --compression_ratio_v to set different ratios.")
    
    if current_usage_k:
        avg_bits_k = total_bits_k / current_usage_k['total_dimensions'] if current_usage_k['total_dimensions'] > 0 else 0
        ratio_k = avg_bits_k / current_usage_k['bits_per_dimension'] if current_usage_k['bits_per_dimension'] > 0 else 0
        logger.info(f"  K: {avg_bits_k:.2f} bits/dim (avg), {(1 - ratio_k) * 100:.1f}% compression")
        expected_ratio = args.compression_ratio_k if args.compression_ratio_k is not None else (args.compression_ratio if args.compression_ratio is not None else 0.5)
        logger.info(f"     Expected: {current_usage_k['bits_per_dimension'] * expected_ratio:.2f} bits/dim")
    if current_usage_v:
        avg_bits_v = total_bits_v / current_usage_v['total_dimensions'] if current_usage_v['total_dimensions'] > 0 else 0
        ratio_v = avg_bits_v / current_usage_v['bits_per_dimension'] if current_usage_v['bits_per_dimension'] > 0 else 0
        logger.info(f"  V: {avg_bits_v:.2f} bits/dim (avg), {(1 - ratio_v) * 100:.1f}% compression")
        expected_ratio = args.compression_ratio_v if args.compression_ratio_v is not None else (args.compression_ratio if args.compression_ratio is not None else 0.5)
        logger.info(f"     Expected: {current_usage_v['bits_per_dimension'] * expected_ratio:.2f} bits/dim")
    
    # Run bit allocation separately for K and V
    bit_alloc_k = {}
    bit_alloc_v = {}
    
    if len(sigma2_k) > 0:
        logger.info("\n" + "=" * 80)
        logger.info("Allocating bits for KEYS (k_proj)...")
        logger.info("=" * 80)
        logger.info(f"DEBUG: Calling allocate_bits with total_bits_k={total_bits_k:.2e}")
        logger.info(f"DEBUG: sigma2_k has {len(sigma2_k)} layers")
        logger.info(f"DEBUG: Per-layer allocation: {args.per_layer}")
        try:
            if args.protect_first_n_layers > 0:
                (ps2, pw, _), (fs2, fw, _) = _split_protected(
                    sigma2_k, w_k_for_alloc, None, args.protect_first_n_layers)
                total_dims_k = get_total_dimensions(sigma2_k)
                prot_dims_k = get_total_dimensions(ps2)
                bpd_k = args.protect_bits_per_dim_k if args.protect_bits_per_dim_k is not None else args.protect_bits_per_dim
                floor_budget_k = bpd_k * prot_dims_k if bpd_k is not None else total_bits_k * (prot_dims_k / max(get_total_dimensions(sigma2_k), 1))
                if args.protect_budget_split is not None:
                    natural_budget_k = _natural_protected_budget(
                        sigma2_k, w_k_for_alloc, list(ps2.keys()), total_bits_k, args.protect_budget_split)
                    prot_budget = max(floor_budget_k, natural_budget_k)
                    logger.info(f"  Hybrid K: {len(ps2)} protected layers, {len(fs2)} free layers, "
                                f"floor={floor_budget_k:.0f} bits ({floor_budget_k/max(prot_dims_k,1):.2f} bpd), "
                                f"natural={natural_budget_k:.0f} bits ({natural_budget_k/max(prot_dims_k,1):.2f} bpd), "
                                f"protected budget={prot_budget:.0f} bits "
                                f"({prot_budget/max(prot_dims_k,1):.2f} bits/dim avg)")
                else:
                    prot_budget = floor_budget_k
                    logger.info(f"  Hybrid K: {len(ps2)} protected layers, {len(fs2)} free layers, "
                                f"protected budget={prot_budget:.0f} bits "
                                f"({prot_budget/max(prot_dims_k,1):.2f} bits/dim avg)")
                prot_budget = min(prot_budget, total_bits_k)
                bit_alloc_k = {}
                if ps2:
                    bit_alloc_k.update(allocate_bits_per_layer(
                        sigma2_dict=ps2, w_dict=pw, total_bits=prot_budget,
                        min_bits=args.min_bits, max_bits=args.max_bits, eps=args.eps,
                        budget_split=args.protect_budget_split or 'proportional', use_variance_only=var_only_k,
                        w_o_dict=None, device=args.device,
                        min_avg_bits_per_layer=args.min_avg_bits_per_layer,
                        w_normalization=args.w_normalization,
                        sigma2_normalization=args.sigma2_normalization,
                        w_o_normalization=args.w_o_normalization
                    ))
                if fs2:
                    bit_alloc_k.update(allocate_bits(
                        sigma2_dict=fs2, w_dict=fw, total_bits=total_bits_k - prot_budget,
                        min_bits=args.min_bits, max_bits=args.max_bits, eps=args.eps,
                        use_variance_only=var_only_k, w_o_dict=None,
                        device=args.device, min_avg_bits_per_layer=args.min_avg_bits_per_layer,
                        w_normalization=args.w_normalization,
                        sigma2_normalization=args.sigma2_normalization,
                        w_o_normalization=args.w_o_normalization
                    ))
            elif args.per_layer:
                bit_alloc_k = allocate_bits_per_layer(
                    sigma2_dict=sigma2_k,
                    w_dict=w_k_for_alloc,
                    total_bits=total_bits_k,
                    min_bits=args.min_bits,
                    max_bits=args.max_bits,
                    eps=args.eps,
                    budget_split=args.budget_split,
                    use_variance_only=var_only_k,
                    w_o_dict=None,  # Keys use w_dict (w_k_distortion when available)
                    device=args.device,
                    min_avg_bits_per_layer=args.min_avg_bits_per_layer,
                    w_normalization=args.w_normalization,
                    sigma2_normalization=args.sigma2_normalization,
                    w_o_normalization=args.w_o_normalization
                )
            else:
                bit_alloc_k = allocate_bits(
                    sigma2_dict=sigma2_k,
                    w_dict=w_k_for_alloc,
                    total_bits=total_bits_k,
                    min_bits=args.min_bits,
                    max_bits=args.max_bits,
                    eps=args.eps,
                    use_variance_only=var_only_k,
                    w_o_dict=None,  # Keys use w_dict (w_k_distortion when available)
                    device=args.device,
                    min_avg_bits_per_layer=args.min_avg_bits_per_layer,
                    w_normalization=args.w_normalization,
                    sigma2_normalization=args.sigma2_normalization,
                    w_o_normalization=args.w_o_normalization
                )
            print_bit_allocation_summary(bit_alloc_k)
        except Exception as e:
            logger.error(f"Failed to allocate bits for keys: {e}")
            import traceback
            traceback.print_exc()
            return
    
    if len(sigma2_v) > 0:
        logger.info("\n" + "=" * 80)
        logger.info("Allocating bits for VALUES (v_proj)...")
        logger.info("=" * 80)
        logger.info(f"DEBUG: Calling allocate_bits with total_bits_v={total_bits_v:.2e}")
        logger.info(f"DEBUG: sigma2_v has {len(sigma2_v)} layers")
        logger.info(f"DEBUG: Per-layer allocation: {args.per_layer}")
        try:
            if args.protect_first_n_layers > 0:
                (ps2, pw, pwo), (fs2, fw, fwo) = _split_protected(
                    sigma2_v, w_v, w_o_v, args.protect_first_n_layers)
                total_dims_v = get_total_dimensions(sigma2_v)
                prot_dims_v = get_total_dimensions(ps2)
                bpd_v = args.protect_bits_per_dim_v if args.protect_bits_per_dim_v is not None else args.protect_bits_per_dim
                floor_budget_v = bpd_v * prot_dims_v if bpd_v is not None else total_bits_v * (prot_dims_v / max(get_total_dimensions(sigma2_v), 1))
                if args.protect_budget_split is not None:
                    natural_budget_v = _natural_protected_budget(
                        sigma2_v, w_v, list(ps2.keys()), total_bits_v, args.protect_budget_split)
                    prot_budget = max(floor_budget_v, natural_budget_v)
                    logger.info(f"  Hybrid V: {len(ps2)} protected layers, {len(fs2)} free layers, "
                                f"floor={floor_budget_v:.0f} bits ({floor_budget_v/max(prot_dims_v,1):.2f} bpd), "
                                f"natural={natural_budget_v:.0f} bits ({natural_budget_v/max(prot_dims_v,1):.2f} bpd), "
                                f"protected budget={prot_budget:.0f} bits "
                                f"({prot_budget/max(prot_dims_v,1):.2f} bits/dim avg)")
                else:
                    prot_budget = floor_budget_v
                    logger.info(f"  Hybrid V: {len(ps2)} protected layers, {len(fs2)} free layers, "
                                f"protected budget={prot_budget:.0f} bits "
                                f"({prot_budget/max(prot_dims_v,1):.2f} bits/dim avg)")
                prot_budget = min(prot_budget, total_bits_v)
                bit_alloc_v = {}
                if ps2:
                    bit_alloc_v.update(allocate_bits_per_layer(
                        sigma2_dict=ps2, w_dict=pw, total_bits=prot_budget,
                        min_bits=args.min_bits, max_bits=args.max_bits, eps=args.eps,
                        budget_split=args.protect_budget_split or 'proportional', verbose=False,
                        use_variance_only=var_only_v,
                        w_o_dict=pwo if pwo else None, device=args.device,
                        min_avg_bits_per_layer=args.min_avg_bits_per_layer,
                        w_normalization=args.w_normalization,
                        sigma2_normalization=args.sigma2_normalization,
                        w_o_normalization=args.w_o_normalization
                    ))
                if fs2:
                    bit_alloc_v.update(allocate_bits(
                        sigma2_dict=fs2, w_dict=fw, total_bits=total_bits_v - prot_budget,
                        min_bits=args.min_bits, max_bits=args.max_bits, eps=args.eps,
                        verbose=False, use_variance_only=var_only_v,
                        w_o_dict=fwo if fwo else None, device=args.device,
                        min_avg_bits_per_layer=args.min_avg_bits_per_layer,
                        w_normalization=args.w_normalization,
                        sigma2_normalization=args.sigma2_normalization,
                        w_o_normalization=args.w_o_normalization
                    ))
            elif args.per_layer:
                bit_alloc_v = allocate_bits_per_layer(
                    sigma2_dict=sigma2_v,
                    w_dict=w_v,
                    total_bits=total_bits_v,
                    min_bits=args.min_bits,
                    max_bits=args.max_bits,
                    eps=args.eps,
                    budget_split=args.budget_split,
                    verbose=False,
                    use_variance_only=var_only_v,
                    w_o_dict=w_o_v,
                    device=args.device,
                    min_avg_bits_per_layer=args.min_avg_bits_per_layer,
                    w_normalization=args.w_normalization,
                    sigma2_normalization=args.sigma2_normalization,
                    w_o_normalization=args.w_o_normalization
                )
            else:
                bit_alloc_v = allocate_bits(
                    sigma2_dict=sigma2_v,
                    w_dict=w_v,
                    total_bits=total_bits_v,
                    min_bits=args.min_bits,
                    max_bits=args.max_bits,
                    eps=args.eps,
                    verbose=False,
                    use_variance_only=var_only_v,
                    w_o_dict=w_o_v,
                    device=args.device,
                    min_avg_bits_per_layer=args.min_avg_bits_per_layer,
                    w_normalization=args.w_normalization,
                    sigma2_normalization=args.sigma2_normalization,
                    w_o_normalization=args.w_o_normalization
                )
            print_bit_allocation_summary(bit_alloc_v, show_values_detail=False)
        except Exception as e:
            logger.error(f"Failed to allocate bits for values: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Combine allocations
    bit_alloc = {**bit_alloc_k, **bit_alloc_v}
    
    # Debug: Check what we actually allocated
    total_allocated_k = sum(bits.sum().item() for layer in bit_alloc_k.values() for bits in layer.values())
    total_allocated_v = sum(bits.sum().item() for layer in bit_alloc_v.values() for bits in layer.values())
    logger.info(f"\nDEBUG: Actually allocated bits:")
    logger.info(f"  K: {total_allocated_k:.2f} bits (target was {total_bits_k:.2e})")
    logger.info(f"  V: {total_allocated_v:.2f} bits (target was {total_bits_v:.2e})")
    logger.info(f"  Total: {total_allocated_k + total_allocated_v:.2f} bits")
    
    # Save results
    if args.output_path is None:
        # Auto-generate output path
        import os
        base_name = os.path.splitext(args.stats_path)[0]
        args.output_path = f"{base_name}_bits-k{total_bits_k:.0f}-v{total_bits_v:.0f}_min-{args.min_bits}_max-{args.max_bits}.pt"
    
    logger.info(f"\nSaving bit allocation results to {args.output_path}...")
    try:
        # Convert nested defaultdicts to regular dicts for saving
        bit_alloc_dict = {}
        for layer_name in bit_alloc:
            bit_alloc_dict[layer_name] = {}
            for head_idx in bit_alloc[layer_name]:
                bit_alloc_dict[layer_name][head_idx] = bit_alloc[layer_name][head_idx].cpu()
        
        torch.save({
            "bit_alloc": bit_alloc_dict,
            "bit_alloc_k": {k: {h: v.cpu() for h, v in heads.items()} for k, heads in bit_alloc_k.items()},
            "bit_alloc_v": {k: {h: v.cpu() for h, v in heads.items()} for k, heads in bit_alloc_v.items()},
            "total_bits_k": total_bits_k,
            "total_bits_v": total_bits_v,
            "min_bits": args.min_bits,
            "max_bits": args.max_bits,
            "stats_path": args.stats_path
        }, args.output_path)
        logger.info(f"Bit allocation saved to {args.output_path}", fg="green")
    except Exception as e:
        logger.error(f"Failed to save bit allocation: {e}")
        return
    
    logger.info("Bit allocation completed successfully!")


if __name__ == "__main__":
    main()

