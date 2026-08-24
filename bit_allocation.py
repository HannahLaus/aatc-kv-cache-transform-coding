"""
Bit allocation using reverse water-filling for KV cache quantization.
Allocates bits per dimension based on variance (sigma^2) and importance (w) statistics.
Can optionally use only variance (sigma^2) and ignore importance weights (w).
"""

import math
import torch
from typing import Dict, Tuple
from loguru import logger


def _solve_lambda(scores, total_bits, min_bits=0, max_bits=8):
    """
    Solve for lambda using binary search for reverse water-filling.
    
    Args:
        scores: Tensor of shape [n] containing w_i * sigma_i^2 for each dimension
        total_bits: Total bit budget
        min_bits: Minimum bits per dimension (for clamping)
        max_bits: Maximum bits per dimension (for clamping)
        
    Returns:
        lambda_star: Optimal lambda value
    """
    # Binary search on lambda
    # Smaller lambda -> more bits allocated
    # Larger lambda -> fewer bits allocated
    
    # Scores should already be cleaned (finite) before being passed here, but add safety checks
    # Replace any remaining inf/nan with finite values
    # Use dtype-appropriate max value
    dtype_max = torch.finfo(scores.dtype).max
    safe_max = min(1e8, dtype_max / 10.0)
    scores = torch.nan_to_num(scores, nan=1e-6, posinf=safe_max, neginf=1e-6)
    
    # Get finite scores for determining bounds
    scores_finite = scores[torch.isfinite(scores) & (scores > 0)]
    if len(scores_finite) == 0:
        # Fallback: use a default lambda if all scores are non-finite
        logger.warning("All scores are non-finite in _solve_lambda, using fallback lambda")
        return torch.tensor(1e-6, dtype=scores.dtype, device=scores.device)
    
    # Start with lambda range from min to max scores
    # Since scores are now normalized and all positive, we can use them directly
    lo = scores_finite.min()
    hi = scores_finite.max()
    
    # Ensure we have valid range (scores should all be positive after normalization)
    if lo <= 0 or not torch.isfinite(lo):
        lo = scores_finite[scores_finite > 0].min() if (scores_finite > 0).any() else torch.tensor(1e-12, dtype=scores.dtype, device=scores.device)
    
    # Ensure hi is finite and within dtype limits
    if not torch.isfinite(hi):
        hi = torch.tensor(min(1e6, safe_max), dtype=scores.dtype, device=scores.device)
    
    # Use all scores (they should all be finite now) to maintain correct indices
    for _ in range(50):
        mid = (lo + hi) / 2
        # Ensure mid is finite and positive, within dtype limits
        mid = torch.clamp(mid, min=1e-12, max=safe_max)
        
        # Compute bits: b = 0.5 * log2(scores / mid)
        # Clamp ratio to prevent inf/nan
        ratio = (scores / mid).clamp(min=1e-12, max=safe_max)
        b = 0.5 * torch.log2(ratio)
        
        # Clamp bits to valid range and round
        b = torch.clamp(b, min=min_bits, max=max_bits)
        b = torch.round(b)
        
        allocated = b.sum().item()
        
        if allocated > total_bits:
            lo = mid  # Need larger lambda to reduce bits
        else:
            hi = mid  # Need smaller lambda to increase bits
    
    return hi


def _allocate_bits_dp(scores, sigma2, total_bits, min_bits=0, max_bits=16):
    """
    Allocate bits using dynamic programming to minimize distortion (vectorized for speed).
    
    Objective: Minimize D = Σ_i w_i · σ²_i · 2^(-2b_i) subject to Σ_i b_i ≤ B
    where b_i ∈ {min_bits, min_bits+1, ..., max_bits}
    
    Args:
        scores: Tensor of shape [n] containing w_i * sigma_i^2 for each dimension
        sigma2: Tensor of shape [n] containing sigma_i^2 for each dimension
        total_bits: Total bit budget (integer)
        min_bits: Minimum bits per dimension (default: 0)
        max_bits: Maximum bits per dimension (default: 16)
        
    Returns:
        LongTensor of shape [n] with optimal bit allocations per dimension
    """
    n = len(scores)
    total_bits = int(total_bits)  # Ensure integer
    dev = scores.device
    dtype = scores.dtype
    B = max_bits - min_bits + 1  # number of bit choices per dimension
    
    # Handle edge cases
    if n == 0:
        return torch.tensor([], dtype=torch.long, device=dev)
    if total_bits <= 0:
        return torch.full((n,), min_bits, dtype=torch.long, device=dev)
    
    # Clamp total_bits to reasonable range to avoid memory issues
    max_total_bits = n * max_bits
    if total_bits > max_total_bits:
        logger.warning(f"total_bits ({total_bits}) exceeds maximum possible ({max_total_bits}), clamping")
        total_bits = max_total_bits
    
    # Ensure scores and sigma2 are finite and positive
    scores = torch.nan_to_num(scores, nan=1e-10, posinf=1e8, neginf=1e-10)
    sigma2 = torch.nan_to_num(sigma2, nan=1e-10, posinf=1e8, neginf=1e-10)
    scores = scores.clamp(min=1e-10)
    sigma2 = sigma2.clamp(min=1e-10)
    
    # Precompute distortion table (vectorized): [n, B]
    b_vals = torch.arange(min_bits, max_bits + 1, device=dev, dtype=dtype)
    distortion_table = scores.unsqueeze(1) * (2.0 ** (-2.0 * b_vals)).unsqueeze(0)  # (n, B)
    
    # DP table: dp[i, b] = minimum distortion for first i dimensions using b bits
    dp = torch.full((n + 1, total_bits + 1), float('inf'), device=dev, dtype=dtype)
    dp[0, 0] = 0.0
    # Backpointers: which b_i was chosen at (i, b)
    backptr = torch.zeros(n + 1, total_bits + 1, dtype=torch.long, device=dev)
    
    # Vectorized DP: for each i, compute dp[i, :] from dp[i-1, :]
    b_vec = torch.arange(total_bits + 1, device=dev)
    j_vec = torch.arange(B, device=dev)
    # index[b, j] = b - min_bits - j (prev budget)
    index = b_vec.unsqueeze(1) - min_bits - j_vec.unsqueeze(0)  # (total_bits+1, B)
    valid = (index >= 0)
    
    for i in range(1, n + 1):
        dim_idx = i - 1
        prev_row = dp[i - 1, :]  # (total_bits+1,)
        # M[b, j] = prev_row[b - min_bits - j] + distortion_table[dim_idx, j], or inf if invalid
        prev_vals = torch.full((total_bits + 1, B), float('inf'), device=dev, dtype=dtype)
        prev_vals[valid] = prev_row[index[valid]]
        dist_row = distortion_table[dim_idx, :].unsqueeze(0)  # (1, B)
        M = torch.where(valid, prev_vals + dist_row, torch.tensor(float('inf'), device=dev, dtype=dtype))
        dp[i, :] = M.min(dim=1).values
        backptr[i, :] = min_bits + M.min(dim=1).indices
    
    # Backtrack (single pass over n)
    bit_alloc = torch.zeros(n, dtype=torch.long, device=dev)
    b_remaining = total_bits
    for i in range(n, 0, -1):
        b_remaining = min(b_remaining, total_bits)
        best_b_i = backptr[i, b_remaining].item()
        dim_idx = i - 1
        bit_alloc[dim_idx] = best_b_i
        b_remaining -= best_b_i
    
    # Validate allocation
    actual_total = bit_alloc.sum().item()
    if actual_total > total_bits:
        logger.warning(f"DP allocation exceeded budget: {actual_total} > {total_bits}, clamping")
        excess = actual_total - total_bits
        sorted_indices = torch.argsort(scores)
        for idx in sorted_indices:
            if excess <= 0:
                break
            if bit_alloc[idx] > min_bits:
                reduction = min(excess, bit_alloc[idx] - min_bits)
                bit_alloc[idx] -= reduction
                excess -= reduction
    
    return bit_alloc


def clamp_and_round(b, min_bits=0, max_bits=8):
    """
    Clamp bits to valid range and round to integers.
    
    Args:
        b: Tensor of bit allocations (can be float)
        min_bits: Minimum bits per dimension
        max_bits: Maximum bits per dimension
        
    Returns:
        LongTensor of rounded and clamped bits
    """
    b = torch.clamp(b, min=min_bits, max=max_bits)
    return torch.round(b).long()


def separate_kv_layers(sigma2_dict, w_dict, w_o_dict=None):
    """
    Separate layers into keys (k_proj) and values (v_proj).
    
    Args:
        sigma2_dict: Nested dict sigma2_dict[layer][head] = Tensor [r]
        w_dict: Nested dict w_dict[layer][head] = Tensor [r]
        w_o_dict: Optional nested dict w_o_dict[layer][head] = Tensor [r] for value layers
        
    Returns:
        Tuple of (sigma2_k_dict, w_k_dict, sigma2_v_dict, w_v_dict, w_o_v_dict)
        where w_o_v_dict is w_o_dict separated for value layers only
    """
    sigma2_k_dict = {}
    w_k_dict = {}
    sigma2_v_dict = {}
    w_v_dict = {}
    w_o_v_dict = {}
    
    for layer_name in sigma2_dict:
        if 'k_proj' in layer_name:
            sigma2_k_dict[layer_name] = sigma2_dict[layer_name]
            if layer_name in w_dict:
                w_k_dict[layer_name] = w_dict[layer_name]
        elif 'v_proj' in layer_name:
            sigma2_v_dict[layer_name] = sigma2_dict[layer_name]
            if layer_name in w_dict:
                w_v_dict[layer_name] = w_dict[layer_name]
            # w_o_dict is keyed by self_attn (e.g. model.layers.0.self_attn), not v_proj
            if w_o_dict is not None:
                w_o_key = layer_name.replace(".v_proj", "") if layer_name.endswith(".v_proj") else layer_name
                if w_o_key in w_o_dict:
                    w_o_v_dict[layer_name] = w_o_dict[w_o_key]
        else:
            logger.warning(f"Layer {layer_name} is neither k_proj nor v_proj, skipping...")
    
    return sigma2_k_dict, w_k_dict, sigma2_v_dict, w_v_dict, w_o_v_dict


def _nested_dict_to_device(d, device):
    """Move all tensors in a nested dict (layer -> head -> tensor) to device. Returns new dict."""
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _nested_dict_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def allocate_bits_per_layer(
    sigma2_dict,
    w_dict,
    total_bits,
    min_bits=0,
    max_bits=8,
    eps=1e-12,
    budget_split='proportional',
    verbose=True,
    use_variance_only=False,
    w_o_dict=None,
    device=None,
    min_avg_bits_per_layer=0.0,
    w_normalization='global',
    sigma2_normalization='global',
    w_o_normalization='global'
):
    """
    Allocate bits per layer using reverse water-filling.
    Each layer gets its own budget and optimizes independently.

    Args:
        sigma2_dict: Nested dict sigma2_dict[layer][head] = Tensor [r] of variance statistics
        w_dict: Nested dict w_dict[layer][head] = Tensor [r] of importance statistics
        total_bits: Total bit budget (scalar)
        min_bits: Minimum bits per dimension (default: 0)
        max_bits: Maximum bits per dimension (default: 8)
        eps: Small epsilon to avoid numerical issues (default: 1e-12)
        budget_split: How to split budget across layers:
            - 'proportional': Split proportionally to number of dimensions per layer
            - 'equal': Split equally across all layers
            - 'fisher': Split proportionally to Σ_c w_c (Fisher weights only, no σ²).
                        Best approximation to the global DP: avoids activation-scale bias.
            - 'importance': Split proportionally to Σ(w·σ²). Legacy; prefer 'fisher'.
        use_variance_only: If True, use only variance (sigma^2) and ignore importance weights (w)
        device: Optional device (e.g. "cuda:0") to run DP on. If set, stats are moved to device
                layer-by-layer and results moved back to CPU to limit GPU memory.
        min_avg_bits_per_layer: Minimum average bits per dimension per layer (default: 0.0).
            Layers whose budget falls below min_avg_bits_per_layer * dims are floored to that
            value; the remaining budget is redistributed proportionally among other layers.

    Returns:
        bit_alloc: Nested dict bit_alloc[layer][head] = LongTensor [r] of bit allocations per dimension
    """
    run_on_device = device if (device is not None and str(device) != "cpu") else None
    # First, calculate dimensions per layer to determine budget split
    layer_dims = {}
    for layer_name in sorted(sigma2_dict.keys()):
        layer_dims[layer_name] = get_total_dimensions({layer_name: sigma2_dict[layer_name]})
    
    total_dims = sum(layer_dims.values())
    
    if total_dims == 0:
        raise ValueError("No dimensions found in sigma2_dict")
    
    # Split budget across layers
    layer_budgets = {}
    if budget_split == 'proportional':
        for layer_name, dims in layer_dims.items():
            layer_budgets[layer_name] = total_bits * (dims / total_dims)
    elif budget_split == 'equal':
        num_layers = len(layer_dims)
        budget_per_layer = total_bits / num_layers if num_layers > 0 else 0
        for layer_name in layer_dims.keys():
            layer_budgets[layer_name] = budget_per_layer
    elif budget_split == 'importance':
        # Inter-layer budget proportional to Σ(w · σ²) per layer.
        # Dominated by activation scale — prefer 'fisher' mode instead.
        layer_importance = {}
        for layer_name in sorted(sigma2_dict.keys()):
            s2_heads = sigma2_dict[layer_name]
            w_heads  = w_dict.get(layer_name, {})
            total_imp = 0.0
            for h in sorted(s2_heads.keys()):
                s2 = s2_heads[h]
                w  = w_heads.get(h, None)
                if s2 is None:
                    continue
                s2_t = s2.float().flatten() if torch.is_tensor(s2) else torch.tensor(s2, dtype=torch.float32)
                if w is not None and torch.is_tensor(w):
                    w_t = w.float().flatten()
                    n = min(s2_t.numel(), w_t.numel())
                    total_imp += (w_t[:n] * s2_t[:n]).sum().item()
                else:
                    total_imp += s2_t.sum().item()
            layer_importance[layer_name] = max(total_imp, 1e-30)

        total_importance = sum(layer_importance.values())
        for layer_name in sorted(sigma2_dict.keys()):
            imp_frac = layer_importance[layer_name] / total_importance
            dims = layer_dims[layer_name]
            raw_budget = total_bits * imp_frac
            clamped = max(dims * min_bits, min(dims * max_bits, raw_budget))
            layer_budgets[layer_name] = clamped

        allocated_sum = sum(layer_budgets.values())
        if allocated_sum > 0:
            scale = total_bits / allocated_sum
            layer_budgets = {k: v * scale for k, v in layer_budgets.items()}

    elif budget_split in ('fisher', 'log_fisher'):
        # Inter-layer budget proportional to Σ_c w_c (Fisher weights only, no σ²).
        # 'fisher'     : budget ∝ sum(w)        — can be extreme when w spans many orders of magnitude
        # 'log_fisher' : budget ∝ log(sum(w)+1) — dampens outlier layers; moderate tilt toward high-w layers
        layer_importance = {}
        for layer_name in sorted(sigma2_dict.keys()):
            w_heads = w_dict.get(layer_name, {})
            total_w = 0.0
            for h in sorted(w_heads.keys()):
                w = w_heads.get(h, None)
                if w is not None and torch.is_tensor(w):
                    total_w += w.float().flatten().sum().item()
            # Fall back to proportional (by dims) if no Fisher weights available
            raw = max(total_w, 1e-30)
            layer_importance[layer_name] = math.log(raw + 1.0) if budget_split == 'log_fisher' else raw

        total_importance = sum(layer_importance.values())
        for layer_name in sorted(sigma2_dict.keys()):
            imp_frac = layer_importance[layer_name] / total_importance
            dims = layer_dims[layer_name]
            raw_budget = total_bits * imp_frac
            clamped = max(dims * min_bits, min(dims * max_bits, raw_budget))
            layer_budgets[layer_name] = clamped

        # Renormalise to exactly meet total_bits
        allocated_sum = sum(layer_budgets.values())
        if allocated_sum > 0:
            scale = total_bits / allocated_sum
            layer_budgets = {k: v * scale for k, v in layer_budgets.items()}
    else:
        raise ValueError(f"Unknown budget_split mode: {budget_split}")

    # Apply per-layer floor: iteratively fix layers below the floor and
    # redistribute their deficit proportionally among the remaining layers.
    if min_avg_bits_per_layer > 0.0:
        floors = {l: min_avg_bits_per_layer * dims for l, dims in layer_dims.items()}
        fixed = {}
        free = dict(layer_budgets)
        for _ in range(len(layer_dims)):
            newly_fixed = {l: floors[l] for l in list(free) if free[l] < floors[l]}
            if not newly_fixed:
                break
            fixed.update(newly_fixed)
            for l in newly_fixed:
                del free[l]
            budget_for_free = total_bits - sum(fixed.values())
            if budget_for_free <= 0:
                free = {l: floors[l] for l in free}
                break
            free_total = sum(free.values())
            if free_total > 0:
                scale = budget_for_free / free_total
                free = {l: v * scale for l, v in free.items()}
            else:
                share = budget_for_free / len(free) if free else 0
                free = {l: share for l in free}
        layer_budgets = {**fixed, **free}
        n_floored = len(fixed)
        if n_floored > 0:
            logger.info(f"  Per-layer floor ({min_avg_bits_per_layer} bits/dim avg): "
                        f"{n_floored} layer(s) raised to floor")

    logger.info(f"Per-layer bit allocation:")
    logger.info(f"  Total layers: {len(layer_dims)}")
    logger.info(f"  Total dimensions: {total_dims}")
    logger.info(f"  Budget split mode: {budget_split}")
    if use_variance_only:
        logger.info(f"  Using variance-only mode (ignoring importance weights w)")
    
    # Allocate bits for each layer independently
    bit_alloc = {}
    total_allocated = 0.0  # Use float to avoid integer overflow issues
    
    for layer_name in sorted(sigma2_dict.keys()):
        layer_budget = layer_budgets[layer_name]
        layer_dims_count = layer_dims[layer_name]
        
        if layer_dims_count == 0:
            logger.warning(f"Layer {layer_name} has 0 dimensions, skipping...")
            bit_alloc[layer_name] = {}
            continue
        
        # Check if this is a values layer
        is_values_layer = 'v_proj' in layer_name
        layer_verbose = verbose and not is_values_layer  # Suppress verbose output for value layers
        
        if layer_verbose:
            logger.info(f"\n  Layer {layer_name}:")
            logger.info(f"    Dimensions: {layer_dims_count}")
            logger.info(f"    Budget: {layer_budget:.2f} bits ({layer_budget/layer_dims_count:.2f} bits/dim avg)")
        
        # Extract layer-specific statistics
        layer_sigma2 = {layer_name: sigma2_dict[layer_name]}
        layer_w = {layer_name: w_dict.get(layer_name, {})}
        
        # Allocate bits for this layer using the global function
        try:
            # For value layers, pass w_o_dict if available
            layer_w_o = None
            if is_values_layer and w_o_dict is not None:
                layer_w_o = {layer_name: w_o_dict.get(layer_name, {})}
            
            # Optionally move this layer's stats to GPU for faster DP (layer-by-layer to save memory)
            if run_on_device is not None:
                layer_sigma2 = _nested_dict_to_device(layer_sigma2, run_on_device)
                layer_w = _nested_dict_to_device(layer_w, run_on_device)
                if layer_w_o is not None:
                    layer_w_o = _nested_dict_to_device(layer_w_o, run_on_device)
            
            layer_bit_alloc = allocate_bits(
                sigma2_dict=layer_sigma2,
                w_dict=layer_w,
                total_bits=layer_budget,
                min_bits=min_bits,
                max_bits=max_bits,
                eps=eps,
                verbose=layer_verbose,
                use_variance_only=use_variance_only,
                w_o_dict=layer_w_o,
                w_normalization=w_normalization,
                sigma2_normalization=sigma2_normalization,
                w_o_normalization=w_o_normalization
            )

            # Calculate actual allocation for this layer (before top-up)
            layer_allocated = 0.0
            if layer_bit_alloc[layer_name]:
                for bits in layer_bit_alloc[layer_name].values():
                    if bits is not None and bits.numel() > 0:
                        bits_sum = bits.sum().item()
                        layer_allocated += float(bits_sum)

            if layer_verbose:
                logger.info(f"    Allocated: {layer_allocated:.2f} bits ({layer_allocated/layer_budget*100:.1f}% of budget)")

            # Warn if significantly under-allocating
            if layer_allocated < layer_budget * 0.9:
                logger.warning(f"    ⚠️  Layer {layer_name} under-allocating: {layer_allocated/layer_budget*100:.1f}% of budget")

            # Final top-up pass: if under 98% of the budget, distribute remaining bits
            # uniformly across dimensions that are not at max_bits. This prevents layers
            # from stopping early when many dimensions sit at min_bits.
            if layer_allocated < layer_budget * 0.98 and layer_bit_alloc[layer_name]:
                deficit = layer_budget - layer_allocated
                flat_bits = []
                heads_meta = []
                for head_idx, head_bits in layer_bit_alloc[layer_name].items():
                    flat_bits.append(head_bits)
                    heads_meta.append((head_idx, head_bits.numel(), head_bits.shape))
                flat_bits = torch.cat(flat_bits).float()
                mask = flat_bits < max_bits
                if mask.any() and deficit > 0:
                    available = mask.sum().item()
                    add_per_dim = deficit / available
                    increment = torch.zeros_like(flat_bits)
                    increment[mask] = add_per_dim
                    flat_bits = torch.round(flat_bits + increment).clamp(min_bits, max_bits).long()

                    # Re-pack into heads
                    start = 0
                    for head_idx, numel, shp in heads_meta:
                        end = start + numel
                        layer_bit_alloc[layer_name][head_idx] = flat_bits[start:end].reshape(shp)
                        start = end

                    # Recompute allocation after top-up
                    layer_allocated = 0.0
                    for bits in layer_bit_alloc[layer_name].values():
                        if bits is not None and bits.numel() > 0:
                            layer_allocated += float(bits.sum().item())
                    if layer_verbose:
                        logger.info(f"    Top-up allocation: {layer_allocated:.2f} bits ({layer_allocated/layer_budget*100:.1f}% of budget)")

            # Accumulate after top-up so the summary line reflects final values
            total_allocated += layer_allocated
            
            # Store result; move to CPU when using GPU so we don't hold GPU memory across layers
            if run_on_device is not None and layer_bit_alloc[layer_name]:
                bit_alloc[layer_name] = _nested_dict_to_device(layer_bit_alloc[layer_name], "cpu")
                if "cuda" in str(run_on_device):
                    torch.cuda.empty_cache()
            else:
                bit_alloc[layer_name] = layer_bit_alloc[layer_name]
            
        except Exception as e:
            logger.error(f"Failed to allocate bits for layer {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            # Set to min_bits as fallback
            bit_alloc[layer_name] = {}
            for head_idx in sigma2_dict[layer_name].keys():
                rank = len(sigma2_dict[layer_name][head_idx])
                bit_alloc[layer_name][head_idx] = torch.zeros(rank, dtype=torch.long, device=sigma2_dict[layer_name][head_idx].device)
    
    # Safety check: ensure total_allocated is valid
    if total_allocated < 0:
        logger.error(f"ERROR: total_allocated is negative: {total_allocated}. This indicates a bug.")
        logger.error("Recalculating total_allocated from bit_alloc dictionary...")
        total_allocated = 0.0
        for layer_name in bit_alloc:
            if layer_name in bit_alloc:
                for bits in bit_alloc[layer_name].values():
                    if bits is not None and bits.numel() > 0:
                        total_allocated += float(bits.sum().item())
    
    logger.info(f"\n  Total allocated across all layers: {total_allocated:.2f} bits ({total_allocated/total_bits*100:.1f}% of budget)")
    
    return bit_alloc


def allocate_bits(
    sigma2_dict,
    w_dict,
    total_bits,
    min_bits=0,
    max_bits=8,
    eps=1e-12,
    verbose=True,
    use_variance_only=False,
    w_o_dict=None,
    device=None,
    min_avg_bits_per_layer=0.0,
    w_normalization='global',
    sigma2_normalization='global',
    w_o_normalization='global'
):
    """
    Allocate bits using reverse water-filling based on variance and importance statistics.

    Args:
        sigma2_dict: Nested dict sigma2_dict[layer][head] = Tensor [r] of variance statistics
        w_dict: Nested dict w_dict[layer][head] = Tensor [r] of importance statistics
        total_bits: Total bit budget (scalar)
        min_bits: Minimum bits per dimension (default: 0)
        max_bits: Maximum bits per dimension (default: 8)
        eps: Small epsilon to avoid numerical issues (default: 1e-12)
        use_variance_only: If True, use only variance (sigma^2) and ignore importance weights (w)
        min_avg_bits_per_layer: Minimum average bits per dimension per layer (default: 0.0).
            After global water-filling, layers whose total falls below
            min_avg_bits_per_layer * n_dims are raised to that floor; the budget is
            maintained by trimming the cheapest (lowest-score) dims in other layers.
        w_normalization: How to normalize importance weights before scoring (default: 'global').
            - 'global': normalize w by the global 95th percentile across all layers.
            - 'per_layer': normalize w by each layer's own 95th percentile. Removes
              gradient-attenuation bias so early layers are not systematically starved.
        sigma2_normalization: How to normalize variance before scoring (default: 'global').
            - 'global': normalize sigma2 by the global 95th percentile across all layers.
            - 'per_layer': normalize sigma2 by each layer's own 95th percentile. Removes
              activation-scale bias (early layers have smaller activations → smaller sigma2)
              so cross-layer variance differences don't dominate inter-layer allocation.

    Returns:
        bit_alloc: Nested dict bit_alloc[layer][head] = LongTensor [r] of bit allocations per dimension
    """
    # Debug: print structure received
    if verbose:
        logger.debug(f"allocate_bits received: {len(sigma2_dict)} layers in sigma2_dict, {len(w_dict)} layers in w_dict")
        if len(sigma2_dict) > 0:
            first_layer = list(sigma2_dict.keys())[0]
            logger.debug(f"  First layer '{first_layer}': sigma2 type={type(sigma2_dict[first_layer])}, keys={list(sigma2_dict[first_layer].keys()) if isinstance(sigma2_dict[first_layer], dict) else 'N/A'}")
            if first_layer in w_dict:
                logger.debug(f"  First layer '{first_layer}': w type={type(w_dict[first_layer])}, keys={list(w_dict[first_layer].keys()) if isinstance(w_dict[first_layer], dict) else 'N/A'}")
        if use_variance_only:
            logger.info("Using variance-only mode: ignoring importance weights (w)")
    
    # First pass: collect all sigma2 and w values to compute normalization factors
    # For value layers, use w_o_dict if available, otherwise fallback to w_dict
    all_sigma2_values = []
    all_w_values = []
    
    for layer_name in sorted(sigma2_dict.keys()):
        is_values_layer = 'v_proj' in layer_name
        
        # For value layers, prefer w_o_dict if available
        if is_values_layer and w_o_dict is not None and layer_name in w_o_dict:
            layer_w_source = w_o_dict
            if verbose:
                logger.debug(f"Layer {layer_name}: Using w_o_dict for value layer")
        else:
            layer_w_source = w_dict
            if is_values_layer and w_o_dict is not None:
                if verbose:
                    logger.debug(f"Layer {layer_name}: w_o_dict not available, using w_dict for value layer")
        
        if not use_variance_only:
            if layer_name not in layer_w_source:
                if verbose:
                    logger.warning(f"Layer {layer_name} not found in {'w_o_dict' if is_values_layer and w_o_dict is not None else 'w_dict'}, skipping...")
                continue
        layer_sigma2 = sigma2_dict[layer_name]
        layer_w = layer_w_source.get(layer_name, {}) if not use_variance_only else {}
        
        if not isinstance(layer_sigma2, dict):
            if verbose:
                logger.warning(f"Layer {layer_name} sigma2 is not a dict (type={type(layer_sigma2)}), skipping...")
            continue
        if not use_variance_only and not isinstance(layer_w, dict):
            if verbose:
                logger.warning(f"Layer {layer_name} w is not a dict (type={type(layer_w)}), skipping...")
            continue
        
        for head_idx in sorted(layer_sigma2.keys()):
            if not use_variance_only and head_idx not in layer_w:
                if verbose:
                    logger.warning(f"Layer {layer_name} head {head_idx} not found in w_dict, skipping...")
                continue
            sigma2_head = layer_sigma2[head_idx]
            w_head = layer_w.get(head_idx, None) if not use_variance_only else None
            
            if sigma2_head is None:
                if verbose:
                    logger.warning(f"Layer {layer_name} head {head_idx} has None sigma2, skipping...")
                continue
            if not use_variance_only and w_head is None:
                if verbose:
                    logger.warning(f"Layer {layer_name} head {head_idx} has None w, skipping...")
                continue
            
            if not torch.is_tensor(sigma2_head):
                if verbose:
                    logger.warning(f"Layer {layer_name} head {head_idx} sigma2 is not a tensor (type={type(sigma2_head)}), skipping...")
                continue
            if not use_variance_only and not torch.is_tensor(w_head):
                if verbose:
                    logger.warning(f"Layer {layer_name} head {head_idx} w is not a tensor (type={type(w_head)}), skipping...")
                continue
            
            if sigma2_head.numel() == 0:
                if verbose:
                    logger.warning(f"Layer {layer_name} head {head_idx} has empty sigma2 tensor (shape={sigma2_head.shape}), skipping...")
                continue
            if not use_variance_only and w_head.numel() == 0:
                if verbose:
                    logger.warning(f"Layer {layer_name} head {head_idx} has empty w tensor (shape={w_head.shape}), skipping...")
                continue
            
            all_sigma2_values.append(sigma2_head.flatten())
            if not use_variance_only:
                all_w_values.append(w_head.flatten())
    
    if len(all_sigma2_values) == 0:
        # More detailed error message
        layer_details = []
        for layer_name in list(sigma2_dict.keys())[:5]:  # Show first 5 layers
            has_sigma2 = layer_name in sigma2_dict
            has_w = layer_name in w_dict
            if has_sigma2:
                sigma2_type = type(sigma2_dict[layer_name])
                if isinstance(sigma2_dict[layer_name], dict):
                    sigma2_keys = list(sigma2_dict[layer_name].keys())
                    sigma2_shapes = [v.shape if torch.is_tensor(v) else None for v in sigma2_dict[layer_name].values()]
                else:
                    sigma2_keys = "N/A"
                    sigma2_shapes = "N/A"
            else:
                sigma2_type = "missing"
                sigma2_keys = "N/A"
                sigma2_shapes = "N/A"
            layer_details.append(f"{layer_name}: sigma2={has_sigma2}(type={sigma2_type}, keys={sigma2_keys}, shapes={sigma2_shapes}), w={has_w}")
        
        raise ValueError(
            f"No valid statistics found in sigma2_dict and w_dict. "
            f"Processed {len(sigma2_dict)} layers but found no valid tensor pairs. "
            f"Sample layers: {layer_details}. "
            f"Check that statistics are properly loaded and have the expected nested dict structure: layer -> head -> tensor."
        )
    
    # Compute global normalization factors
    all_sigma2 = torch.cat(all_sigma2_values)
    if not use_variance_only:
        all_w = torch.cat(all_w_values)
    
    # Use percentile-based normalization (e.g., 95th percentile) instead of max
    # This better scales small values and is more robust to outliers
    percentile = 95.0
    sigma2_nonzero = all_sigma2[all_sigma2 > eps]
    
    if len(sigma2_nonzero) > 0:
        sigma2_norm = torch.quantile(sigma2_nonzero.float(), percentile / 100.0)
    else:
        sigma2_norm = torch.tensor(1.0)
    
    if not use_variance_only:
        w_nonzero = all_w[all_w > eps]
        if len(w_nonzero) > 0:
            w_norm = torch.quantile(w_nonzero.float(), percentile / 100.0)
        else:
            w_norm = torch.tensor(1.0)
        w_norm = w_norm.clamp(min=eps)
    else:
        w_norm = torch.tensor(1.0)  # Not used in variance-only mode
    
    # Avoid division by zero
    sigma2_norm = sigma2_norm.clamp(min=eps)
    
    # Additional scaling factor to boost small values and prevent underflow
    # Apply this to the final product, not to individual values, to avoid scale_factor^2 effect
    # Use a small factor to prevent underflow while keeping scores in a reasonable range
    # The percentile normalization brings most scores to [0, 1], so a small multiplier helps
    scale_factor = 10.0  # Small scale to prevent underflow without breaking lambda solver
    
    if verbose:
        logger.info(f"Normalizing statistics (using {percentile}th percentile):")
        logger.info(f"  sigma2: {percentile}th percentile={sigma2_norm.item():.2e}, max={all_sigma2.max().item():.2e}")
        if not use_variance_only:
            logger.info(f"  w: {percentile}th percentile={w_norm.item():.2e}, max={all_w.max().item():.2e}")
            if w_norm.item() < 1e-4:
                logger.info(f"  Note: w values are small (normal for squared gradients). They will be normalized appropriately.")
        else:
            logger.info(f"  w: Not used (variance-only mode)")
        logger.info(f"  Scale factor (applied to final product): {scale_factor:.0e}")
    
    # Second pass: compute normalized scores
    all_scores = []
    all_sigma2_for_dp = []  # Store sigma2 for DP algorithm
    index_map = []  # Track (layer, head, dim_idx) for reassembly
    
    # Iterate over layers, then heads
    for layer_name in sorted(sigma2_dict.keys()):
        is_values_layer = 'v_proj' in layer_name
        
        # For value layers, prefer w_o_dict if available
        if is_values_layer and w_o_dict is not None and layer_name in w_o_dict:
            layer_w_source = w_o_dict
        else:
            layer_w_source = w_dict
        
        if not use_variance_only and layer_name not in layer_w_source:
            logger.warning(f"Layer {layer_name} not found in {'w_o_dict' if is_values_layer and w_o_dict is not None else 'w_dict'}, skipping...")
            continue
            
        layer_sigma2 = sigma2_dict[layer_name]
        layer_w = layer_w_source.get(layer_name, {}) if not use_variance_only else {}

        # Compute effective w normalization for this layer.
        # V layers that use w_o use w_o_normalization; K layers use w_normalization.
        using_w_o = is_values_layer and w_o_dict is not None and layer_name in w_o_dict
        effective_w_norm_mode = w_o_normalization if using_w_o else w_normalization
        if effective_w_norm_mode == 'per_layer' and not use_variance_only:
            layer_w_vals = [v.flatten() for v in layer_w.values()
                            if v is not None and torch.is_tensor(v) and v.numel() > 0]
            if layer_w_vals:
                lw_cat = torch.cat(layer_w_vals)
                lw_nonzero = lw_cat[lw_cat > eps]
                w_norm_eff = torch.quantile(lw_nonzero.float(), percentile / 100.0).clamp(min=eps) \
                    if len(lw_nonzero) > 0 else w_norm
            else:
                w_norm_eff = w_norm
        else:
            w_norm_eff = w_norm

        # Compute effective sigma2 normalization for this layer
        if sigma2_normalization == 'per_layer':
            layer_s2_vals = [v.flatten() for v in layer_sigma2.values()
                             if v is not None and torch.is_tensor(v) and v.numel() > 0]
            if layer_s2_vals:
                ls2_cat = torch.cat(layer_s2_vals)
                ls2_nonzero = ls2_cat[ls2_cat > eps]
                sigma2_norm_eff = torch.quantile(ls2_nonzero.float(), percentile / 100.0).clamp(min=eps) \
                    if len(ls2_nonzero) > 0 else sigma2_norm
            else:
                sigma2_norm_eff = sigma2_norm
        else:
            sigma2_norm_eff = sigma2_norm

        for head_idx in sorted(layer_sigma2.keys()):
            if not use_variance_only and head_idx not in layer_w:
                logger.warning(f"Layer {layer_name}, Head {head_idx} not found in w_dict, skipping...")
                continue
            
            sigma2_head = layer_sigma2[head_idx]
            w_head = layer_w.get(head_idx, None) if not use_variance_only else None
            
            if sigma2_head is None:
                logger.warning(f"Layer {layer_name}, Head {head_idx} has None sigma2, skipping...")
                continue
            if not use_variance_only and w_head is None:
                logger.warning(f"Layer {layer_name}, Head {head_idx} has None w, skipping...")
                continue
            
            # Normalize sigma2 and w using percentile-based normalization
            # Clamp normalized values to prevent overflow to inf
            # Use dtype-appropriate max value (float16 max is ~65504, float32 can handle larger)
            dtype_max = torch.finfo(sigma2_head.dtype).max
            # Use a safe fraction of max to avoid overflow (use 1/10th of max for safety)
            max_normalized_value = min(1e6, dtype_max / 10.0)
            
            # Handle zero sigma2: dimensions with zero variance (constant/zero) should get minimum bits
            # Use a small epsilon to ensure they still get some score for the lambda solver
            sigma2_head_safe = sigma2_head.clamp(min=eps)  # Replace zeros with eps
            sigma2_normalized = (sigma2_head_safe / sigma2_norm_eff).clamp(max=max_normalized_value)
            
            if use_variance_only:
                # In variance-only mode, use only normalized variance as the score
                score = sigma2_normalized
            else:
                # Handle zero w values: use a small epsilon to ensure variance-only dimensions get some allocation
                # For dimensions with w=0 (no gradients), we still want to allocate bits based on variance
                w_head_safe = w_head.clamp(min=eps)  # Replace zeros with eps
                w_normalized = (w_head_safe / w_norm_eff).clamp(max=max_normalized_value)
                
                # Compute score = (normalized w) * (normalized sigma^2)
                # For dimensions with zero w, we use a small weight (eps/w_norm) so they still get allocation
                # based on variance, but less than dimensions with non-zero gradients
                score = w_normalized * sigma2_normalized
                
                # For dimensions that had zero w originally, add a small variance-only component
                # This ensures dimensions with high variance but zero gradients still get some bits
                # Use a small weight (e.g., 1% of the normalized variance) for variance-only allocation
                zero_w_mask = (w_head <= eps)
                if zero_w_mask.any():
                    variance_only_weight = 0.01  # Small weight for variance-only dimensions
                    score[zero_w_mask] = score[zero_w_mask].clamp(min=variance_only_weight * sigma2_normalized[zero_w_mask])
            
            # For dimensions with zero sigma2 (constant/zero dimensions), ensure they get minimum score
            # These will get min_bits allocation, but we need a non-zero score for the lambda solver
            zero_sigma2_mask = (sigma2_head <= eps)
            if zero_sigma2_mask.any():
                # Use a very small score for zero-variance dimensions so they get min_bits
                # This is much smaller than even the variance-only weight to ensure they're lowest priority
                min_score_for_zero_variance = eps * 0.1  # Very small score
                score[zero_sigma2_mask] = score[zero_sigma2_mask].clamp(min=min_score_for_zero_variance)
            
            # Apply scale_factor only if needed to boost very small products
            # But keep it small to avoid making scores too large
            if scale_factor != 1.0:
                score = score * scale_factor
            
            # Clamp very small values to eps and very large values to prevent inf
            # Use dtype-appropriate max value to prevent overflow
            dtype_max = torch.finfo(score.dtype).max
            score_max_safe = min(1e8, dtype_max / 10.0)
            score = score.clamp(min=eps, max=score_max_safe)
            
            # Replace any remaining inf or NaN with finite values
            score = torch.nan_to_num(score, nan=eps, posinf=score_max_safe, neginf=eps)
            
            # Ensure score is positive and has correct shape
            if score.numel() == 0:
                logger.warning(f"Layer {layer_name}, Head {head_idx} has empty score tensor, skipping...")
                continue
            
            # Debug: check for zero sigma2 or zero w (only log if significant zeros)
            num_zero_sigma2 = (sigma2_head <= eps).sum().item()
            num_zero_score = (score <= eps).sum().item()
            if not use_variance_only:
                num_zero_w = (w_head <= eps).sum().item()
                if num_zero_sigma2 > len(score) * 0.1 or num_zero_w > len(score) * 0.1:  # Log if >10% zeros in either
                    logger.debug(f"Layer {layer_name}, Head {head_idx}: {num_zero_sigma2}/{len(score)} zero sigma2, "
                               f"{num_zero_w}/{len(score)} zero w, {num_zero_score}/{len(score)} zero scores")
            else:
                if num_zero_sigma2 > len(score) * 0.1:  # Log if >10% zeros
                    logger.debug(f"Layer {layer_name}, Head {head_idx}: {num_zero_sigma2}/{len(score)} zero sigma2, "
                               f"{num_zero_score}/{len(score)} zero scores")
            if num_zero_sigma2 > len(score) * 0.3:
                logger.warning(f"  ⚠️  {num_zero_sigma2}/{len(score)} dimensions ({num_zero_sigma2/len(score)*100:.1f}%) "
                             f"have zero variance (constant/zero). These will get min_bits allocation.")
            
            all_scores.append(score)
            all_sigma2_for_dp.append(sigma2_head.flatten())  # Store sigma2 for DP
            
            # Track indices for reassembly
            for dim_idx in range(len(score)):
                index_map.append((layer_name, head_idx, dim_idx))
    
    if len(all_scores) == 0:
        raise ValueError("No valid statistics found in sigma2_dict and w_dict")
    
    # Concatenate all scores
    # Use dtype-appropriate max value to prevent overflow
    dtype_max = torch.finfo(all_scores[0].dtype).max if len(all_scores) > 0 else torch.finfo(torch.float32).max
    scores_max_safe = min(1e8, dtype_max / 10.0)
    scores = torch.cat(all_scores).clamp(min=eps, max=scores_max_safe)
    
    # Final safety check: replace any inf or NaN with finite values
    scores = torch.nan_to_num(scores, nan=eps, posinf=scores_max_safe, neginf=eps)
    
    # Verify scores are finite before proceeding
    if torch.isinf(scores).any() or torch.isnan(scores).any():
        num_inf = torch.isinf(scores).sum().item()
        num_nan = torch.isnan(scores).sum().item()
        logger.warning(f"Found {num_inf} inf and {num_nan} NaN values in scores after normalization. "
                      f"Replacing with finite values.")
        scores = torch.nan_to_num(scores, nan=eps, posinf=1e8, neginf=eps)
    
    if verbose:
        logger.info(f"Allocating {total_bits} bits across {len(scores)} dimensions")
        # Use finite values for logging to avoid inf in output
        scores_finite = scores[torch.isfinite(scores)]
        if len(scores_finite) > 0:
            logger.info(f"Score range: [{scores_finite.min().item():.2e}, {scores_finite.max().item():.2e}]")
            logger.info(f"Score mean: {scores_finite.mean().item():.2e}, median: {scores_finite.median().item():.2e}")
        else:
            logger.warning(f"All scores are non-finite! Using fallback values.")
            logger.info(f"Score range: [{eps:.2e}, {eps:.2e}]")
            logger.info(f"Score mean: {eps:.2e}, median: {eps:.2e}")
        
        # Check for zero scores
        num_zero_scores = (scores <= eps).sum().item()
        num_nonzero_scores = (scores > eps).sum().item()
        logger.info(f"Zero scores (<= {eps:.2e}): {num_zero_scores} ({num_zero_scores/len(scores)*100:.1f}%)")
        logger.info(f"Non-zero scores: {num_nonzero_scores} ({num_nonzero_scores/len(scores)*100:.1f}%)")
    else:
        num_zero_scores = (scores <= eps).sum().item()
        num_nonzero_scores = (scores > eps).sum().item()
    
    if num_zero_scores > len(scores) * 0.5:
        logger.warning(f"More than 50% of dimensions have zero scores! This may indicate:")
        logger.warning(f"  - Many dimensions have zero variance (sigma2=0)")
        if not use_variance_only:
            logger.warning(f"  - Many dimensions have zero importance (w=0)")
        logger.warning(f"  - Statistics collection may have issues")
        logger.warning(f"  - Consider checking RD calibration statistics")
    
    # Global Lagrangian water-filling allocation (GPU-friendly, O(n * 50 iters)).
    # Replaces the per-dim DP which required an (n+1)×(budget+1) table — infeasible
    # for the joint-global case where n can be 30k–60k dims.
    if device is not None:
        scores = scores.to(device)

    if verbose:
        logger.info(f"Using Lagrangian water-filling for global bit allocation")
        logger.info(f"  Dimensions: {len(scores)}, Total bits: {total_bits}, Bit range: [{min_bits}, {max_bits}]")

    total_bits_int = int(total_bits)
    lambda_star = _solve_lambda(scores, total_bits_int, min_bits, max_bits)

    ratio = (scores / lambda_star).clamp(min=1e-12)
    b_float = (0.5 * torch.log2(ratio)).clamp(min=min_bits, max=max_bits)
    b = b_float.round().clamp(min_bits, max_bits).long()

    # Greedy top-up / trim to hit budget exactly (Lagrangian rounding can be off by O(1) bits)
    deficit = total_bits_int - int(b.sum().item())
    if deficit > 0:
        frac = b_float - b.float()
        upgradeable = (b < max_bits)
        frac_adj = torch.where(upgradeable, frac, torch.full_like(frac, -float('inf')))
        n_up = int(upgradeable.sum().item())
        _, idx = torch.topk(frac_adj, min(deficit, n_up))
        b[idx] = (b[idx] + 1).clamp(max=max_bits)
    elif deficit < 0:
        frac = b_float - b.float()
        downgradeable = (b > min_bits)
        frac_adj = torch.where(downgradeable, frac, torch.full_like(frac, float('inf')))
        n_dn = int(downgradeable.sum().item())
        _, idx = torch.topk(frac_adj, min(-deficit, n_dn), largest=False)
        b[idx] = (b[idx] - 1).clamp(min=min_bits)

    b = b.clamp(min_bits, max_bits)

    # Uniform top-up: when many dims have near-zero scores, the single-pass greedy
    # fix above can only add at most n_upgradeable bits (1 per dim). Any remaining
    # large deficit is left unspent. Distribute it uniformly across dims below max_bits.
    large_deficit = total_bits_int - int(b.sum().item())
    if large_deficit > 0:
        upgradeable = (b < max_bits)
        available = int(upgradeable.sum().item())
        if available > 0:
            add_per_dim = large_deficit / available
            increment = torch.zeros(len(b), dtype=torch.float32, device=b.device)
            increment[upgradeable] = add_per_dim
            b = torch.round(b.float() + increment).clamp(min_bits, max_bits).long()
            # Single-bit trim/top-up to correct for rounding
            residual = total_bits_int - int(b.sum().item())
            if residual > 0:
                up2 = (b < max_bits)
                n_up2 = int(up2.sum().item())
                frac2 = increment - increment.round()
                frac2_adj = torch.where(up2, frac2, torch.full_like(frac2, -float('inf')))
                _, idx2 = torch.topk(frac2_adj, min(residual, n_up2))
                b[idx2] = (b[idx2] + 1).clamp(max=max_bits)
            elif residual < 0:
                dn2 = (b > min_bits)
                n_dn2 = int(dn2.sum().item())
                frac2 = increment - increment.round()
                frac2_adj = torch.where(dn2, frac2, torch.full_like(frac2, float('inf')))
                _, idx2 = torch.topk(frac2_adj, min(-residual, n_dn2), largest=False)
                b[idx2] = (b[idx2] - 1).clamp(min=min_bits)
        b = b.clamp(min_bits, max_bits)

    if device is not None:
        b = b.cpu()

    # Per-layer floor: raise layers whose total allocation is below
    # min_avg_bits_per_layer * n_dims, compensating by trimming the lowest-score
    # dims in layers that are above their floor. Iterates until all layers meet the floor.
    if min_avg_bits_per_layer > 0.0:
        scores_cpu = scores.cpu()
        layer_pos = {}
        for pos, (lname, _, _) in enumerate(index_map):
            if lname not in layer_pos:
                layer_pos[lname] = []
            layer_pos[lname].append(pos)

        for _ in range(len(layer_pos)):
            underfunded = {}
            for lname, positions in layer_pos.items():
                floor_l = math.ceil(min_avg_bits_per_layer * len(positions))
                if int(b[positions].sum().item()) < floor_l:
                    underfunded[lname] = floor_l
            if not underfunded:
                break

            total_added = 0

            # Raise underfunded layers via local water-filling with budget = floor.
            # This is optimal for the new budget and correctly accounts for existing b_i.
            for lname, floor_l in underfunded.items():
                pos_t = torch.tensor(layer_pos[lname])
                old_total = int(b[pos_t].sum().item())
                layer_scores = scores_cpu[pos_t]

                lam = _solve_lambda(layer_scores, floor_l, min_bits, max_bits)
                ratio = (layer_scores / lam).clamp(min=1e-12)
                b_float_l = (0.5 * torch.log2(ratio)).clamp(min_bits, max_bits)
                b_new = b_float_l.round().clamp(min_bits, max_bits).long()

                # Greedy top-up / trim to hit floor_l exactly
                d = floor_l - int(b_new.sum().item())
                if d > 0:
                    frac = b_float_l - b_new.float()
                    up = b_new < max_bits
                    frac_adj = torch.where(up, frac, torch.full_like(frac, -float('inf')))
                    _, idx = torch.topk(frac_adj, min(d, int(up.sum().item())))
                    b_new[idx] = (b_new[idx] + 1).clamp(max=max_bits)
                elif d < 0:
                    frac = b_float_l - b_new.float()
                    dn = b_new > min_bits
                    frac_adj = torch.where(dn, frac, torch.full_like(frac, float('inf')))
                    _, idx = torch.topk(frac_adj, min(-d, int(dn.sum().item())), largest=False)
                    b_new[idx] = (b_new[idx] - 1).clamp(min=min_bits)

                b[pos_t] = b_new
                total_added += int(b_new.sum().item()) - old_total

            # Compensate: trim lowest-score dims from layers above their floor
            surplus_pos = []
            for lname, positions in layer_pos.items():
                if lname in underfunded:
                    continue
                pos_t = torch.tensor(positions)
                floor_l = math.ceil(min_avg_bits_per_layer * len(positions))
                if int(b[pos_t].sum().item()) > floor_l:
                    downgradeable = (b[pos_t] > min_bits).tolist()
                    surplus_pos.extend(p for p, ok in zip(positions, downgradeable) if ok)
            if surplus_pos:
                sp_t = torch.tensor(surplus_pos)
                n_dn = min(total_added, len(surplus_pos))
                _, bot = torch.topk(scores_cpu[sp_t], n_dn, largest=False)
                b[sp_t[bot]] = (b[sp_t[bot]] - 1).clamp(min=min_bits)

        b = b.clamp(min_bits, max_bits)
        if verbose:
            n_floored = sum(
                1 for lname, positions in layer_pos.items()
                if int(b[positions].sum().item()) >= math.ceil(min_avg_bits_per_layer * len(positions))
            )
            logger.info(f"Per-layer floor ({min_avg_bits_per_layer} bits/dim avg): "
                        f"{n_floored}/{len(layer_pos)} layers meet floor after adjustment")

    actual_total = b.sum().item()
    if verbose:
        logger.info(f"Allocation complete: {actual_total} bits ({actual_total/total_bits*100:.1f}% of budget)")
        logger.info(f"  Bit range: [{b.min().item()}, {b.max().item()}]")
        logger.info(f"  Dims at {min_bits} bits: {(b == min_bits).sum().item()}, at {max_bits} bits: {(b == max_bits).sum().item()}")
    
    # Reassemble per layer and head
    bit_alloc = {}
    ptr = 0
    
    for layer_name in sorted(sigma2_dict.keys()):
        if not use_variance_only and layer_name not in w_dict:
            continue
            
        layer_sigma2 = sigma2_dict[layer_name]
        layer_w = w_dict.get(layer_name, {}) if not use_variance_only else {}
        bit_alloc[layer_name] = {}
        
        for head_idx in sorted(layer_sigma2.keys()):
            if not use_variance_only and head_idx not in layer_w:
                continue
            
            sigma2_head = layer_sigma2[head_idx]
            if sigma2_head is None:
                continue
            
            rank = len(sigma2_head)
            head_bits = b[ptr:ptr + rank].clone()  # Clone to avoid view issues
            
            # Validate bit allocation for this head
            if torch.isnan(head_bits.float()).any() or torch.isinf(head_bits.float()).any():
                logger.warning(f"Layer {layer_name}, Head {head_idx}: Bit allocation contains NaN/Inf, setting to {min_bits}")
                head_bits = torch.full((rank,), min_bits, dtype=torch.long, device=head_bits.device)
            
            # Check for integer overflow (very large negative values indicate corruption)
            head_min = head_bits.min().item()
            head_max = head_bits.max().item()
            if head_min < -1000 or head_max > 1000 or head_min < min_bits or head_max > max_bits:
                logger.warning(f"Layer {layer_name}, Head {head_idx}: Bit allocation contains invalid values "
                             f"[{head_min}, {head_max}], clamping to valid range [{min_bits}, {max_bits}]")
                head_bits = torch.clamp(head_bits, min=min_bits, max=max_bits).long()
            
            bit_alloc[layer_name][head_idx] = head_bits
            ptr += rank
    
    return bit_alloc


def get_total_dimensions(sigma2_dict):
    """
    Get total number of dimensions across all layers and heads.
    
    Args:
        sigma2_dict: Nested dict sigma2_dict[layer][head] = Tensor [r]
        
    Returns:
        Total number of dimensions
    """
    total = 0
    for layer_name in sigma2_dict:
        for head_idx in sigma2_dict[layer_name]:
            if sigma2_dict[layer_name][head_idx] is not None:
                total += len(sigma2_dict[layer_name][head_idx])
    return total


def calculate_current_bit_usage(sigma2_dict, bits_per_dimension=16, seq_len=2048, batch_size=1):
    """
    Calculate current bit usage for unquantized low-rank KV cache.
    
    The KV cache stores Z = VT(x) which has shape [batch_size, seq_len, sum(ranks)] per layer.
    
    IMPORTANT: Bit allocation allocates bits PER DIMENSION. So if we have 8000 dimensions
    and want 8 bits per dimension on average, the total_bits budget should be 8000 * 8 = 64,000.
    We don't multiply by seq_len because allocation is per dimension, not per token.
    
    Args:
        sigma2_dict: Nested dict sigma2_dict[layer][head] = Tensor [r] 
                     (used to determine dimensions per layer)
        bits_per_dimension: Current bits per dimension (default: 16 for float16)
        seq_len: Sequence length (for display purposes only)
        batch_size: Batch size (for display purposes only)
        
    Returns:
        Dictionary with:
            - 'total_bits_for_allocation': Total bits budget for allocation (per dimension)
            - 'total_bits_kv_cache': Total bits for full KV cache (for reference)
            - 'bits_per_dimension': Current bits per dimension
            - 'total_dimensions': Total dimensions across all layers
            - 'bits_per_token': Bits per token position (all layers)
            - 'num_layers': Number of layers
    """
    total_dims = get_total_dimensions(sigma2_dict)
    
    # Count layers (assuming each unique layer name is a separate layer)
    num_layers = len(sigma2_dict)
    
    # Calculate bits per token position (all layers combined)
    # Each token position stores sum(ranks) dimensions per layer
    bits_per_token = total_dims * bits_per_dimension
    
    # Total bits for full KV cache (for reference/display)
    total_bits_kv_cache = bits_per_token * seq_len * batch_size
    
    # Total bits for allocation = bits_per_dimension * total_dimensions
    # This is what we pass to allocate_bits() - it allocates bits per dimension
    total_bits_for_allocation = total_dims * bits_per_dimension
    
    # Average bits per layer
    bits_per_layer = total_bits_kv_cache / num_layers if num_layers > 0 else 0
    
    return {
        'total_bits_for_allocation': total_bits_for_allocation,
        'total_bits_kv_cache': total_bits_kv_cache,
        'bits_per_layer': bits_per_layer,
        'bits_per_dimension': bits_per_dimension,
        'total_dimensions': total_dims,
        'bits_per_token': bits_per_token,
        'num_layers': num_layers,
        'seq_len': seq_len,
        'batch_size': batch_size
    }


def suggest_bit_budget(sigma2_dict, compression_ratio=1.0, bits_per_dimension=16, 
                      seq_len=2048, batch_size=1):
    """
    Suggest meaningful bit budgets based on current usage and desired compression.
    
    IMPORTANT: Bit allocation allocates bits PER DIMENSION. So if we have 8000 dimensions
    and want 8 bits per dimension on average (50% compression from 16 bits), the budget
    should be: 8000 * 8 = 64,000 bits.
    
    Args:
        sigma2_dict: Nested dict sigma2_dict[layer][head] = Tensor [r]
        compression_ratio: Desired compression ratio (e.g., 0.5 for 50% of original bits per dimension)
        bits_per_dimension: Current bits per dimension (default: 16 for float16)
        seq_len: Sequence length (for display purposes only)
        batch_size: Batch size (for display purposes only)
        
    Returns:
        Dictionary with suggested budgets and current usage info
    """
    current = calculate_current_bit_usage(sigma2_dict, bits_per_dimension, seq_len, batch_size)
    
    # Calculate suggested bits per dimension
    suggested_bits_per_dim = bits_per_dimension * compression_ratio
    
    # Total budget for allocation = bits_per_dim * num_dimensions
    suggested_total_bits = current['total_dimensions'] * suggested_bits_per_dim
    
    return {
        'current_usage': current,
        'suggested_total_bits': suggested_total_bits,
        'suggested_bits_per_dimension': suggested_bits_per_dim,
        'compression_ratio': compression_ratio,
        'avg_bits_per_dimension': suggested_bits_per_dim,
        'suggestions': {
            'conservative': current['total_dimensions'] * bits_per_dimension * 0.8,
            'moderate': current['total_dimensions'] * bits_per_dimension * 0.5,
            'aggressive': current['total_dimensions'] * bits_per_dimension * 0.25,
        }
    }


def print_bit_allocation_summary(bit_alloc, show_values_detail=False):
    """
    Print a summary of bit allocations.
    
    Args:
        bit_alloc: Nested dict bit_alloc[layer][head] = LongTensor [r]
        show_values_detail: If False, only show summary for v_proj layers (default: False)
    """
    print("\n" + "=" * 80)
    print("Bit Allocation Summary")
    print("=" * 80)
    
    total_bits = 0.0  # Use float to avoid integer overflow
    total_dims = 0
    
    for layer_name in sorted(bit_alloc.keys()):
        layer_bits = 0.0  # Use float to avoid integer overflow
        layer_dims = 0
        
        # Check if this is a values layer (v_proj)
        is_values_layer = 'v_proj' in layer_name
        show_detail = not is_values_layer or show_values_detail
        
        for head_idx in sorted(bit_alloc[layer_name].keys()):
            bits = bit_alloc[layer_name][head_idx]
            if bits is None:
                if show_detail:
                    logger.warning(f"  {layer_name:50s} Head {head_idx:2d}: No bit allocation data")
                continue
            
            # Safely calculate sum, handling potential integer overflow
            try:
                # Convert to float before summing to avoid integer overflow
                head_bits = float(bits.float().sum().item())
            except (OverflowError, RuntimeError) as e:
                if show_detail:
                    logger.error(f"  {layer_name:50s} Head {head_idx:2d}: Error calculating bits sum: {e}")
                    # Try to detect invalid values
                    if torch.isnan(bits).any() or torch.isinf(bits.float()).any():
                        logger.error(f"    Contains NaN/Inf values!")
                    if (bits < 0).any():
                        logger.error(f"    Contains negative values: min={bits.min().item()}")
                head_bits = 0.0  # Fallback
            
            head_dims = len(bits)
            
            # Check for invalid values
            if head_bits < 0 or head_bits > 1e10:
                if show_detail:
                    logger.warning(f"  {layer_name:50s} Head {head_idx:2d}: Invalid bit sum {head_bits}, "
                                 f"bits range: [{bits.min().item()}, {bits.max().item()}]")
                # Try to recalculate by summing individual values as floats
                head_bits = sum(float(b.item()) for b in bits.flatten())
            
            layer_bits += head_bits
            layer_dims += head_dims
            
            # Only print per-head details if show_detail is True
            if show_detail:
                print(f"  {layer_name:50s} Head {head_idx:2d}: {head_bits:8.2f} bits ({head_dims} dims, "
                      f"avg: {head_bits/head_dims:.2f} bits/dim)")
        
        total_bits += layer_bits
        total_dims += layer_dims
        
        # Always print layer total
        print(f"  {layer_name:50s} TOTAL:     {layer_bits:8.2f} bits ({layer_dims} dims)")
        print("-" * 80)
    
    print(f"\nGrand Total: {total_bits:.2f} bits across {total_dims} dimensions")
    print(f"Average: {total_bits/total_dims:.2f} bits per dimension")
    print("=" * 80 + "\n")

