import gc
import os
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from loguru import logger
from palu.data_utils import get_calib_data


class RDCapture:
    def __init__(self):
        self.sigma2 = defaultdict(lambda: None)
        self.w = defaultdict(lambda: None)
        self.count = defaultdict(int)

    def register(self, module, name):
        """
        module: HeadwiseLowRankModule
        name: layer identifier
        """
        # Hook into the VT layer to capture Z = VT(x) which is part of the computation graph
        def vt_forward_hook(vt_layer, inputs, output):
            # output is Z = VT(x), which is part of the computation graph
            Z = output  # [B, T, r] or [T, r] where r = sum(ranks)
            
            # Debug: check if Z has any non-zero values
            if self.count[name] == 0:  # Only log on first call
                logger.debug(f"Forward hook for {name}: Z shape={Z.shape}, "
                           f"non-zero={Z.abs().gt(1e-8).sum().item()}/{Z.numel()}, "
                           f"max={Z.abs().max().item():.2e}, mean={Z.abs().mean().item():.2e}")
            
            # Reshape for processing
            if Z.dim() == 3:
                Z_flat = Z.reshape(-1, Z.shape[-1])
            elif Z.dim() == 2:
                Z_flat = Z
            else:
                logger.warning(f"Unexpected Z shape for {name}: {Z.shape}")
                Z_flat = Z.reshape(-1, Z.shape[-1]) if Z.numel() > 0 else Z
            
            # Initialize sigma2 if needed
            if self.sigma2[name] is None:
                self.sigma2[name] = torch.zeros(Z_flat.shape[-1], device=Z.device, dtype=Z.dtype)
            
            # Accumulate variance (use detached to save memory)
            # We compute E[Z^2] = mean(Z^2) per dimension
            Z_squared_mean = Z_flat.detach().pow(2).mean(dim=0)
            self.sigma2[name] += Z_squared_mean
            self.count[name] += 1
            
            # Debug: check for zero dimensions
            if self.count[name] == 1:  # Only log on first call
                num_zero_dims = (Z_squared_mean <= 1e-10).sum().item()
                if num_zero_dims > 0:
                    logger.debug(f"  {name}: {num_zero_dims}/{len(Z_squared_mean)} dimensions have zero variance in first batch")
            
            # Retain gradient on Z so we can access Z.grad in backward hook
            if Z.requires_grad:
                Z.retain_grad()
                # Store reference to Z for backward hook
                if not hasattr(module, "_last_Z_list"):
                    module._last_Z_list = []
                module._last_Z_list.append(Z)
            else:
                # Try to enable gradients (may not work if computation graph doesn't support it)
                logger.warning(f"Z for {name} doesn't require grad, trying to enable...")
                Z.requires_grad_(True)
                Z.retain_grad()
                if not hasattr(module, "_last_Z_list"):
                    module._last_Z_list = []
                module._last_Z_list.append(Z)
        
        def vt_backward_hook(vt_layer, grad_input, grad_output):
            # grad_output[0] is the gradient of loss w.r.t. Z = VT(x)
            # This is what we want for w statistics
            if grad_output is None or len(grad_output) == 0 or grad_output[0] is None:
                logger.debug(f"Backward hook for {name}: grad_output is None or empty")
                return
            
            gradZ = grad_output[0]  # Gradient w.r.t. Z
            
            # Check if gradient is actually non-zero
            if gradZ.abs().max().item() < 1e-10:
                logger.debug(f"Backward hook for {name}: gradient is essentially zero (max={gradZ.abs().max().item():.2e})")
                # Still accumulate, but log a warning if it's consistently zero
            
            # Reshape to match Z
            if gradZ.dim() == 3:
                gradZ_flat = gradZ.reshape(-1, gradZ.shape[-1])
            else:
                gradZ_flat = gradZ.reshape(-1, gradZ.shape[-1]) if gradZ.dim() == 2 else gradZ
            
            # Initialize w if needed
            if self.w[name] is None:
                self.w[name] = torch.zeros(gradZ_flat.shape[-1], device=gradZ.device, dtype=gradZ.dtype)
            
            # Accumulate squared gradients (importance statistics)
            self.w[name] += gradZ_flat.pow(2).mean(dim=0)
        
        # Register hooks on the VT layer (which is a nn.Linear)
        module.VT.register_forward_hook(vt_forward_hook)
        module.VT.register_full_backward_hook(vt_backward_hook)


def extract_w_o_weights(model):
    """
    Extract W_O (output projection) weights from model and compute ||W_O[:, i]||²
    for each value latent dimension i.
    
    Args:
        model: The model with LlamaPaluAttention modules
        
    Returns:
        w_o_dict: Dictionary w_o_dict[layer_name][head_idx] = Tensor[r] 
                 containing ||W_O[:, i]||² for each value latent dimension i
    """
    w_o_dict = {}
    
    for name, module in model.named_modules():
        # Check if this is a LlamaPaluAttention module
        if hasattr(module, 'v_proj') and hasattr(module, 'o_proj'):
            # Check if v_proj is a HeadwiseLowRankModule
            if hasattr(module.v_proj, 'VT') and isinstance(module.o_proj, nn.Linear):
                # Get o_proj weight matrix: shape (hidden_size, fused_hidden_dim_o)
                o_proj_weight = module.o_proj.weight.data  # (hidden_size, group_rank_v * num_heads)
                
                # Get value projection dimensions
                v_proj = module.v_proj
                total_rank_v = sum(v_proj.ranks)  # Total value latent dimensions
                
                # The o_proj input dimension should match the fused dimension
                # In fusion version: fused_hidden_dim_o = group_rank_v * num_heads
                # Each value latent dimension maps to one column in o_proj input
                # (after going through U_list, but since it's fused, we can directly use o_proj columns)
                
                # For each value latent dimension i, compute ||W_O[:, i]||²
                # Since o_proj has been fused with v_proj.U_list, each latent dimension
                # corresponds to one column in o_proj
                if o_proj_weight.shape[1] >= total_rank_v:
                    # Compute squared norm for each column (each value latent dimension)
                    w_o_values = o_proj_weight[:, :total_rank_v].norm(dim=0).pow(2)  # (total_rank_v,)
                else:
                    # If dimensions don't match, we need to handle the mapping differently
                    # This might happen if fusion wasn't done or structure is different
                    logger.warning(
                        f"Layer {name}: o_proj input dim ({o_proj_weight.shape[1]}) "
                        f"doesn't match expected value rank ({total_rank_v}). "
                        f"Using available columns only."
                    )
                    # Use available columns and pad with zeros if needed
                    w_o_values = o_proj_weight.norm(dim=0).pow(2)  # (available_dims,)
                    if w_o_values.shape[0] < total_rank_v:
                        # Pad with zeros for missing dimensions
                        padding = torch.zeros(
                            total_rank_v - w_o_values.shape[0],
                            device=w_o_values.device,
                            dtype=w_o_values.dtype
                        )
                        w_o_values = torch.cat([w_o_values, padding])
                    else:
                        # Truncate if too many
                        w_o_values = w_o_values[:total_rank_v]
                
                # Store in the same structure as w_dict: layer_name -> {head_idx: tensor}
                # For now, we'll use head_idx=0 as default (can be extended for multi-head if needed)
                if name not in w_o_dict:
                    w_o_dict[name] = {}
                w_o_dict[name][0] = w_o_values.cpu()
                
                logger.debug(
                    f"Extracted W_O weights for {name}: shape={w_o_values.shape}, "
                    f"min={w_o_values.min().item():.2e}, max={w_o_values.max().item():.2e}, "
                    f"mean={w_o_values.mean().item():.2e}"
                )
    
    return w_o_dict




def _cache_key_for_layer_name(layer_name: str) -> str:
    """Normalize layer name for cache key (e.g. .self_attn.inner. -> .self_attn.)."""
    if layer_name is None:
        return layer_name
    if ".self_attn.inner." in layer_name:
        return layer_name.replace(".self_attn.inner.", ".self_attn.", 1)
    return layer_name


def _layer_suffix_for_match(layer_name: str) -> str:
    """Return a short suffix for matching across different key prefixes (e.g. base_model.model.... vs model....)."""
    if layer_name is None:
        return layer_name
    key = _cache_key_for_layer_name(layer_name)
    # Strip leading prefix so "base_model.model.layers.0..." and "model.layers.0..." both -> "layers.0..."
    for prefix in ("base_model.model.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    return key


def _canonicalize_stat_dict(raw_dict, prefer_non_none=True):
    """
    Convert a raw stats dict (layer_name -> tensor or dict) to use canonical layer keys
    so that sigma2 and w always share the same key set (e.g. .inner. vs no .inner.).
    If two raw keys map to the same canonical key, keep the one with non-None data when prefer_non_none.
    """
    if not isinstance(raw_dict, dict):
        return raw_dict
    out = {}
    for k, v in raw_dict.items():
        ck = _cache_key_for_layer_name(k)
        if ck not in out:
            out[ck] = v
        elif prefer_non_none and (v is not None and (torch.is_tensor(v) and v.numel() > 0) or (isinstance(v, dict) and any(x is not None for x in v.values()))):
            if out[ck] is None or (torch.is_tensor(out[ck]) and out[ck].numel() == 0) or (isinstance(out[ck], dict) and not any(x is not None for x in out[ck].values())):
                out[ck] = v
    return out


def _sanitize_finite_stat_dict(stat_dict, tensor_max=1e10, tensor_min=1e-10, log_prefix="RD stats"):
    """Replace inf/nan in tensor values so downstream bit allocation never sees non-finite stats."""
    if not stat_dict:
        return stat_dict
    out = {}
    layers_sanitized = []
    for k, v in stat_dict.items():
        if v is None or not torch.is_tensor(v):
            out[k] = v
            continue
        v = v.float()
        need_fix = ~torch.isfinite(v)
        need_clamp = (v < tensor_min) | (v > tensor_max)
        if need_fix.any().item() or need_clamp.any().item():
            layers_sanitized.append((k, need_fix.sum().item(), need_clamp.sum().item()))
        finite_vals = v[torch.isfinite(v)]
        repl = (finite_vals.max().item() if finite_vals.numel() else 1.0)
        v = torch.where(torch.isfinite(v), v, torch.full_like(v, repl))
        v = v.clamp(min=tensor_min, max=tensor_max)
        out[k] = v
    if layers_sanitized:
        sample = layers_sanitized[:3]
        logger.error(
            f"{log_prefix}: clamped/sanitized non-finite or out-of-range values when loading: "
            f"{len(layers_sanitized)} layers affected (e.g. {sample})."
        )
    return out


def compute_key_distortion_weights(
    model,
    query_capture_full=None,
):
    """
    Compute key distortion weights for bit allocation.

    For each K latent dim c, the weight is the query importance:
      w_k_c = (1/total_T) * Σ_{q_head} Σ_t (B_c^T q_{q_head,t})^2

    This is the per-token expected squared projection, summed over all GQA query heads.
    sigma2 is computed separately during the forward pass and saved independently.
    allocate_bits combines both: score_c = w_k_c * sigma2_c.

    Args:
        model: Model with HeadwiseLowRankModule k_proj layers.
        query_capture_full: Optional. Dict k_proj_layer_name -> list of (B, T, D) query outputs.
                            D = num_heads * head_dim.

    Returns:
        w_k_distortion_dict: keyed by cache_key (k_proj layer name).
    """
    from palu.model.modules.svd_linear import HeadwiseLowRankModule

    w_k_distortion_dict = {}
    eps = 1e-10
    num_attention_heads = getattr(model.config, "num_attention_heads", None)
    num_key_value_heads = getattr(model.config, "num_key_value_heads", num_attention_heads)

    for name, module in model.named_modules():
        if "k_proj" not in name or not isinstance(module, HeadwiseLowRankModule):
            continue
        cache_key = _cache_key_for_layer_name(name)
        r_total = sum(module.ranks)

        # q_c = sum_t (B_c^T q_t)^2: query projected into low-rank via B, then squared (both B and q included).
        # Query has shape (B,T,num_attention_heads,head_dim); key has num_key_value_heads groups (GQA).
        q_per_channel = None
        if query_capture_full and num_attention_heads is not None:
            q_list = query_capture_full.get(cache_key)
            if q_list is None:
                q_list = query_capture_full.get(name)
            if q_list:
                # Use float32 for Q to avoid overflow: (Q @ B_g).pow(2).sum(0) can overflow in float16
                Q = torch.cat([q.float().reshape(-1, q.shape[-1]) for q in q_list], dim=0)
                total_T, D = Q.shape
                head_dim = D // num_attention_heads
                if head_dim * num_attention_heads != D:
                    logger.warning(
                        f"Key calibration {name}: D={D} not divisible by num_attention_heads={num_attention_heads}, skipping query."
                    )
                else:
                    # Q: (total_T, num_attention_heads, head_dim)
                    Q = Q.reshape(total_T, num_attention_heads, head_dim)

                    # GQA layout:
                    # - num_key_value_heads KV heads
                    # - num_attention_heads query heads
                    # - q_per_k = num_attention_heads // num_key_value_heads queries per KV head
                    # - module.num_groups low-rank groups, each spanning multiple KV heads
                    if num_key_value_heads is None or num_key_value_heads == 0:
                        logger.warning(
                            f"Key calibration {name}: num_key_value_heads is None or 0, skipping query-based weighting."
                        )
                    elif num_attention_heads % num_key_value_heads != 0:
                        logger.warning(
                            f"Key calibration {name}: num_attention_heads={num_attention_heads} not divisible by "
                            f"num_key_value_heads={num_key_value_heads}, skipping query-based weighting."
                        )
                    else:
                        q_per_k = num_attention_heads // num_key_value_heads

                        num_groups = getattr(module, "num_groups", None)
                        if num_groups is None:
                            num_groups = len(module.U)

                        if not num_groups:
                            logger.warning(
                                f"Key calibration {name}: num_groups is 0 or None on HeadwiseLowRankModule, skipping query-based weighting."
                            )
                        elif num_key_value_heads % num_groups != 0:
                            logger.warning(
                                f"Key calibration {name}: num_key_value_heads={num_key_value_heads} not divisible by "
                                f"num_groups={num_groups}, skipping query-based weighting."
                            )
                        else:
                            # Each low-rank group spans n_kv_per_group KV heads, each of head_dim dims
                            n_kv_per_group = num_key_value_heads // num_groups
                            expected_group_dim = n_kv_per_group * head_dim

                            q_per_channel_list = []

                            # Precompute per-KV-head aggregated queries: mean over its q_per_k query heads
                            # Q_k_list[k]: (total_T, head_dim) for KV head k
                            # For each KV head, keep ALL q_per_k query heads separately
                            # (not averaged) so squared projections are summed over all
                            # query heads. Taking mean-then-square (Jensen) underestimates
                            # the distortion weight.
                            # Q_k_list[k]: (total_T * q_per_k, head_dim)
                            Q_k_list = []
                            for k in range(num_key_value_heads):
                                q_start = k * q_per_k
                                q_end = (k + 1) * q_per_k
                                if q_end > num_attention_heads:
                                    logger.warning(
                                        f"Key calibration {name}: KV head {k} has q_end={q_end} > num_attention_heads={num_attention_heads}, "
                                        "skipping query-based weighting."
                                    )
                                    Q_k_list = []
                                    break
                                # (total_T, q_per_k, head_dim) → (total_T * q_per_k, head_dim)
                                Q_k = Q[:, q_start:q_end, :].reshape(-1, Q.shape[-1])
                                Q_k_list.append(Q_k)

                            if len(Q_k_list) != num_key_value_heads:
                                # Something went wrong in KV/query partitioning; fall back
                                logger.warning(
                                    f"Key calibration {name}: could not build per-KV-head query aggregates; "
                                    "falling back to variance-only key weights."
                                )
                            else:
                                # Now for each low-rank group g, pack its KV heads' aggregated queries and apply B_g
                                for g, U_g in enumerate(module.U):
                                    B_g = U_g.weight  # (group_dim, r_g)
                                    B_g = (
                                        B_g.cpu().float()
                                        if B_g.device.type != "cpu"
                                        else B_g.float()
                                    )
                                    group_dim = B_g.shape[0]

                                    if group_dim != expected_group_dim:
                                        logger.warning(
                                            f"Key calibration {name} group {g}: group_dim={group_dim} "
                                            f"!= n_kv_per_group*head_dim={expected_group_dim}, skipping this group."
                                        )
                                        continue

                                    kv_start = g * n_kv_per_group
                                    kv_end = (g + 1) * n_kv_per_group
                                    if kv_end > num_key_value_heads:
                                        logger.warning(
                                            f"Key calibration {name} group {g}: kv_end={kv_end} exceeds "
                                            f"num_key_value_heads={num_key_value_heads}, skipping this group."
                                        )
                                        continue

                                    # Concatenate per-KV-head aggregated queries for this group:
                                    # shape (total_T, n_kv_per_group * head_dim) == (total_T, group_dim)
                                    Q_group = torch.cat(
                                        Q_k_list[kv_start:kv_end], dim=-1
                                    )

                                    if Q_group.shape[-1] != group_dim:
                                        logger.warning(
                                            f"Key calibration {name} group {g}: Q_group last dim {Q_group.shape[-1]} "
                                            f"!= B_g group_dim {group_dim}, skipping this group."
                                        )
                                        continue

                                    # q_c_g: (r_g,) query-aware weight for this low-rank group.
                                    # Q_group rows = total_T * q_per_k (all query heads flattened).
                                    # Divide by total_T → Σ_{q_heads} E_t[(B^T q_{q,t})^2]:
                                    # per-token expected squared projection, summed over all
                                    # q_per_k query heads attending this KV group.
                                    q_c_g = (Q_group @ B_g).pow(2).sum(dim=0) / max(total_T, 1)
                                    q_per_channel_list.append(q_c_g)

                                if q_per_channel_list:
                                    q_per_channel = torch.cat(q_per_channel_list)
                                    # Sanitize: float16 overflow in earlier code paths can leave inf/nan
                                    q_per_channel = torch.where(
                                        torch.isfinite(q_per_channel),
                                        q_per_channel,
                                        torch.zeros_like(q_per_channel),
                                    )
                                    q_per_channel = q_per_channel.clamp(
                                        min=eps, max=1e10
                                    )
                                else:
                                    logger.warning(
                                        f"Key calibration {name}: no valid groups for query-based weighting; "
                                        "falling back to variance-only key weights."
                                    )

        # w_k: pure query importance E_q[(B^T q)^2], NO sigma2.
        # allocate_bits computes score = w_k * sigma2 — sigma2 must not be included here.
        if q_per_channel is not None and q_per_channel.numel() == r_total:
            w_k = q_per_channel.float()
        else:
            # Fallback: no query info, uniform weight (sigma2 still applied by allocate_bits)
            w_k = torch.ones(r_total, dtype=torch.float32)

        w_k_finite = torch.isfinite(w_k)
        if not w_k_finite.all():
            logger.error(
                f"Key calibration {name}: w_k_distortion had {(~w_k_finite).sum().item()} inf/nan; "
                "replacing with 1.0 and clamping to [1e-10, 1e10]."
            )
        w_k = torch.where(w_k_finite, w_k, torch.ones_like(w_k))
        w_k = w_k.clamp(min=eps, max=1e10)

        w_k_distortion_dict[cache_key] = w_k.cpu()
        if query_capture_full is not None and cache_key in query_capture_full:
            del query_capture_full[cache_key]
    return w_k_distortion_dict




def save_rd_statistics(sigma2, w, save_path, w_o=None, w_k_distortion=None):
    """
    Save RD calibration statistics to a file.

    Args:
        sigma2: Dictionary of variance statistics
        w: Dictionary of gradient-based importance statistics
        save_path: Path to save the statistics file
        w_o: Optional dict of W_O weights (||fused_W_O[:, i]||² for each V latent dim)
        w_k_distortion: Optional dict layer_name -> tensor [r] key query-importance weights
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    sigma2_save = {_cache_key_for_layer_name(k): v.cpu() if v is not None else None for k, v in sigma2.items()}
    w_save = {_cache_key_for_layer_name(k): v.cpu() if v is not None else None for k, v in w.items()}
    stats = {
        "sigma2": sigma2_save,
        "w": w_save,
        "version": 6,
    }

    if w_o is not None:
        stats["w_o"] = {}
        for layer_name, layer_dict in w_o.items():
            ck = _cache_key_for_layer_name(layer_name)
            stats["w_o"][ck] = {}
            for head_idx, val in layer_dict.items():
                stats["w_o"][ck][head_idx] = val.cpu() if torch.is_tensor(val) else val

    if w_k_distortion is not None:
        stats["w_k_distortion"] = {_cache_key_for_layer_name(k): v.cpu() if torch.is_tensor(v) else v for k, v in w_k_distortion.items()}

    torch.save(stats, save_path)
    logger.info(f"RD statistics saved to {save_path}", fg="green")


def load_rd_statistics(load_path):
    """
    Load RD calibration statistics from a file.
    
    Args:
        load_path: Path to load the statistics file from
        
    Returns:
        Tuple of (sigma2, w) dictionaries
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Statistics file not found: {load_path}")
    
    stats = torch.load(load_path, map_location="cpu")
    logger.info(f"RD statistics loaded from {load_path}", fg="green")
    
    # Debug: print structure of loaded stats
    logger.debug(f"Stats keys: {list(stats.keys())}")
    if "sigma2" in stats:
        logger.debug(f"sigma2 type: {type(stats['sigma2'])}, num entries: {len(stats['sigma2']) if isinstance(stats['sigma2'], dict) else 'N/A'}")
        if isinstance(stats['sigma2'], dict) and len(stats['sigma2']) > 0:
            first_key = list(stats['sigma2'].keys())[0]
            first_val = stats['sigma2'][first_key]
            logger.debug(f"  First sigma2 entry: key={first_key}, type={type(first_val)}, shape={first_val.shape if torch.is_tensor(first_val) else 'N/A'}")
    if "w" in stats:
        logger.debug(f"w type: {type(stats['w'])}, num entries: {len(stats['w']) if isinstance(stats['w'], dict) else 'N/A'}")
        if isinstance(stats['w'], dict) and len(stats['w']) > 0:
            first_key = list(stats['w'].keys())[0]
            first_val = stats['w'][first_key]
            logger.debug(f"  First w entry: key={first_key}, type={type(first_val)}, shape={first_val.shape if torch.is_tensor(first_val) else 'N/A'}")
    
    def _normalize(stat_dict, fill_from=None, fill_value=None):
        """
        Older caches may store layer stats as a single Tensor per layer
        instead of a head-indexed dict. Normalize to dict format:
        layer -> {0: tensor_or_none}
        """
        if not isinstance(stat_dict, dict):
            raise ValueError(f"Expected dict for stats, got {type(stat_dict)}")
        
        normalized = {}
        for layer_name, val in stat_dict.items():
            if isinstance(val, dict):
                normalized[layer_name] = val
            elif torch.is_tensor(val) or val is None:
                normalized[layer_name] = {0: val}
            elif isinstance(val, (list, tuple)):
                # If a list/tuple of tensors, index by position
                normalized[layer_name] = {idx: v for idx, v in enumerate(val)}
            else:
                logger.warning(f"Unrecognized stats format for layer {layer_name}: {type(val)}; skipping")
        
        # Optionally fill missing entries from reference dict
        filled_count = 0
        filled_layer_samples = []
        if fill_from is not None:
            for layer_name, head_dict in fill_from.items():
                if layer_name not in normalized:
                    normalized[layer_name] = {}
                for head_idx, ref_val in head_dict.items():
                    if head_idx not in normalized[layer_name] or normalized[layer_name][head_idx] is None:
                        if fill_value == "ones" and torch.is_tensor(ref_val):
                            normalized[layer_name][head_idx] = torch.ones_like(ref_val)
                        elif fill_value == "zeros" and torch.is_tensor(ref_val):
                            normalized[layer_name][head_idx] = torch.zeros_like(ref_val)
                        else:
                            normalized[layer_name][head_idx] = ref_val
                        filled_count += 1
                        if len(filled_layer_samples) < 5:
                            filled_layer_samples.append(layer_name)
                        # Log only first few at DEBUG to avoid spam (full count in WARNING below)
                        if filled_count <= 3:
                            logger.debug(
                                f"[RD stats] Missing/None entry for layer={layer_name} head={head_idx} in stats; "
                                f"filling with {fill_value or 'reference'}."
                            )
        if filled_count > 0:
            logger.warning(
                f"[RD stats] {filled_count} layer/head entries were missing or None in w; "
                f"filled from sigma2. Re-run rd_calibration without --use_cache to regenerate w if needed."
            )
            logger.info(
                f"[RD stats] Sample missing layers: {filled_layer_samples}"
            )
        return normalized
    
    if "sigma2" not in stats:
        raise KeyError(f"Statistics file {load_path} missing 'sigma2' key. Keys found: {list(stats.keys())}")
    if "w" not in stats:
        raise KeyError(f"Statistics file {load_path} missing 'w' key. Keys found: {list(stats.keys())}")
    
    sigma2_raw = stats["sigma2"]
    w_raw = stats["w"]

    # Canonicalize keys so sigma2 and w use the same layer names (e.g. .inner. -> no .inner.)
    sigma2_raw = _canonicalize_stat_dict(sigma2_raw)
    w_raw = _canonicalize_stat_dict(w_raw)

    # Try to match w entries by layer suffix so different prefixes (e.g. base_model.model.... vs model....) still match
    suffix_to_w = {}
    for k, v in stats["w"].items():
        if v is None:
            continue
        if isinstance(v, dict) and not any(x is not None for x in v.values()):
            continue
        sk = _layer_suffix_for_match(k)
        if sk not in suffix_to_w or (suffix_to_w[sk] is None and v is not None):
            suffix_to_w[sk] = v
    def _w_entry_empty(key):
        val = w_raw.get(key)
        if val is None:
            return True
        if torch.is_tensor(val):
            return val.numel() == 0
        if isinstance(val, dict):
            return not any(x is not None for x in val.values())
        return False

    missing_before_suffix = [k for k in sigma2_raw if _w_entry_empty(k)]
    if missing_before_suffix:
        logger.debug(
            f"[RD stats] Sigma2 keys missing from w (before suffix match): {len(missing_before_suffix)} layers, "
            f"sample: {missing_before_suffix[:3]}; w file had keys (sample): {list(w_raw.keys())[:3]}"
        )

    filled_by_suffix = 0
    for sigma2_key in sigma2_raw:
        need_w = sigma2_key not in w_raw
        if not need_w and w_raw[sigma2_key] is not None:
            if torch.is_tensor(w_raw[sigma2_key]) and w_raw[sigma2_key].numel() > 0:
                continue
            if isinstance(w_raw[sigma2_key], dict) and any(x is not None for x in w_raw[sigma2_key].values()):
                continue
        sk = _layer_suffix_for_match(sigma2_key)
        if sk in suffix_to_w and suffix_to_w[sk] is not None:
            w_raw[sigma2_key] = suffix_to_w[sk]
            filled_by_suffix += 1
    if filled_by_suffix > 0:
        logger.info(f"[RD stats] Matched {filled_by_suffix} w entries by layer suffix (different key names in file)")

    sigma2 = _normalize(sigma2_raw)
    # Fill any remaining missing w with ones so bit allocation can proceed
    w = _normalize(w_raw, fill_from=sigma2, fill_value="ones")
    
    # Debug: print normalized structure
    logger.debug(f"After normalization: sigma2 has {len(sigma2)} layers, w has {len(w)} layers")
    if len(sigma2) > 0:
        first_layer = list(sigma2.keys())[0]
        logger.debug(f"  First layer {first_layer}: sigma2 has {len(sigma2[first_layer])} heads, w has {len(w.get(first_layer, {}))} heads")
    
    # Validate that we actually have some statistics
    has_sigma2 = any(
        torch.is_tensor(v) and v.numel() > 0
        for layer in sigma2.values()
        for v in layer.values()
    )
    has_w = any(
        torch.is_tensor(v) and v.numel() > 0
        for layer in w.values()
        for v in layer.values()
    )
    
    if not has_sigma2:
        # More detailed error message
        sigma2_details = []
        for layer_name, layer_dict in sigma2.items():
            for head_idx, val in layer_dict.items():
                if val is None:
                    sigma2_details.append(f"{layer_name}[{head_idx}]=None")
                elif torch.is_tensor(val):
                    sigma2_details.append(f"{layer_name}[{head_idx}]=tensor(shape={val.shape}, numel={val.numel()})")
                else:
                    sigma2_details.append(f"{layer_name}[{head_idx}]={type(val)}")
        raise RuntimeError(
            f"RD statistics in {load_path} have no valid sigma2 tensors. "
            f"Found {len(sigma2)} layers but all entries are None or empty. "
            f"Sample entries: {sigma2_details[:5]}. "
            "Re-run rd_calibration.py to regenerate stats."
        )
    
    if not has_w:
        # More detailed error message
        w_details = []
        for layer_name, layer_dict in w.items():
            for head_idx, val in layer_dict.items():
                if val is None:
                    w_details.append(f"{layer_name}[{head_idx}]=None")
                elif torch.is_tensor(val):
                    w_details.append(f"{layer_name}[{head_idx}]=tensor(shape={val.shape}, numel={val.numel()})")
                else:
                    w_details.append(f"{layer_name}[{head_idx}]={type(val)}")
        raise RuntimeError(
            f"RD statistics in {load_path} have no valid w tensors. "
            f"Found {len(w)} layers but all entries are None or empty. "
            f"Sample entries: {w_details[:5]}. "
            "Re-run rd_calibration.py to regenerate stats."
        )
    
    # Load w_o_dict if available (backward compatible).
    # w_o is keyed by self_attn (e.g. model.layers.0.self_attn), not k_proj/v_proj; do not fill from sigma2.
    w_o = None
    if "w_o" in stats:
        w_o_raw = _canonicalize_stat_dict(stats["w_o"])
        w_o = _normalize(w_o_raw, fill_from=None, fill_value=None)
        logger.info(f"Loaded w_o statistics for {len(w_o)} layers (self_attn keys)")
    else:
        logger.info("No w_o statistics found in file (backward compatibility: will use w_dict for values)")
    
    # Key distortion weights (backward compatible)
    w_k_distortion = None
    if "w_k_distortion" in stats:
        w_k_distortion = _canonicalize_stat_dict(stats["w_k_distortion"])
        w_k_distortion = _sanitize_finite_stat_dict(w_k_distortion, tensor_max=1e10, tensor_min=1e-10, log_prefix="w_k_distortion (loaded)")
        logger.info(f"Loaded w_k_distortion (key query-importance weights) for {len(w_k_distortion)} layers")

    return sigma2, w, w_o, w_k_distortion


def run_rd_calibration(model, tokenizer, device, nsamples=128, seqlen=1024,
                       cache_file=None, use_cache=True,
                       max_query_capture_batches=16, gradient_checkpointing=False,
                       dataset="wikitext2"):
    """
    Run RD calibration to collect variance and importance statistics.

    To avoid OOM: use low_memory (seqlen=1024, nsamples=128, max_query_capture_batches=8,
    gradient_checkpointing), or reduce these further.
    
    Args:
        model: The compressed model with HeadwiseLowRankModule layers
        tokenizer: Tokenizer for the model
        device: Device to run calibration on
        nsamples: Number of calibration samples
        seqlen: Sequence length for calibration
        cache_file: Optional path to cache file. If None, auto-generates based on model_id
        use_cache: Whether to use cached results if available
        max_query_capture_batches: Max number of batches to keep for key calibration (limits RAM).
        gradient_checkpointing: If True, enable to trade compute for lower GPU memory.
        
    Returns:
        Tuple of (sigma2, w) dictionaries containing statistics
    """
    model_id = model.config._name_or_path

    # Generate cache file path if not provided
    if cache_file is None:
        os.makedirs("cache", exist_ok=True)
        cache_file = f"cache/rd_stats_{model_id.replace('/', '_')}_{dataset}_{nsamples}_{seqlen}.pt"
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        logger.info(f"[RD Calibration] Cache file exists: {cache_file}", fg="yellow")
        logger.info(f"[RD Calibration] Loading statistics from cache...", fg="yellow")
        result = load_rd_statistics(cache_file)
        # load_rd_statistics returns (sigma2, w, w_o, sc, sc_k, w_k_distortion); 6 values
        return result
    
    logger.info(f"[RD Calibration] No cache file found. Running calibration...", fg="yellow")
    
    # Enable gradients for calibration (needed for w statistics)
    # We use eval mode for consistent statistics, but still need gradients to flow
    model.eval()
    model.to(device)
    # Disable KV cache so the quantized-cache detach path doesn't cut the gradient graph
    _use_cache_orig = model.config.use_cache
    model.config.use_cache = False
    
    # Ensure HeadwiseLowRankModule parameters have requires_grad=True for gradient flow
    # (We don't actually update them, but gradients need to flow through)
    for name, module in model.named_modules():
        if module.__class__.__name__ == "HeadwiseLowRankModule":
            # Enable gradients on VT layer so we can capture gradients
            for param in module.VT.parameters():
                param.requires_grad = True
            logger.debug(f"Enabled gradients for {name}.VT layer")

    calib_loader = get_calib_data(
        dataset,
        tokenizer,
        model_id,
        nsamples=nsamples,
        seqlen=seqlen
    )

    capture = RDCapture()

    # Query capture for key calibration: full q (B, T, D) so we can compute per-channel q_c = sum_t (B_c^T q_t)^2
    # Cap number of batches kept to avoid OOM (RAM grows with nsamples otherwise)
    query_capture_full = defaultdict(list)  # k_proj_layer_name -> list of (B, T, D) tensors

    def _make_query_hook(k_proj_name, max_batches):
        def hook(module, inputs, output):
            lst = query_capture_full[k_proj_name]
            if len(lst) < max_batches:
                lst.append(output.detach().cpu())
        return hook

    num_q_proj = 0
    for name, module in model.named_modules():
        if "q_proj" in name and "self_attn" in name and isinstance(module, nn.Linear):
            k_proj_name = name.replace("q_proj", "k_proj", 1)
            module.register_forward_hook(_make_query_hook(k_proj_name, max_query_capture_batches))
            num_q_proj += 1
    logger.info(f"Registered query (q_proj) hooks for key calibration: {num_q_proj} layers (max {max_query_capture_batches} batches stored)")

    # Register hooks and verify module structure
    num_registered = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == "HeadwiseLowRankModule":
            # Verify that the module has a VT attribute
            if not hasattr(module, 'VT'):
                logger.error(f"HeadwiseLowRankModule {name} does not have VT attribute! Available attributes: {dir(module)}")
                continue
            
            # Verify VT is a nn.Linear
            if not isinstance(module.VT, nn.Linear):
                logger.warning(f"HeadwiseLowRankModule {name}.VT is not a nn.Linear, got {type(module.VT)}")
            
            # Check VT layer dimensions
            logger.debug(f"Registering hooks for {name}: VT shape={module.VT.weight.shape}, "
                        f"out_features={module.VT.out_features}, in_features={module.VT.in_features}")
            
            # Diagnostic: check if VT layer has zero rows (which would cause zero output dimensions)
            if hasattr(module.VT, 'weight') and module.VT.weight is not None:
                VT_weight = module.VT.weight.data
                # Check for zero rows (output dimensions that are always zero)
                row_norms = VT_weight.norm(dim=1)
                num_zero_rows = (row_norms < 1e-8).sum().item()
                if num_zero_rows > 0:
                    logger.warning(f"  ⚠️  {name}.VT has {num_zero_rows}/{len(row_norms)} zero rows "
                                 f"(output dimensions that are always zero). This will cause zero sigma2.")
                # Check for zero columns (input dimensions that don't contribute)
                col_norms = VT_weight.norm(dim=0)
                num_zero_cols = (col_norms < 1e-8).sum().item()
                if num_zero_cols > 0:
                    logger.debug(f"  {name}.VT has {num_zero_cols}/{len(col_norms)} zero columns "
                               f"(input dimensions that don't contribute)")
            
            capture.register(module, name)
            num_registered += 1
    
    if num_registered == 0:
        raise RuntimeError("No HeadwiseLowRankModule instances found in model! Cannot run RD calibration.")
    
    logger.info(f"Registered hooks for {num_registered} HeadwiseLowRankModule instances")

    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (reduces GPU memory, increases compute)")
        else:
            logger.warning("gradient_checkpointing=True but model has no gradient_checkpointing_enable")

    for batch_idx, batch in enumerate(tqdm(calib_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        # HF causal LM returns loss only when labels are provided.
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"]

        loss = model(**batch).loss
        if loss is None:
            raise RuntimeError(
                "Model returned None loss during RD calibration. "
                "Ensure labels are provided and the model supports loss computation."
            )
        
        # Verify that loss requires grad (needed for backward pass)
        if not loss.requires_grad:
            logger.warning(
                "Loss does not require grad. This may prevent gradient capture for w statistics. "
                "Trying to enable gradients..."
            )
            # This might not work if the computation graph doesn't support it
            loss.requires_grad_(True)

        loss.backward()
        model.zero_grad()
        # Clean up stored Z references
        for name, module in model.named_modules():
            if module.__class__.__name__ == "HeadwiseLowRankModule":
                if hasattr(module, "_last_Z_list"):
                    delattr(module, "_last_Z_list")
        # Free batch tensors and periodically clear CUDA cache to reduce OOM risk
        del batch, loss
        if device != "cpu":
            torch.cuda.empty_cache()
            if (batch_idx + 1) % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = _use_cache_orig

    if device != "cpu":
        gc.collect()
        torch.cuda.empty_cache()

    # Verify that we collected statistics
    if len(capture.sigma2) == 0:
        raise RuntimeError("No sigma2 statistics collected! Hooks may not have been called.")
    if len(capture.w) == 0:
        raise RuntimeError("No w statistics collected! Backward hooks may not have been called.")
    
    # Diagnostic: check first few layers
    logger.info(f"Collected statistics for {len(capture.sigma2)} layers")
    for i, (name, sigma2_val) in enumerate(list(capture.sigma2.items())[:3]):
        if sigma2_val is not None and torch.is_tensor(sigma2_val):
            w_val = capture.w.get(name)
            logger.debug(f"Sample layer {name}: sigma2 shape={sigma2_val.shape}, "
                        f"non-zero={sigma2_val.gt(1e-10).sum().item()}/{sigma2_val.numel()}, "
                        f"max={sigma2_val.max().item():.2e}, mean={sigma2_val.mean().item():.2e}, "
                        f"count={capture.count[name]}")
            if w_val is not None and torch.is_tensor(w_val):
                logger.debug(f"  w: shape={w_val.shape}, non-zero={w_val.gt(1e-10).sum().item()}/{w_val.numel()}, "
                           f"max={w_val.max().item():.2e}, mean={w_val.mean().item():.2e}")

    # Normalize
    for name in capture.sigma2:
        capture.sigma2[name] /= capture.count[name]
        if capture.w[name] is not None:
            capture.w[name] /= capture.count[name]
    
    # Create hybrid importance measure: combine gradients with variance
    # This gives non-zero importance to dimensions with variance even if they have zero gradients
    # Formula: w_hybrid = w_gradients + alpha * sigma2_normalized
    # We normalize sigma2 relative to w to make them comparable
    hybrid_alpha = 0.1  # Weight for variance-based importance (10% of gradient importance)
    sigma2_scale = hybrid_alpha  # Default scale if normalization fails
    
    # First, compute global normalization factors for combining
    hybrid_applied = False
    all_w_values = []
    all_sigma2_values = []
    for name in capture.w:
        if capture.w[name] is not None and torch.is_tensor(capture.w[name]):
            w_tensor = capture.w[name]
            w_nonzero = w_tensor[w_tensor > 1e-10]
            if len(w_nonzero) > 0:
                all_w_values.append(w_nonzero)
        if capture.sigma2.get(name) is not None and torch.is_tensor(capture.sigma2[name]):
            sigma2_tensor = capture.sigma2[name]
            sigma2_nonzero = sigma2_tensor[sigma2_tensor > 1e-10]
            if len(sigma2_nonzero) > 0:
                all_sigma2_values.append(sigma2_nonzero)
    
    if len(all_w_values) > 0 and len(all_sigma2_values) > 0:
        # Compute percentile-based normalization
        all_w_flat = torch.cat(all_w_values)
        all_sigma2_flat = torch.cat(all_sigma2_values)
        w_percentile = torch.quantile(all_w_flat.float(), 0.95)
        sigma2_percentile = torch.quantile(all_sigma2_flat.float(), 0.95)
        
        # Normalize sigma2 to be comparable to w
        if sigma2_percentile > 1e-10:
            sigma2_scale = (w_percentile / sigma2_percentile).item() * hybrid_alpha
        else:
            sigma2_scale = hybrid_alpha
        
        logger.info(f"Creating hybrid importance: w_percentile={w_percentile.item():.2e}, "
                   f"sigma2_percentile={sigma2_percentile.item():.2e}, "
                   f"sigma2_scale={sigma2_scale:.2e}")
        
        # Apply hybrid importance to all layers
        for name in capture.w:
            if capture.w[name] is not None and torch.is_tensor(capture.w[name]):
                w_orig = capture.w[name]
                sigma2_val = capture.sigma2.get(name)
                
                if sigma2_val is not None and torch.is_tensor(sigma2_val):
                    # Create hybrid: w = w_gradients + alpha * normalized_sigma2
                    # Normalize sigma2 relative to w scale
                    sigma2_normalized = sigma2_val * sigma2_scale
                    w_hybrid = w_orig + sigma2_normalized
                    capture.w[name] = w_hybrid
                    
                    # Count how many zero-w dimensions got non-zero importance
                    zero_w_before = (w_orig <= 1e-10).sum().item()
                    zero_w_after = (w_hybrid <= 1e-10).sum().item()
                    if zero_w_before > zero_w_after:
                        logger.debug(f"Layer {name}: {zero_w_before - zero_w_after} dimensions "
                                   f"got non-zero importance from variance (was {zero_w_before}, now {zero_w_after})")
        hybrid_applied = True
    else:
        logger.warning("Could not create hybrid importance: insufficient statistics for normalization")
    
    # Summary: count zero w (after hybrid) and zero sigma2 dimensions across all layers
    total_dims = 0
    zero_w_dims = 0
    zero_sigma2_dims = 0
    for name in capture.w:
        if capture.w[name] is not None and torch.is_tensor(capture.w[name]):
            w_tensor = capture.w[name]
            total_dims += w_tensor.numel()
            zero_w_dims += (w_tensor <= 1e-10).sum().item()
    
    for name in capture.sigma2:
        if capture.sigma2[name] is not None and torch.is_tensor(capture.sigma2[name]):
            sigma2_tensor = capture.sigma2[name]
            zero_sigma2_dims += (sigma2_tensor <= 1e-10).sum().item()
    
    if total_dims > 0:
        zero_w_pct = (zero_w_dims / total_dims) * 100
        zero_sigma2_pct = (zero_sigma2_dims / total_dims) * 100
        if hybrid_applied:
            logger.info(f"Summary (after hybrid importance): {zero_w_dims}/{total_dims} dimensions ({zero_w_pct:.1f}%) have zero w.")
            logger.info(f"  Hybrid importance combines gradients (w) with variance (sigma2) using alpha={hybrid_alpha}, scale={sigma2_scale:.2e}.")
            logger.info(f"  Dimensions with variance but zero gradients now get non-zero importance.")
        else:
            logger.info(f"Summary: {zero_w_dims}/{total_dims} dimensions ({zero_w_pct:.1f}%) have zero w (no gradients).")
        logger.info(f"Summary: {zero_sigma2_dims}/{total_dims} dimensions ({zero_sigma2_pct:.1f}%) have zero sigma2 (constant/zero values).")
        logger.info(f"  Zero-sigma2 dimensions will get minimum bits (min_bits) since they have no variance.")
        if zero_w_pct > 50:
            logger.warning(
                f"More than 50% of dimensions still have zero w (even after hybrid). "
                f"This may indicate many dimensions have both zero gradients and zero variance."
            )
        if zero_sigma2_pct > 50:
            logger.warning(
                f"More than 50% of dimensions have zero sigma2. This may indicate: "
                f"(1) The compression created many unused/dead dimensions, "
                f"(2) The model has many dimensions that are always zero, "
                f"(3) The calibration data doesn't activate these dimensions, or "
                f"(4) There's an issue with the HeadwiseLowRankModule structure."
            )
    
    # Validate that w statistics were captured (not all zeros or all ones)
    w_valid = True
    w_issues = []
    for name in capture.w:
        if capture.w[name] is not None and torch.is_tensor(capture.w[name]):
            w_tensor = capture.w[name]
            w_max = w_tensor.max().item()
            w_min = w_tensor.min().item()
            w_mean = w_tensor.mean().item()
            w_std = w_tensor.std().item()
            
            # Check if all values are exactly zero (no gradients captured)
            if w_max < 1e-10:
                logger.warning(
                    f"Layer {name}: w statistics are all zero "
                    f"(max={w_max:.2e}). No gradients captured."
                )
                w_issues.append(f"{name}: all zeros")
                w_valid = False
            # Check if w is essentially constant (relative to magnitude)
            # Use coefficient of variation (std/mean) to check for meaningful variation
            elif w_mean > 1e-10:
                cv = w_std / w_mean  # Coefficient of variation
                if cv < 0.01:  # Less than 1% variation relative to mean
                    logger.warning(
                        f"Layer {name}: w statistics have very low variation "
                        f"(min={w_min:.2e}, max={w_max:.2e}, mean={w_mean:.2e}, std={w_std:.2e}, CV={cv:.2%}). "
                        "Gradients may not have been captured correctly."
                    )
                    w_issues.append(f"{name}: low variation (CV={cv:.2%})")
                    w_valid = False
                else:
                    logger.debug(
                        f"Layer {name}: w statistics look valid "
                        f"(min={w_min:.2e}, max={w_max:.2e}, mean={w_mean:.2e}, std={w_std:.2e}, CV={cv:.2%})"
                    )
            # Check if all values are ~1.0 (fallback values)
            elif abs(w_mean - 1.0) < 0.01 and w_std < 0.01:
                logger.warning(
                    f"Layer {name}: w statistics are all ~1.0 "
                    f"(mean={w_mean:.4f}, std={w_std:.4f}). "
                    "This suggests gradients were not captured - using fallback values."
                )
                w_issues.append(f"{name}: all ones (fallback)")
                w_valid = False
            else:
                logger.debug(
                    f"Layer {name}: w statistics look valid "
                    f"(min={w_min:.2e}, max={w_max:.2e}, mean={w_mean:.2e}, std={w_std:.2e})"
                )
    
    if not w_valid:
        logger.warning(
            f"Some w statistics appear invalid ({len(w_issues)} layers). "
            "Issues: " + ", ".join(w_issues[:5]) + ("..." if len(w_issues) > 5 else "") + ". "
            "However, small w values are normal for squared gradients. "
            "The bit allocation algorithm will normalize these values appropriately."
        )
    else:
        logger.info("All w statistics appear valid (non-zero with meaningful variation).")

    # Extract W_O weights for value layers
    logger.info("Extracting W_O weights from model...")
    w_o_dict = extract_w_o_weights(model)
    logger.info(f"Extracted W_O weights for {len(w_o_dict)} value layers")

    # Key calibration: q_c = sum_t (B^T q_t)_c^2 (query in low-rank), w_c = sigma^2_c * q_c (or sigma^2_c if no query).
    from bit_allocation import separate_kv_layers

    sigma2_k_dict, _, sigma2_v_dict, _, _ = separate_kv_layers(capture.sigma2, capture.w, w_o_dict)
    w_k_distortion_dict = {}
    if sigma2_k_dict:
        logger.info("Computing key distortion weights (query importance E_q[(B^T q)^2] per channel)...")
        w_k_distortion_dict = compute_key_distortion_weights(
            model, query_capture_full=query_capture_full if query_capture_full else None
        )
        logger.info(f"Key distortion weights computed for {len(w_k_distortion_dict)} layers")
        if query_capture_full:
            query_capture_full.clear()
        if device != "cpu":
            gc.collect()
            torch.cuda.empty_cache()

    if cache_file:
        save_rd_statistics(
            capture.sigma2,
            capture.w,
            cache_file,
            w_o=w_o_dict,
            w_k_distortion=w_k_distortion_dict if w_k_distortion_dict else None,
        )
        logger.info(f"RD statistics saved to {cache_file}")

    return capture.sigma2, capture.w, w_o_dict, None, None, w_k_distortion_dict
