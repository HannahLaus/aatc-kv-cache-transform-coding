"""
Custom DynamicCache that stores only compressed data and reconstructs on-the-fly.

This ensures we only have ONE cache (the compressed one) and nothing dense is stored.
Reconstruction happens only when attention needs it.
"""

from transformers.cache_utils import DynamicCache
from typing import Optional, Dict
import os
import torch
from loguru import logger

# Global flag to enable detailed cache debug logging in hot paths.
_ADAPTIVE_DEBUG = os.environ.get("PALU_ADAPTIVE_DEBUG", "0") == "1"

# One-time log for per-dim dequant success (layer 0 only)
_per_dim_logged: set = set()
# One-time log for per-dim dequant magnitude (layer 0 only, to compare with uniform)
_per_dim_magnitude_logged: set = set()
# One-time log for uniform dequant success (layer 0 only)
_uniform_logged: set = set()
# One-time log for uniform path BEFORE bitwidth update (seq_len == 32)
_uniform_logged_before_update: set = set()
# One-time log when wrapper reads bitwidths + cache.scales after adaptive (per-dim, layer 0, seq_len > 32)
_wrapper_logged_read_after_update: set = set()
# One-time log for uniform path AFTER bitwidth update has run (first read with seq_len > 32)
_uniform_logged_after_update: set = set()
# First "after update" log when seq_len exceeds this (33 = first read after "Updated bitwidths at seq_len=32")
_UNIFORM_LOG_AFTER_UPDATE_SEQ_LEN = 32
# One-time log for recent-window debug (key choice + scaling/bitwidth in truncated zone)
_recent_window_debug_logged: set = set()
# One-time log for dequant magnitude by zone (sink vs truncated vs recent) to trace garbled output
_zone_magnitude_debug_logged: set = set()
# One-time log: adaptive vs static-6 dequant reconstruction difference (reader path; runs when seq_len >= 32)
_adaptive_error_debug_logged: set = set()
# One-time trace: scale/bitwidth path for keys vs values (to debug key-specific issues)
_scale_bitwidth_trace_logged: set = set()
# Cache stores uniform-path scale (x = Q*scale). dequantize_with_per_dim_bitwidths expects alpha (stepsize = 2*alpha/2^b).
# For 8-bit: scale = 2*alpha/256 => alpha = scale * 2^7.
_UNIFORM_SCALE_TO_ALPHA = 2.0 ** 7  # 128.0


def _bitwidths_summary(bw: torch.Tensor) -> str:
    """Summary of bitwidths tensor for logging."""
    b = bw.detach().float().flatten()
    n = b.numel()
    f5 = b[:5].tolist() if n >= 5 else b.tolist()
    l5 = b[-5:].tolist() if n >= 5 else b.tolist()
    return (
        f"shape={tuple(bw.shape)} min={b.min().item():.1f} max={b.max().item():.1f} mean={b.mean().item():.2f} "
        f"first5={[round(x, 1) for x in f5]} last5={[round(x, 1) for x in l5]}"
    )


def _scales_dict_tail(
    scales_dict: Optional[Dict],
    batch_size: int,
    npos: int,
    seq_len: int,
    device: torch.device,
) -> Optional[Dict]:
    """Build scales_dict for positions [npos, seq_len) only, for use with dequantize_latent_from_integers(tail, ...)."""
    if not scales_dict or npos >= seq_len:
        return None
    expected_full = batch_size * seq_len
    expected_tail = batch_size * (seq_len - npos)
    out = {}
    for group_idx, (scales, base) in scales_dict.items():
        if scales is None or base is None or scales.numel() < expected_full or base.numel() < expected_full:
            return None
        flat_s = scales.detach().float().flatten()
        flat_b = base.detach().float().flatten()
        start = batch_size * npos
        end = batch_size * seq_len
        tail_s = flat_s[start:end].to(device).reshape(expected_tail, 1)
        tail_b = flat_b[start:end].to(device).reshape(expected_tail, 1)
        out[group_idx] = (tail_s, tail_b)
    return out if out else None


def _scale_summary(scales: torch.Tensor, base: torch.Tensor) -> str:
    """One-line summary of first group scales/base for before/after comparison."""
    s = scales.detach().float().flatten()
    b = base.detach().float().flatten()
    n = s.numel()
    s_f3 = s[:3].tolist() if n >= 3 else s.tolist()
    s_l3 = s[-3:].tolist() if n >= 3 else s.tolist()
    b_f3 = b[:3].tolist() if n >= 3 else b.tolist()
    b_l3 = b[-3:].tolist() if n >= 3 else b.tolist()
    return (
        f"scales min={s.min().item():.6f} max={s.max().item():.6f} mean={s.mean().item():.6f} "
        f"first3={[round(x, 5) for x in s_f3]} last3={[round(x, 5) for x in s_l3]} | "
        f"base min={b.min().item():.6f} max={b.max().item():.6f} mean={b.mean().item():.6f} "
        f"first3={[round(x, 5) for x in b_f3]} last3={[round(x, 5) for x in b_l3]}"
    )

# Try to import RoPE utility from transformers if available
try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    HAS_ROPE_UTILITY = True
except ImportError:
    try:
        from transformers.modeling_utils import apply_rotary_pos_emb
        HAS_ROPE_UTILITY = True
    except ImportError:
        HAS_ROPE_UTILITY = False


def _head_major_to_group_major_permutation(
    ranks, num_heads_per_group: int, num_dims: int, device: torch.device
) -> torch.Tensor:
    """
    Return permutation indices so that head_major[perm] gives group_major order.
    Module layout (group-major): group 0 [head0_dims][head1_dims]..., group 1 [head0_dims]...
    File/cache layout (head-major): head 0 dims, head 1 dims, ... (one block per head).
    Returns long tensor of shape (num_dims,) so that bitwidths_group_major = bitwidths_head_major[:, perm].
    """
    perm = []
    for g in range(len(ranks)):
        group_rank = ranks[g]
        head_rank = group_rank // num_heads_per_group
        for head_in_group in range(num_heads_per_group):
            head_idx = g * num_heads_per_group + head_in_group
            head_start_in_group = head_in_group * head_rank
            for d in range(head_rank):
                group_major_idx = sum(ranks[:g]) + head_start_in_group + d
                head_major_idx = head_idx * head_rank + d
                perm.append(head_major_idx)
    return torch.tensor(perm, device=device, dtype=torch.long)


def _cache_key_for_layer_name(layer_name: str) -> str:
    """
    QuantizedCache is keyed by names from before attention wrapping (e.g. model.layers.0.self_attn.k_proj).
    After PaluLlamaAttention wraps attention, named_modules() gives model.layers.0.self_attn.inner.k_proj.
    Normalize so we look up with the key the patched HeadwiseLowRankModule uses.
    """
    if layer_name is None:
        return layer_name
    if ".self_attn.inner." in layer_name:
        return layer_name.replace(".self_attn.inner.", ".self_attn.", 1)
    return layer_name


class ReconstructingDict:
    """
    Dict-like wrapper that reconstructs values on-the-fly from QuantizedCache.
    
    When attention accesses cache.key_cache[layer_idx], we reconstruct BA(x) from
    quantized A(x) stored in QuantizedCache.
    """
    
    def __init__(self, quantized_cache, model, layer_mapping, proj_type):
        """
        Args:
            quantized_cache: QuantizedCache instance
            model: Model reference for reconstruction
            layer_mapping: Dict mapping (layer_idx, proj_type) -> (layer_name, module)
            proj_type: 'k' for key or 'v' for value
        """
        self.quantized_cache = quantized_cache
        self.model = model
        self.layer_mapping = layer_mapping
        self.proj_type = proj_type
        # Store reconstructed values temporarily to avoid redundant reconstruction
        self._reconstructed_cache = {}
        # Cached attention modules; populated by QuantizedDynamicCache._build_layer_mapping().
        self._attention_modules = {}
        
        # Get num_key_value_heads from model config if available (for k_proj and v_proj)
        self.num_key_value_heads = None
        if model is not None and hasattr(model, 'config'):
            if hasattr(model.config, 'num_key_value_heads'):
                self.num_key_value_heads = model.config.num_key_value_heads
            elif hasattr(model.config, 'num_attention_heads'):
                # Fallback to num_attention_heads if num_key_value_heads not available
                self.num_key_value_heads = model.config.num_attention_heads
    
    def __getitem__(self, layer_idx):
        """Reconstruct BA(x) on-the-fly when accessed - returns PREVIOUS tokens only."""
        # Debug logging for layer 0 only to keep output manageable
        debug_layer = (layer_idx == 0)
        
        # Check if we have data for this layer in QuantizedCache
        key = (layer_idx, self.proj_type)
        if key not in self.layer_mapping:
            # Layer not in mapping - return None
            if debug_layer:
                logger.debug(f"[DEBUG Layer {layer_idx} {self.proj_type}] __getitem__: No layer mapping")
            return None
        
        layer_name, module = self.layer_mapping[key]
        cache_key = _cache_key_for_layer_name(layer_name)
        
        # Check if quantized data exists in QuantizedCache (use cache_key: cache is keyed by pre-wrap names)
        quantized_integers = self.quantized_cache.get_quantized(cache_key, self.proj_type)
        if quantized_integers is None:
            # Layer hasn't been processed yet - return None (transformers expects this)
            if debug_layer:
                logger.debug(f"[DEBUG Layer {layer_idx} {self.proj_type}] __getitem__: No quantized data yet")
            return None
        
        if debug_layer:
            quantized_shape = quantized_integers.shape
            logger.debug(f"[DEBUG Layer {layer_idx} {self.proj_type}] __getitem__: Accessing cache - quantized shape={quantized_shape}, seq_len={quantized_shape[1]}")
        
        # If the adaptive manager has just reconstructed BA(x) for this layer and stored it via __setitem__,
        # we can use that value ONCE, as long as its sequence length matches the number of previous tokens
        # (current cache length minus one new token). This uses the manager's reconstruction immediately
        # after truncation, then future reads go through QuantizedCache → _get_quantized_ba as usual.
        if layer_idx in self._reconstructed_cache:
            reconstructed_cached = self._reconstructed_cache.get(layer_idx)
            try:
                seq_len_cache = self.quantized_cache.get_seq_length(cache_key)
            except Exception:
                seq_len_cache = 0
            expected_prev_len = max(seq_len_cache - 1, 0)
            rec_prev_len = (
                reconstructed_cached.shape[2]
                if reconstructed_cached is not None and reconstructed_cached.ndim >= 3
                else None
            )
            if expected_prev_len > 0 and rec_prev_len == expected_prev_len:
                if debug_layer:
                    logger.debug(
                        f"[DEBUG Layer {layer_idx} {self.proj_type}] __getitem__: "
                        f"using stored reconstructed cache (seq_len={rec_prev_len}) and clearing it"
                    )
                # Use once, then clear so subsequent reads reconstruct from QuantizedCache
                del self._reconstructed_cache[layer_idx]
                return reconstructed_cached
        
        # CRITICAL: Transformers expects the cache to return only PREVIOUS tokens (excluding the new token).
        # The new token is provided separately via update() return value, and transformers concatenates them.
        #
        # Flow:
        # 1. HeadwiseLowRankModule.forward() updates QuantizedCache with full sequence (cached + new token)
        # 2. Transformers calls update() with new token's key/value states
        # 3. Transformers accesses cache via __getitem__() to get PREVIOUS tokens only
        # 4. Transformers concatenates previous tokens (from cache) + new token (from update() return)
        # 5. Transformers uses the concatenated full sequence for attention
        #
        # We always exclude the last token (the new token that was just added) and reconstruct
        # BA(x) fresh from QuantizedCache via _get_quantized_ba.
        reconstructed = self._get_quantized_ba(layer_idx, debug_layer=debug_layer, exclude_last_token=True)
        
        if reconstructed is not None:
            # Keys must have RoPE applied: the attention layer reads key_cache and concats with the new key
            # (which has RoPE applied there). So we must return cached keys WITH RoPE for positions 0..seq_len-1.
            if self.proj_type == 'k' and self.model is not None and reconstructed.dim() == 4:
                seq_len = reconstructed.shape[2]
                if seq_len > 0:
                    reconstructed = self._apply_rope_to_cached_keys(reconstructed, layer_idx, seq_len)
                    if reconstructed is None and debug_layer:
                        logger.warning(f"[DEBUG Layer {layer_idx} k] RoPE apply failed, returning keys without RoPE")
            return reconstructed
        else:
            if debug_layer:
                logger.debug(f"[DEBUG Layer {layer_idx} {self.proj_type}] __getitem__: Reconstruction returned None")
            return None
    
    def __setitem__(self, layer_idx, value):
        """Store reconstructed value (called by update())."""
        self._reconstructed_cache[layer_idx] = value
    
    def __contains__(self, layer_idx):
        """Check if we have data for this layer."""
        key = (layer_idx, self.proj_type)
        if key not in self.layer_mapping:
            result = False
        else:
            # Also check if quantized data actually exists (use cache key: cache is keyed by pre-wrap names)
            layer_name, _ = self.layer_mapping[key]
            cache_key = _cache_key_for_layer_name(layer_name)
            quantized_integers = self.quantized_cache.get_quantized(cache_key, self.proj_type)
            result = quantized_integers is not None
        
        return result

    def _apply_rope_to_cached_keys(self, key_states: torch.Tensor, layer_idx: int, seq_len: int) -> Optional[torch.Tensor]:
        """Apply RoPE to cached keys (positions 0..seq_len-1). Attention concats these with the new key; the new key gets RoPE in the attention layer, so cached keys must already have RoPE."""
        if not HAS_ROPE_UTILITY or self.model is None:
            return key_states
        # Use cached attention module to avoid repeated named_modules() traversal.
        attention_module = self._attention_modules.get(layer_idx)
        if attention_module is None:
            # Fallback: walk named_modules() once and populate cache for future calls.
            for prefix in ("model.layers.", "model.model.layers."):
                attn_name = f"{prefix}{layer_idx}.self_attn"
                for name, mod in self.model.named_modules():
                    if name == attn_name:
                        attn = mod.inner if hasattr(mod, "inner") and hasattr(mod.inner, "rotary_emb") else mod
                        self._attention_modules[layer_idx] = attn
                        attention_module = attn
                        break
                if attention_module is not None:
                    break
        if attention_module is None or not hasattr(attention_module, "rotary_emb"):
            return key_states
        device = key_states.device
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        try:
            cos, sin = attention_module.rotary_emb(key_states, position_ids=position_ids)
        except (TypeError, ValueError):
            cos, sin = attention_module.rotary_emb(key_states)
        key_states_rope, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin, position_ids)
        return key_states_rope
    
    def get(self, layer_idx, default=None):
        """Get with default value."""
        value = self[layer_idx]  # This will call __getitem__()
        return value if value is not None else default
    
    def keys(self):
        """Return layer indices that have data."""
        # Only return layers that actually have quantized data in QuantizedCache
        keys = []
        for (layer_idx, proj_type) in self.layer_mapping.keys():
            if proj_type == self.proj_type:
                layer_name, _ = self.layer_mapping[(layer_idx, proj_type)]
                cache_key = _cache_key_for_layer_name(layer_name)
                if self.quantized_cache.get_quantized(cache_key, self.proj_type) is not None:
                    keys.append(layer_idx)
        
        return keys
    
    def __len__(self):
        """Return number of layers with data."""
        length = len(self.keys())
        return length
    
    def __iter__(self):
        """Iterate over layer indices."""
        keys = self.keys()
        return iter(keys)
    
    def values(self):
        """Return reconstructed values for all layers."""
        return [self[layer_idx] for layer_idx in self.keys()]
    
    def items(self):
        """Return (layer_idx, value) pairs."""
        return [(layer_idx, self[layer_idx]) for layer_idx in self.keys()]
    
    def _get_quantized_ba(self, layer_idx: int, debug_layer: bool = False, exclude_last_token: bool = False) -> Optional[torch.Tensor]:
        """Get quantized BA(x) for a specific layer.
        
        Args:
            layer_idx: Layer index
            debug_layer: Whether to enable debug logging
            exclude_last_token: If True, return only previous tokens (excluding the last one).
                               If False, return the full sequence.
        """
        key = (layer_idx, self.proj_type)
        if key not in self.layer_mapping:
            return None
        
        layer_name, module = self.layer_mapping[key]
        cache_key = _cache_key_for_layer_name(layer_name)
        # Writer uses _captured_name from patch time (model.layers.N.self_attn.k_proj, no .inner.); reader layer_name
        # can be model.layers.N.self_attn.inner.k_proj. Try cache_key first to match writer, then layer_name.
        def _get_both(getter, proj):
            out = getter(cache_key, proj)
            if out is None and cache_key != layer_name:
                out = getter(layer_name, proj)
            return out
        quantized_integers = _get_both(self.quantized_cache.get_quantized, self.proj_type)
        scales_dict = _get_both(self.quantized_cache.get_scales_dict, self.proj_type)
        scales_tensor = _get_both(self.quantized_cache.get_scales, self.proj_type)
        bitwidths = _get_both(self.quantized_cache.get_bitwidths, self.proj_type)
        
        if quantized_integers is None:
            return None
        
        # Cache stores int8/int16; convert to float for dequantization
        if not quantized_integers.is_floating_point():
            quantized_integers = quantized_integers.float()
        # Avoid NaN propagating (e.g. from sliding-step); replace with 0 so dequant stays finite
        if not torch.isfinite(quantized_integers).all():
            quantized_integers = torch.where(
                torch.isfinite(quantized_integers), quantized_integers, torch.zeros_like(quantized_integers)
            )
        
        # Warn if we have quantized cache but no way to dequant (no scales_dict, no per-dim scales/bitwidths)
        # — otherwise we'd silently use integers as full-precision and get bad generation
        if (scales_dict is None or len(scales_dict) == 0) and (bitwidths is None or scales_tensor is None):
            logger.warning(
                f"[uniform Layer {layer_idx} {self.proj_type}] No scales_dict and no per-dim scales/bitwidths. "
                f"Quantized cache cannot be dequantized; using integers as float (wrong magnitude, bad output). "
                f"Check cache key: cache_key={cache_key!r}, layer_name={layer_name!r}"
            )
        
        # Determine quantization type: uniform (scales_dict), per-dim (scales tensor + bitwidths), or full-precision.
        # Prefer per-dim only when BOTH bitwidths and scales_tensor exist (e.g. after adaptive update). If we have
        # bitwidths but no cache.scales (e.g. layer only has scales_dict from uniform path), use uniform path instead.
        is_full_precision = True  # Default
        use_per_dim_dequant = False
        per_dim_degraded_no_scales = False
        if (
            bitwidths is not None
            and scales_tensor is not None
            and hasattr(module, 'dequantize_with_per_dim_bitwidths')
        ):
            scales_all_ones = torch.allclose(scales_tensor, torch.ones_like(scales_tensor), atol=1e-5)
            if scales_all_ones:
                logger.warning(
                    f"[per-dim Layer {layer_idx} {self.proj_type}] scales are all ones (not from quantization?). "
                    f"Dequant may be wrong magnitude."
                )
            use_per_dim_dequant = True
            is_full_precision = False
        elif scales_dict is not None and len(scales_dict) > 0:
            has_non_trivial_scales = False
            for group_idx, (scales, base) in scales_dict.items():
                if not torch.allclose(scales, torch.ones_like(scales), atol=1e-6):
                    has_non_trivial_scales = True
                    break
                if not torch.allclose(base, torch.zeros_like(base), atol=1e-6):
                    has_non_trivial_scales = True
                    break
            is_full_precision = not has_non_trivial_scales
        
        # Validate quantized_integers has correct shape
        # It should already be correct if we fixed low_rank_latents at the source
        expected_total_ranks = sum(module.ranks)
        if quantized_integers.shape[2] != expected_total_ranks:
            if quantized_integers.shape[2] > expected_total_ranks:
                # Slice to expected size (this should only happen for old cached data)
                logger.warning(
                    f"[FIX Layer {layer_idx} {self.proj_type}] quantized_integers has wrong shape. "
                    f"Slicing from {quantized_integers.shape[2]} to {expected_total_ranks} dimensions "
                    f"(module.ranks={module.ranks}, sum={expected_total_ranks}). "
                    f"This suggests the cache was created before the fix."
                )
                quantized_integers = quantized_integers[:, :, :expected_total_ranks].contiguous()
                # Verify slicing worked
                if quantized_integers.shape[2] != expected_total_ranks:
                    logger.error(
                        f"[ERROR Layer {layer_idx} {self.proj_type}] Slicing failed: "
                        f"quantized_integers.shape[2]={quantized_integers.shape[2]}, expected={expected_total_ranks}"
                    )
                    return None
            else:
                logger.error(
                    f"[ERROR Layer {layer_idx} {self.proj_type}] quantized_integers has fewer dimensions "
                    f"({quantized_integers.shape[2]}) than expected ({expected_total_ranks}). "
                    f"Cannot fix by slicing. Clearing invalid cache entry."
                )
                # Clear the invalid cache entry so it can be regenerated with correct dimensions
                # This happens when cache was created before the fix (all layers shared same VT)
                if cache_key in self.quantized_cache.quantized_latents:
                    if self.proj_type in self.quantized_cache.quantized_latents[cache_key]:
                        del self.quantized_cache.quantized_latents[cache_key][self.proj_type]
                        logger.info(f"  Cleared invalid cache entry for {cache_key}[{self.proj_type}]")
                if cache_key in self.quantized_cache.scales_dict:
                    if self.proj_type in self.quantized_cache.scales_dict[cache_key]:
                        del self.quantized_cache.scales_dict[cache_key][self.proj_type]
                if cache_key in self.quantized_cache.bitwidths:
                    if self.proj_type in self.quantized_cache.bitwidths[cache_key]:
                        del self.quantized_cache.bitwidths[cache_key][self.proj_type]
                return None
        
        # Dequantize and reconstruct BA(x) for the FULL sequence
        try:
            if per_dim_degraded_no_scales:
                # Have bitwidths but no scales (e.g. wrong cache key) — cannot dequant correctly
                dequantized_latents = quantized_integers.float()
            elif is_full_precision:
                if layer_idx == 0:
                    log_key = (0, self.proj_type, "fullprec")
                    if log_key not in _uniform_logged:  # reuse set for "already logged path"
                        _uniform_logged.add(log_key)
                        logger.info(
                            f"[CACHE PATH Layer 0 {self.proj_type}] using full-precision branch (quantized_integers as-is). "
                            f"scales_dict present={scales_dict is not None and len(scales_dict or {}) > 0}, "
                            f"has_non_trivial_scales=False?"
                        )
                dequantized_latents = quantized_integers
            elif use_per_dim_dequant:
                # Per-dim (adaptive) path: use scales and bitwidths from when each token was quantized
                if _ADAPTIVE_DEBUG and layer_idx == 0:
                    log_key = (0, self.proj_type, "perdim")
                    if log_key not in _uniform_logged:
                        _uniform_logged.add(log_key)
                        logger.info(
                            f"[CACHE PATH Layer 0 {self.proj_type}] using PER-DIM dequant (scales_tensor + bitwidths)."
                        )
                batch_size, seq_len, num_dims = quantized_integers.shape
                scales_for_dequant = scales_tensor.to(quantized_integers.device)
                bitwidths_for_dequant = bitwidths.to(quantized_integers.device)
                # Attention sink only: force first N tokens to 16 bits at read time (sink is never requantized).
                # Do NOT overlay the recent window: after a sliding step the older half is requantized to base
                # and update_bitwidths() stores that; we must use those stored bitwidths for dequant. Overlaying
                # last n_recent_full with 16 would wrongly dequant base-bitwidth positions with 16 → corrupt output.
                sink = getattr(module, "attention_sink_tokens", 0)
                if sink > 0 and bitwidths_for_dequant.dim() == 2 and bitwidths_for_dequant.shape[0] > 0:
                    bitwidths_for_dequant = bitwidths_for_dequant.clone()
                    n_sink = min(sink, bitwidths_for_dequant.shape[0])
                    bitwidths_for_dequant[:n_sink, :] = 16
                # Debug: log a few per-token bitwidth rows to compare with formula table (layer 0 only)
                if _ADAPTIVE_DEBUG and layer_idx == 0 and bitwidths_for_dequant.dim() == 2 and seq_len > 0:
                    token_indices = [0, seq_len // 2, seq_len - 1]
                    token_indices = sorted(set(i for i in token_indices if 0 <= i < seq_len))
                    for ti in token_indices:
                        row = bitwidths_for_dequant[ti]
                        b_min = row.min().item()
                        b_max = row.max().item()
                        b_mean = row.float().mean().item()
                        first5 = row[:5].tolist() if row.numel() >= 5 else row.tolist()
                        logger.info(
                            f"[CACHE DEBUG] Layer 0 {self.proj_type} bitwidths token {ti}: "
                            f"min={b_min:.1f} max={b_max:.1f} mean={b_mean:.2f} "
                            f"first5={[int(x) for x in first5]}"
                        )
                # Number of positions covered by cache (adaptive length)
                if scales_for_dequant.dim() == 3:
                    npos_cache = scales_for_dequant.shape[1]
                elif scales_for_dequant.dim() == 2:
                    npos_cache = (
                        scales_for_dequant.shape[1]
                        if scales_for_dequant.shape[0] == batch_size
                        else scales_for_dequant.shape[0]
                    )
                else:
                    npos_cache = 0
                # Log what we read from cache (bitwidths + cache.scales) first time after adaptive (seq_len > 32)
                if _ADAPTIVE_DEBUG and layer_idx == 0 and seq_len > _UNIFORM_LOG_AFTER_UPDATE_SEQ_LEN:
                    read_key = (0, self.proj_type, "read_after")
                    if read_key not in _wrapper_logged_read_after_update:
                        _wrapper_logged_read_after_update.add(read_key)
                        logger.info(
                            f"[CACHE READ AFTER UPDATE Layer 0 {self.proj_type}] bitwidths from cache: {_bitwidths_summary(bitwidths_for_dequant)}"
                        )
                        s_flat = scales_for_dequant.detach().float().flatten()
                        n = s_flat.numel()
                        logger.info(
                            f"[CACHE READ AFTER UPDATE Layer 0 {self.proj_type}] cache.scales from cache: "
                            f"shape={tuple(scales_for_dequant.shape)} min={s_flat.min().item():.6f} max={s_flat.max().item():.6f} "
                            f"mean={s_flat.mean().item():.6f} first3={[round(s_flat[i].item(), 5) for i in range(min(3, n))]} "
                            f"last3={[round(s_flat[n-1-i].item(), 5) for i in range(min(3, n))][::-1]}"
                        )
                # Slice/pad to seq_len (source of truth from quantized_integers) so scales and bitwidths stay in sync
                if scales_for_dequant.dim() == 3:
                    npos = scales_for_dequant.shape[1]
                    ng = scales_for_dequant.shape[2]
                    if npos >= seq_len:
                        scales_slice = scales_for_dequant[:, :seq_len, :].contiguous()
                    elif npos > 0:
                        # Fewer scale rows than tokens: pad by repeating last row (wrong scale for padded positions)
                        if _ADAPTIVE_DEBUG and layer_idx == 0 and (0, self.proj_type, "scale_pad") not in _per_dim_logged:
                            _per_dim_logged.add((0, self.proj_type, "scale_pad"))
                            logger.warning(
                                f"[per-dim Layer 0 {self.proj_type}] scales per-token mismatch: cache has {npos} rows, "
                                f"need {seq_len}. Padding with last row → tokens [{npos}:{seq_len}] have WRONG scale (magnitude may be off)."
                            )
                        last_pos = scales_for_dequant[:, -1:, :].expand(batch_size, seq_len - npos, ng)
                        scales_slice = torch.cat([scales_for_dequant[:, :npos, :], last_pos], dim=1).contiguous()
                    else:
                        scales_slice = None
                    # If cache has per-group scales (ng) but module has different num_groups, expand scales
                    # to (batch, seq, num_dims) only when the module uses channelwise scaling (accepts per-dim scales).
                    # Tokenwise modules only accept (batch, seq, num_heads) or (batch, seq, num_groups), so we must not expand.
                    mod_ng = getattr(module, 'num_groups', None)
                    is_channelwise = getattr(module, 'scaling_type', None) == 'channelwise'
                    if (
                        scales_slice is not None
                        and is_channelwise
                        and mod_ng is not None
                        and ng != mod_ng
                        and num_dims % ng == 0
                        and ng != num_dims
                    ):
                        rep = num_dims // ng
                        scales_slice = scales_slice.unsqueeze(-1).expand(
                            batch_size, seq_len, ng, rep
                        ).reshape(batch_size, seq_len, num_dims).contiguous()
                elif scales_for_dequant.dim() == 2 and scales_for_dequant.shape[0] == batch_size:
                    npos = scales_for_dequant.shape[1]
                    ng = getattr(module, 'num_groups', 1)
                    if npos >= seq_len:
                        scales_slice = scales_for_dequant[:, :seq_len].contiguous()
                    elif npos > 0:
                        last_pos = scales_for_dequant[:, -1:].expand(batch_size, seq_len - npos)
                        scales_slice = torch.cat([scales_for_dequant[:, :npos], last_pos], dim=1).contiguous()
                    else:
                        scales_slice = None
                elif scales_for_dequant.dim() == 2 and scales_for_dequant.shape[0] == seq_len:
                    # Cache stores (seq_len, num_groups) e.g. (33, 1); add batch dim -> (1, seq_len, num_groups)
                    npos = scales_for_dequant.shape[0]
                    ng = scales_for_dequant.shape[1]
                    if getattr(module, 'num_groups', ng) != ng:
                        scales_slice = None
                    elif npos >= seq_len:
                        scales_slice = scales_for_dequant[:seq_len, :].unsqueeze(0).contiguous()
                    elif npos > 0:
                        last_row = scales_for_dequant[-1:].expand(seq_len - npos, ng)
                        scales_slice = torch.cat([scales_for_dequant[:npos], last_row], dim=0).unsqueeze(0).contiguous()
                    else:
                        scales_slice = None
                else:
                    scales_slice = None
                # bitwidths must be (seq_len, num_dims) in group-major order (module layout).
                if bitwidths_for_dequant.dim() == 1:
                    bitwidths_for_dequant = bitwidths_for_dequant.unsqueeze(0).expand(seq_len, num_dims)
                # If the bitwidth file is in head-major order (per dim in each head), reorder to group-major.
                # Default OFF: many files are group-major; set PALU_HEAD_MAJOR_BITWIDTHS=1 to enable.
                _head_major_bitwidths = os.environ.get("PALU_HEAD_MAJOR_BITWIDTHS", "0") == "1"
                ranks = getattr(module, "ranks", None)
                num_heads_per_group = getattr(module, "num_heads_per_group", None)
                num_heads = getattr(module, "num_heads", None)
                if (
                    _head_major_bitwidths
                    and bitwidths_for_dequant.shape[1] == num_dims
                    and ranks is not None
                    and num_heads_per_group is not None
                    and num_heads is not None
                    and sum(ranks) == num_dims
                ):
                    perm = _head_major_to_group_major_permutation(
                        ranks, num_heads_per_group, num_dims, bitwidths_for_dequant.device
                    )
                    bitwidths_for_dequant = bitwidths_for_dequant[:, perm].contiguous()
                    if _ADAPTIVE_DEBUG and layer_idx == 0 and (0, self.proj_type, "bw_head2group") not in _per_dim_logged:
                        _per_dim_logged.add((0, self.proj_type, "bw_head2group"))
                        logger.info(
                            f"[per-dim Layer 0 {self.proj_type}] reordered bitwidths head-major -> group-major "
                            f"(num_heads={num_heads}, num_heads_per_group={num_heads_per_group})"
                        )
                # If cache has per-group bitwidths (seq_len, num_groups) and num_dims = sum(ranks), expand to (seq_len, num_dims).
                # This path is used for both keys and values: each uses its own module.ranks and num_dims (k_proj vs v_proj).
                # Module layout: group g occupies dimensions [sum(ranks[:g]), sum(ranks[:g+1])), so we repeat
                # group g's bitwidth ranks[g] times.
                bw_dim1 = bitwidths_for_dequant.shape[1]
                if bw_dim1 < num_dims:
                    ranks = getattr(module, "ranks", None)
                    if ranks is not None and len(ranks) == bw_dim1 and sum(ranks) == num_dims:
                        # Per-group expansion using actual ranks (matches quantizer group layout)
                        seq_len_bw = bitwidths_for_dequant.shape[0]
                        parts = []
                        for g, r in enumerate(ranks):
                            parts.append(
                                bitwidths_for_dequant[:, g : g + 1].expand(seq_len_bw, r)
                            )
                        bitwidths_for_dequant = torch.cat(parts, dim=1).contiguous()
                        if _ADAPTIVE_DEBUG and layer_idx == 0 and (0, self.proj_type, "bw_expand") not in _per_dim_logged:
                            _per_dim_logged.add((0, self.proj_type, "bw_expand"))
                            logger.info(
                                f"[per-dim Layer 0 {self.proj_type}] expanded per-group bitwidths "
                                f"({bw_dim1} groups, ranks={ranks[:4]}...) -> ({seq_len_bw}, {num_dims})"
                            )
                    elif num_dims % bw_dim1 == 0:
                        # Fallback: equal groups
                        rep = num_dims // bw_dim1
                        bitwidths_for_dequant = bitwidths_for_dequant.unsqueeze(2).expand(
                            bitwidths_for_dequant.shape[0], bw_dim1, rep
                        ).reshape(bitwidths_for_dequant.shape[0], num_dims).contiguous()
                # Log once when cache already had full num_dims (no expansion) so we know which path was used
                if _ADAPTIVE_DEBUG and (
                    bitwidths_for_dequant.shape[1] == num_dims
                    and layer_idx == 0
                    and (0, self.proj_type, "bw_full") not in _per_dim_logged
                ):
                    _per_dim_logged.add((0, self.proj_type, "bw_full"))
                    logger.info(
                        f"[per-dim Layer 0 {self.proj_type}] bitwidths already (seq, num_dims)=({bitwidths_for_dequant.shape[0]}, {num_dims}), using as-is (no expansion)"
                    )
                if bitwidths_for_dequant.shape[1] < num_dims:
                    bitwidths_slice = None
                elif bitwidths_for_dequant.shape[0] >= seq_len:
                    bitwidths_slice = bitwidths_for_dequant[:seq_len, :num_dims].contiguous()
                elif bitwidths_for_dequant.shape[0] > 0:
                    # New tokens not yet adapted: pad with uniform bitwidth (8) so dequant uses correct scale for them
                    n_adapt = bitwidths_for_dequant.shape[0]
                    uniform_bw = int(getattr(module, 'bitwidth', 8)) if hasattr(module, 'bitwidth') else 8
                    pad_bw = torch.full(
                        (seq_len - n_adapt, num_dims),
                        float(uniform_bw),
                        device=bitwidths_for_dequant.device,
                        dtype=bitwidths_for_dequant.dtype,
                    )
                    bitwidths_slice = torch.cat(
                        [bitwidths_for_dequant[:n_adapt, :num_dims].contiguous(), pad_bw], dim=0
                    ).contiguous()
                    if _ADAPTIVE_DEBUG:
                        logger.debug(
                            f"[per-dim Layer {layer_idx} {self.proj_type}] bitwidths length ({n_adapt}) < seq_len ({seq_len}). Padded with {uniform_bw}-bit for new positions."
                        )
                else:
                    bitwidths_slice = None
                # Debug: key choice and scaling/bitwidth in recent-window (truncated) zone (layer 0, once per proj_type)
                recent_tokens = self.quantized_cache.get_recent_tokens()
                if _ADAPTIVE_DEBUG and (
                    layer_idx == 0
                    and recent_tokens > 0
                    and seq_len > (sink + recent_tokens)
                    and (layer_idx, self.proj_type) not in _recent_window_debug_logged
                    and scales_slice is not None
                    and bitwidths_slice is not None
                    and bitwidths_slice.shape[0] >= seq_len
                ):
                    _recent_window_debug_logged.add((layer_idx, self.proj_type))
                    key_used_quant = (
                        cache_key
                        if self.quantized_cache.get_quantized(cache_key, self.proj_type) is not None
                        else layer_name
                    )
                    key_used_scales = (
                        cache_key
                        if self.quantized_cache.get_scales(cache_key, self.proj_type) is not None
                        else layer_name
                    )
                    key_used_bw = (
                        cache_key
                        if self.quantized_cache.get_bitwidths(cache_key, self.proj_type) is not None
                        else layer_name
                    )
                    logger.info(
                        f"[RECENT_WINDOW_DEBUG] Layer 0 {self.proj_type} key_choice: "
                        f"quantized={key_used_quant!r}, scales={key_used_scales!r}, bitwidths={key_used_bw!r} "
                        f"(cache_key={cache_key!r}, layer_name={layer_name!r})"
                    )
                    # Use actual n_recent_full from cache: last n_recent_full tokens are 16-bit;
                    # truncated zone is the rest of the recent window [start, seq_len - n_recent_full).
                    n_recent_full = self.quantized_cache.get_n_recent_full()
                    start = seq_len - recent_tokens
                    end_trunc = seq_len - n_recent_full  # first index that is recent_16bit
                    # truncated_zone = [start, end_trunc), recent_16bit_zone = [end_trunc, seq_len)
                    token_indices_to_log = []
                    if start >= 0 and start < seq_len and end_trunc > start:
                        token_indices_to_log.append((start, "truncated"))
                    if end_trunc > start + 1 and end_trunc - 1 < seq_len:
                        token_indices_to_log.append((end_trunc - 1, "truncated"))
                    if end_trunc < seq_len:
                        token_indices_to_log.append((end_trunc, "recent_16bit"))
                    for token_idx, zone in token_indices_to_log:
                        row = bitwidths_slice[token_idx]
                        b_min = row.min().item()
                        b_max = row.max().item()
                        b_mean = row.float().mean().item()
                        if scales_slice.dim() == 3:
                            scale_val = scales_slice[0, token_idx, :].float().mean().item()
                        else:
                            scale_val = scales_slice[0, token_idx].float().item()
                        q_sample = quantized_integers[0, token_idx, :3].tolist()
                        logger.info(
                            f"[RECENT_WINDOW_DEBUG] Layer 0 {self.proj_type} token {token_idx} ({zone}): "
                            f"bitwidth min={b_min:.1f} max={b_max:.1f} mean={b_mean:.2f}, "
                            f"scale(alpha)={scale_val:.6f}, Q[:3]={[round(x, 2) for x in q_sample]}"
                        )
                    logger.info(
                        f"[RECENT_WINDOW_DEBUG] Layer 0 {self.proj_type} recent window: seq_len={seq_len} "
                        f"recent_tokens={recent_tokens} n_recent_full={n_recent_full} sink={sink} "
                        f"truncated_zone=[{start},{end_trunc}) recent_16bit_zone=[{end_trunc},{seq_len})"
                    )
                # When cache covers only part of the sequence, use hybrid: head = per-dim (cache), tail = uniform (scales_dict)
                dequantized_latents = None
                if (
                    npos_cache > 0
                    and npos_cache < seq_len
                    and scales_dict
                    and bitwidths_for_dequant.shape[0] >= npos_cache
                ):
                    # Build head scales and bitwidths (first npos_cache from cache)
                    if scales_for_dequant.dim() == 3:
                        scales_head = scales_for_dequant[:, :npos_cache, :].contiguous()
                    elif scales_for_dequant.dim() == 2 and scales_for_dequant.shape[0] == batch_size:
                        scales_head = scales_for_dequant[:, :npos_cache].contiguous()
                    else:
                        scales_head = scales_for_dequant[:npos_cache, :].unsqueeze(0).contiguous()
                    # When use_per_dim_dequant is True, cache was written by per-dim path: scales are already alpha.
                    # Do NOT multiply by _UNIFORM_SCALE_TO_ALPHA (that is for uniform-path scale only).
                    scales_head_alpha = scales_head.float().to(quantized_integers.device)
                    bitwidths_head = bitwidths_for_dequant[:npos_cache, :num_dims].contiguous()
                    quant_head = quantized_integers[:, :npos_cache, :]
                    # Only use full uniform dequant for tail when scales_dict has full sequence length (sync with quantized_integers)
                    expected_elems = batch_size * seq_len
                    first_group = next(iter(scales_dict.values()), None) if scales_dict else None
                    scales_dict_len_ok = (
                        first_group is not None
                        and first_group[0] is not None
                        and first_group[0].numel() == expected_elems
                    )
                    if not scales_dict_len_ok and first_group is not None and first_group[0] is not None:
                        logger.warning(
                            f"[per-dim Layer {layer_idx} {self.proj_type}] scales_dict length mismatch: "
                            f"numel={first_group[0].numel()}, expected batch*seq_len={expected_elems}. Skipping hybrid."
                        )
                    try:
                        dequant_head = module.dequantize_with_per_dim_bitwidths(
                            quant_head,
                            scales_head_alpha,
                            target_dtype=None,
                            bitwidths_per_token=bitwidths_head,
                        )
                        if not scales_dict_len_ok:
                            raise ValueError("scales_dict length mismatch")
                        # Tail = full uniform dequant then slice: guarantees same formula/layout as uniform path
                        dequantized_full_uniform = module.dequantize_latent_from_integers(
                            quantized_integers, scales_dict
                        )
                        dequant_tail = dequantized_full_uniform[:, npos_cache:, :]
                        dequantized_latents = torch.cat([dequant_head, dequant_tail], dim=1)
                        if _ADAPTIVE_DEBUG and layer_idx == 0 and (0, self.proj_type, "hybrid") not in _uniform_logged:
                            _uniform_logged.add((0, self.proj_type, "hybrid"))
                            logger.debug(
                                f"[per-dim Layer 0 {self.proj_type}] hybrid dequant: head [0:{npos_cache}] per-dim, "
                                f"tail [{npos_cache}:{seq_len}] from full uniform dequant slice"
                            )
                    except (ValueError, RuntimeError) as e:
                        logger.debug(
                            f"[per-dim Layer {layer_idx} {self.proj_type}] hybrid dequant failed ({e}), using padded single dequant"
                        )
                        dequantized_latents = None
                if dequantized_latents is None:
                    if scales_slice is not None and bitwidths_slice is not None and bitwidths_slice.shape == (seq_len, num_dims):
                        try:
                            # Cache scales = tokenwise alpha from quantize time. In quantize_with_per_dim_bitwidths
                            # (tokenwise) we compute alpha per token per group (e.g. max(|latent|)), use
                            # stepsize = 2*alpha/2^b, quantize, and write that alpha to the cache. So at read
                            # time we use the same stored alpha for dequant (stepsize = 2*alpha/2^b). Do NOT
                            # multiply by _UNIFORM_SCALE_TO_ALPHA — that is for uniform-path scales only.
                            # Keep scales in float32 for the clamp — 1e-8 underflows in float16.
                            # We'll cast the dequant *output* to the module's compute dtype instead.
                            scales_for_dequant = scales_slice.float().to(quantized_integers.device)
                            # Avoid NaN/Inf/zero so dequant never blows up (defensive after sliding-step requant)
                            scales_for_dequant = torch.where(
                                torch.isfinite(scales_for_dequant),
                                scales_for_dequant.clamp(min=1e-8, max=1e4),
                                torch.full_like(scales_for_dequant, 1e-6),
                            )
                            # Determine the module's compute dtype (B weight dtype) so the dequant
                            # output is already in the right dtype before the B·z matmul, avoiding
                            # an implicit float32→float16 downcast in the attention path.
                            _B = getattr(module, 'B', None)
                            _compute_dtype = _B.weight.dtype if (_B is not None and hasattr(_B, 'weight')) else None
                            # One-time log: verify scales at read match what writer stored (quantize/dequant are inverses)
                            if _ADAPTIVE_DEBUG and layer_idx == 0 and (0, self.proj_type, "scales_check") not in _per_dim_logged:
                                _per_dim_logged.add((0, self.proj_type, "scales_check"))
                                s = scales_slice.float()
                                logger.info(
                                    f"[per-dim READ Layer 0 {self.proj_type}] scales (alpha) from cache shape={tuple(scales_slice.shape)} "
                                    f"min={s.min().item():.6f} max={s.max().item():.6f} mean={s.mean().item():.6f}"
                                )
                            # One-time trace: scale/bitwidth path for k vs v (debug key-specific diff_adapt_vs_6)
                            if layer_idx == 0 and seq_len >= 32 and (0, self.proj_type, "scale_bw_trace") not in _scale_bitwidth_trace_logged:
                                _scale_bitwidth_trace_logged.add((0, self.proj_type, "scale_bw_trace"))
                                scaling_type = getattr(module, "scaling_type", None)
                                num_heads = getattr(module, "num_heads", None)
                                num_groups = getattr(module, "num_groups", None)
                                tok = min(16, seq_len - 1)
                                if scales_slice.dim() == 3:
                                    scale_tok = scales_slice[0, tok, :].float().flatten()
                                else:
                                    scale_tok = scales_slice[0, tok].float().flatten() if scales_slice.dim() == 2 else scales_slice.float().flatten()
                                scale_mean = scale_tok.mean().item() if scale_tok.numel() else 0.0
                                scale_min = scale_tok.min().item() if scale_tok.numel() else 0.0
                                scale_max = scale_tok.max().item() if scale_tok.numel() else 0.0
                                bw_row = bitwidths_slice[tok].float()
                                q_tok = quantized_integers[0, tok, :].float()
                                q_abs_mean = q_tok.abs().mean().item()
                                logger.info(
                                    f"[SCALE_BW_TRACE] Layer 0 {self.proj_type}: scaling_type={scaling_type!r} "
                                    f"num_dims={num_dims} num_heads={num_heads} num_groups={num_groups} "
                                    f"scales_slice.shape={tuple(scales_slice.shape)} bitwidths.shape={tuple(bitwidths_slice.shape)}"
                                )
                                logger.info(
                                    f"[SCALE_BW_TRACE] Layer 0 {self.proj_type} token {tok}: "
                                    f"scale mean={scale_mean:.6f} min={scale_min:.6f} max={scale_max:.6f} | "
                                    f"bw min={bw_row.min().item():.1f} max={bw_row.max().item():.1f} mean={bw_row.mean().item():.2f} | "
                                    f"|Q|_mean={q_abs_mean:.4f}"
                                )
                            dequantized_latents = module.dequantize_with_per_dim_bitwidths(
                                quantized_integers,
                                scales_for_dequant,
                                target_dtype=_compute_dtype,
                                bitwidths_per_token=bitwidths_slice,
                            )
                            if _ADAPTIVE_DEBUG and layer_idx == 0:
                                log_key = (0, self.proj_type)
                                if log_key not in _per_dim_logged:
                                    _per_dim_logged.add(log_key)
                                    logger.info(
                                        f"[per-dim Layer {layer_idx} {self.proj_type}] dequant ok: scales {tuple(scales_slice.shape)}, "
                                        f"bitwidths {tuple(bitwidths_slice.shape)}, seq_len={seq_len}"
                                    )
                                # One-time magnitude log for token 0 and last token (compare with uniform path)
                                mag_key = (0, self.proj_type)
                                if mag_key not in _per_dim_magnitude_logged and dequantized_latents is not None and seq_len > 0:
                                    _per_dim_magnitude_logged.add(mag_key)
                                    t0 = dequantized_latents[0, 0, :].float().flatten()
                                    t_last = dequantized_latents[0, seq_len - 1, :].float().flatten()
                                    logger.info(
                                        f"[CACHE DEBUG] Layer 0 {self.proj_type} dequant magnitude: "
                                        f"token 0   min={t0.min().item():.4f} max={t0.max().item():.4f} mean={t0.mean().item():.4f} std={t0.std().item():.4f}"
                                    )
                                    logger.info(
                                        f"[CACHE DEBUG] Layer 0 {self.proj_type} dequant magnitude: "
                                        f"token {seq_len - 1} min={t_last.min().item():.4f} max={t_last.max().item():.4f} mean={t_last.mean().item():.4f} std={t_last.std().item():.4f}"
                                    )
                                # Debug: sink vs truncated vs recent magnitude when past window (trace garbled output)
                                sink = getattr(module, "attention_sink_tokens", 0)
                                recent_tokens = self.quantized_cache.get_recent_tokens()
                                n_recent_full = self.quantized_cache.get_n_recent_full()
                                if (
                                    dequantized_latents is not None
                                    and seq_len > 0
                                    and sink + recent_tokens > 0
                                    and seq_len > sink + recent_tokens
                                ):
                                    zone_key = (0, self.proj_type, "zone_magnitude")
                                    if zone_key not in _zone_magnitude_debug_logged:
                                        _zone_magnitude_debug_logged.add(zone_key)
                                        start_trunc = seq_len - recent_tokens
                                        end_trunc = seq_len - n_recent_full if n_recent_full > 0 else seq_len
                                        t_sink = dequantized_latents[0, 0, :].float().flatten()
                                        t_trunc = dequantized_latents[0, start_trunc, :].float().flatten() if start_trunc < seq_len else t_sink
                                        t_recent = dequantized_latents[0, min(end_trunc, seq_len - 1), :].float().flatten()
                                        logger.info(
                                            f"[ZONE_DEBUG] Layer 0 {self.proj_type} seq_len={seq_len} sink={sink} recent={recent_tokens} n_recent_full={n_recent_full} "
                                            f"trunc_zone=[{start_trunc},{end_trunc})"
                                        )
                                        logger.info(
                                            f"[ZONE_DEBUG] Layer 0 {self.proj_type} dequant magnitude SINK tok0:     "
                                            f"min={t_sink.min().item():.4f} max={t_sink.max().item():.4f} mean={t_sink.mean().item():.4f}"
                                        )
                                        logger.info(
                                            f"[ZONE_DEBUG] Layer 0 {self.proj_type} dequant magnitude TRUNC tok{start_trunc}: "
                                            f"min={t_trunc.min().item():.4f} max={t_trunc.max().item():.4f} mean={t_trunc.mean().item():.4f}"
                                        )
                                        logger.info(
                                            f"[ZONE_DEBUG] Layer 0 {self.proj_type} dequant magnitude RECENT tok{min(end_trunc, seq_len - 1)}: "
                                            f"min={t_recent.min().item():.4f} max={t_recent.max().item():.4f} mean={t_recent.mean().item():.4f}"
                                        )
                                        # Scale used for same tokens (mean scale)
                                        if scales_for_dequant.dim() == 3 and scales_for_dequant.shape[1] >= seq_len:
                                            s0 = scales_for_dequant[0, 0, :].float().mean().item()
                                            s_trunc = scales_for_dequant[0, start_trunc, :].float().mean().item() if start_trunc < seq_len else s0
                                            s_rec = scales_for_dequant[0, min(end_trunc, seq_len - 1), :].float().mean().item()
                                            logger.info(
                                                f"[ZONE_DEBUG] Layer 0 {self.proj_type} scale(alpha) mean: sink tok0={s0:.6f} trunc tok{start_trunc}={s_trunc:.6f} recent tok{min(end_trunc, seq_len - 1)}={s_rec:.6f}"
                                            )
                                # Adaptive vs static-6 dequant comparison (reader path: we have full seq from cache)
                                if (0, self.proj_type) not in _adaptive_error_debug_logged and seq_len >= 32:
                                    _adaptive_error_debug_logged.add((0, self.proj_type))
                                    try:
                                        start_trunc_d = seq_len - recent_tokens
                                        end_trunc_d = seq_len - n_recent_full if n_recent_full > 0 else seq_len
                                        bw = bitwidths_slice.clone().to(quantized_integers.device)
                                        mask_non16 = (bw != 16)
                                        bw_6 = bw.clone()
                                        bw_6[mask_non16] = 6
                                        rec_6 = module.dequantize_with_per_dim_bitwidths(
                                            quantized_integers,
                                            scales_for_dequant,
                                            target_dtype=None,
                                            bitwidths_per_token=bw_6,
                                        ).float()
                                        rec_adapt = dequantized_latents.float()
                                        for tok_idx in [start_trunc_d, end_trunc_d - 1]:
                                            if 0 <= tok_idx < seq_len:
                                                xa = rec_adapt[0, tok_idx, :].flatten()
                                                x6 = rec_6[0, tok_idx, :].flatten()
                                                diff_l2 = (xa - x6).pow(2).mean().sqrt().item()
                                                row = bitwidths_slice[tok_idx].float()
                                                logger.info(
                                                    f"[ADAPTIVE_ERROR_DEBUG] Layer 0 {self.proj_type} token {tok_idx}: "
                                                    f"bw_min={row.min().item():.1f} bw_max={row.max().item():.1f} "
                                                    f"diff_adapt_vs_6_L2={diff_l2:.6f}"
                                                )
                                    except Exception as e:
                                        logger.debug(f"[ADAPTIVE_ERROR_DEBUG] Layer 0 {self.proj_type} failed: {e}")
                        except (ValueError, RuntimeError) as e:
                            logger.warning(
                                f"[per-dim Layer {layer_idx} {self.proj_type}] dequant failed ({e}), "
                                f"using quantized as float (degraded)."
                            )
                            dequantized_latents = quantized_integers.float()
                    else:
                        if scales_slice is None or bitwidths_slice is None or bitwidths_slice.shape != (seq_len, num_dims):
                            logger.warning(
                                f"[per-dim Layer {layer_idx} {self.proj_type}] scales/bitwidths shape mismatch, "
                                f"using quantized as float (degraded). scales={scales_for_dequant.shape}, "
                                f"bitwidths={bitwidths_for_dequant.shape}, need scales (batch={batch_size}, seq={seq_len}, ...), "
                                f"bitwidths ({seq_len}, {num_dims})"
                            )
                        dequantized_latents = quantized_integers.float()
                # Restore magnitude: if quantize used channelwise_scaling (SC) but module at read has None,
                # apply stored SC from cache so dequant magnitude matches pre-cache.
                sc_applied = False
                sc_reason = ""
                if dequantized_latents is not None and getattr(module, "channelwise_scaling", None) is None:
                    sc = self.quantized_cache.get_channelwise_scaling(cache_key, self.proj_type)
                    if sc is None and cache_key != layer_name:
                        sc = self.quantized_cache.get_channelwise_scaling(layer_name, self.proj_type)
                    if sc is not None and sc.numel() == dequantized_latents.shape[-1]:
                        sc = sc.to(dequantized_latents.device).float().unsqueeze(0).unsqueeze(0)
                        dequantized_latents = dequantized_latents * sc
                        sc_applied = True
                        if layer_idx == 0 and (0, self.proj_type, "sc_restore") not in _per_dim_logged:
                            _per_dim_logged.add((0, self.proj_type, "sc_restore"))
                            logger.info(
                                f"[per-dim Layer 0 {self.proj_type}] applied channelwise_scaling from cache to restore magnitude"
                            )
                    else:
                        if sc is None:
                            sc_reason = "cache has no channelwise_scaling"
                        else:
                            sc_reason = f"sc shape mismatch (sc.numel()={sc.numel()}, latent dim={dequantized_latents.shape[-1]})"
                elif dequantized_latents is not None:
                    sc_reason = "module has channelwise_scaling (using module's SC at dequant)"
                # One-time log: which per-dim path (with or without SC) for layer 0 and quantize/dequant inverse
                if layer_idx == 0 and (0, self.proj_type, "sc_path") not in _per_dim_logged:
                    _per_dim_logged.add((0, self.proj_type, "sc_path"))
                    if sc_applied:
                        path = "WITH_SC (applied from cache)"
                    else:
                        path = f"WITHOUT_SC ({sc_reason or 'dequantized_latents unavailable'})"
                    logger.info(
                        f"[per-dim Layer 0 {self.proj_type}] channelwise_scaling path: {path}"
                    )
                    logger.info(
                        f"[per-dim Layer 0 {self.proj_type}] quantize path: tokenwise per-head alpha, "
                        f"stepsize=2*alpha/2^b; dequant uses same stored alpha → designed to be inverse. "
                        f"If magnitude is wrong, compare WRITE vs READ scales (alpha) min/max in logs."
                    )
            else:
                # Uniform path: dequantize with scales_dict
                # Log as soon as we enter this path (layer 0 only, once per k/v) so it's visible
                batch_size, seq_len, _ = quantized_integers.shape
                expected_elems = batch_size * seq_len
                # Sanity check: scales_dict must have expected_elems positions per group (or dequant will be wrong)
                scales_len_ok = True
                if scales_dict:
                    first_group = next(iter(scales_dict.values()), None)
                    if first_group is not None:
                        scales, base = first_group
                        n = scales.numel()
                        if n != expected_elems:
                            scales_len_ok = False
                            logger.warning(
                                f"[uniform Layer {layer_idx} {self.proj_type}] scales_dict length mismatch: "
                                f"scales numel={n}, expected batch*seq_len={expected_elems}. Dequant will be wrong."
                            )
                if layer_idx == 0:
                    log_key = (0, self.proj_type, "uniform")
                    if log_key not in _uniform_logged:
                        _uniform_logged.add(log_key)
                        logger.info(
                            f"[CACHE PATH Layer 0 {self.proj_type}] using UNIFORM dequant (scales_dict). "
                            f"seq_len={seq_len}, scales_dict keys={list(scales_dict.keys()) if scales_dict else []}, scales_len_ok={scales_len_ok}"
                        )
                    # Log the first time we read at seq_len==32 (BEFORE bitwidth update) with scale values and bitwidths
                    before_update_key = (0, self.proj_type)
                    if seq_len == _UNIFORM_LOG_AFTER_UPDATE_SEQ_LEN and before_update_key not in _uniform_logged_before_update:
                        _uniform_logged_before_update.add(before_update_key)
                        first_group = next(iter(scales_dict.values()), None) if scales_dict else None
                        if first_group is not None:
                            s, b = first_group
                            summary = _scale_summary(s, b)
                            logger.info(
                                f"[CACHE PATH Layer 0 {self.proj_type}] BEFORE BITWIDTH UPDATE (seq_len=32): {summary}"
                            )
                        if bitwidths is not None:
                            logger.info(
                                f"[CACHE PATH Layer 0 {self.proj_type}] BEFORE BITWIDTH UPDATE (seq_len=32) bitwidths at read: {_bitwidths_summary(bitwidths)}"
                            )
                        else:
                            logger.info(
                                f"[CACHE PATH Layer 0 {self.proj_type}] BEFORE BITWIDTH UPDATE (seq_len=32): no bitwidths in cache"
                            )
                    # Log the first time we read AFTER a bitwidth update has run (seq_len > 32), so scales_dict is checked after the update
                    after_update_key = (0, self.proj_type)
                    if seq_len > _UNIFORM_LOG_AFTER_UPDATE_SEQ_LEN and after_update_key not in _uniform_logged_after_update:
                        _uniform_logged_after_update.add(after_update_key)
                        first_group = next(iter(scales_dict.values()), None) if scales_dict else None
                        scales_shape = base_shape = None
                        summary_after = ""
                        if first_group is not None:
                            s, b = first_group
                            scales_shape, base_shape = s.shape, b.shape
                            summary_after = " " + _scale_summary(s, b)
                        logger.info(
                            f"[CACHE PATH Layer 0 {self.proj_type}] AFTER BITWIDTH UPDATE: seq_len={seq_len}, "
                            f"expected_elems={expected_elems}, scales_len_ok={scales_len_ok}, "
                            f"scales shape={scales_shape}, base shape={base_shape}"
                        )
                        if summary_after:
                            logger.info(
                                f"[CACHE PATH Layer 0 {self.proj_type}] AFTER scale/base summary:{summary_after}"
                            )
                        if bitwidths is not None:
                            logger.info(
                                f"[CACHE PATH Layer 0 {self.proj_type}] AFTER BITWIDTH UPDATE (seq_len={seq_len}) bitwidths at read: {_bitwidths_summary(bitwidths)}"
                            )
                        else:
                            logger.info(
                                f"[CACHE PATH Layer 0 {self.proj_type}] AFTER BITWIDTH UPDATE (seq_len={seq_len}): no bitwidths in cache"
                            )
                try:
                    dequantized_latents = module.dequantize_latent_from_integers(quantized_integers, scales_dict)
                except Exception as e:
                    logger.error(
                        f"[ERROR Layer {layer_idx} {self.proj_type}] Failed to dequantize: {e}\n"
                        f"  quantized_integers shape={quantized_integers.shape}, dtype={quantized_integers.dtype}, "
                        f"  range=[{quantized_integers.min().item():.2f}, {quantized_integers.max().item():.2f}]\n"
                        f"  scales_dict keys={list(scales_dict.keys()) if scales_dict else None}\n"
                        f"  module.num_groups={module.num_groups}, module.ranks={module.ranks}, sum(ranks)={sum(module.ranks)}"
                    )
                    if scales_dict:
                        for group_idx in sorted(scales_dict.keys())[:2]:
                            scales, base = scales_dict[group_idx]
                            logger.error(
                                f"  Group {group_idx}: scales shape={scales.shape}, base shape={base.shape}"
                            )
                    import traceback
                    logger.error(traceback.format_exc())
                    raise
            
            # Check if dequantized_latents has the correct shape
            expected_total_ranks = sum(module.ranks)
            if dequantized_latents.shape[2] != expected_total_ranks:
                logger.error(
                    f"[ERROR Layer {layer_idx} {self.proj_type}] Shape mismatch after dequantization:\n"
                    f"  dequantized_latents shape={dequantized_latents.shape}\n"
                    f"  expected last dim (sum of ranks)={expected_total_ranks}\n"
                    f"  module.ranks={module.ranks}, module.num_groups={module.num_groups}\n"
                    f"  quantized_integers shape={quantized_integers.shape}"
                )
                raise RuntimeError(
                    f"Dequantized latents shape mismatch: got {dequantized_latents.shape[2]}, expected {expected_total_ranks}"
                )
            
                # Reduced logging
            
            # Reconstruct BA(x) from the full sequence
            # Get the current module (try layer_name first, then cache_key after unwrap e.g. remove_hooks())
            current_module = None
            for n, m in self.model.named_modules():
                if n == layer_name:
                    current_module = m
                    break
            if current_module is None and cache_key != layer_name:
                for n, m in self.model.named_modules():
                    if n == cache_key:
                        current_module = m
                        break
            if current_module is None:
                logger.error(f"[ERROR Layer {layer_idx} {self.proj_type}] Could not find module {layer_name} in model!")
                return None
            
            # Check if we're using the correct module (not a stale reference)
            if id(current_module) != id(module):
                logger.warning(
                    f"[WARNING Layer {layer_idx} {self.proj_type}] Using STALE module reference! "
                    f"layer_mapping has module_id={id(module)}, but current module_id={id(current_module)}. "
                    f"Updating layer_mapping to use current module."
                )
                # Update layer_mapping to use the current module
                self.layer_mapping[key] = (layer_name, current_module)
                module = current_module
            
            # Reduced logging - only log on errors
            # if debug_layer:
            #     logger.debug(f"[RECONSTRUCT DEBUG Layer {layer_idx} {self.proj_type}]: Using module_id={id(module)}")
            
            try:
                # Reconstruct uses module parameters (often float16); dequant returns float32 — match dtypes to avoid matmul error
                module_dtype = next((p.dtype for p in module.parameters()), torch.float32)
                if dequantized_latents.dtype != module_dtype:
                    dequantized_latents = dequantized_latents.to(module_dtype)
                reconstructed_ba = module.reconstruct(dequantized_latents)
                # Reduced logging
                # if debug_layer:
                #     logger.debug(f"[RECONSTRUCT DEBUG Layer {layer_idx} {self.proj_type}]: After reconstruction: shape={reconstructed_ba.shape}")
            except Exception as e:
                logger.error(
                    f"[ERROR Layer {layer_idx} {self.proj_type}] Failed to reconstruct: {e}\n"
                    f"  dequantized_latents shape={dequantized_latents.shape}\n"
                    f"  module.ranks={module.ranks}, sum(ranks)={sum(module.ranks)}, module.num_groups={module.num_groups}\n"
                    f"  module_id={id(module)}"
                )
                raise
            
                # Reduced logging
            
            # Reshape to transformers format: (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
            batch_size, seq_len, out_features = reconstructed_ba.shape
            
            # Get num_heads from model config if available (for k_proj and v_proj, use num_key_value_heads)
            num_heads = self.num_key_value_heads
            
            if num_heads is None:
                # Fallback: infer from out_features
                for head_dim in [64, 128, 256]:
                    if out_features % head_dim == 0:
                        num_heads = out_features // head_dim
                        break
                
                if num_heads is None:
                    # Final fallback: assume 32 heads
                    num_heads = 32
            
            head_dim = out_features // num_heads
            
            # Reshape to (batch_size, num_heads, seq_len, head_dim)
            reconstructed_reshaped = reconstructed_ba.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # If exclude_last_token is True, return only previous tokens (excluding the last one)
            if exclude_last_token and seq_len > 0:
                reconstructed_reshaped = reconstructed_reshaped[:, :, :-1, :]  # Exclude last token
                # _get_quantized_ba logging removed for cleaner output
            
            # Reduced logging - only log on errors
            # if debug_layer:
            #     logger.debug(f"[RECONSTRUCT RESULT Layer {layer_idx} {self.proj_type}]: Returning shape={reconstructed_reshaped.shape}")

            return reconstructed_reshaped
            
        except Exception as e:
            logger.debug(f"Failed to get quantized BA(x) for layer {layer_idx} {self.proj_type}: {e}")
            return None


class QuantizedDynamicCache(DynamicCache):
    """
    Custom DynamicCache that stores ONLY compressed data.
    
    - update(): Does NOT store dense values, just marks that we have quantized data
    - key_cache/value_cache: Reconstruct BA(x) on-the-fly when accessed
    - Memory reduction is real: only compressed A(x) is stored, nothing dense
    """
    
    def __init__(
        self,
        quantized_cache,
        model=None,
        **kwargs
    ):
        """
        Initialize the quantized cache.
        
        Args:
            quantized_cache: QuantizedCache instance containing quantized A(x) values
            model: Optional model reference to get modules for reconstruction
            **kwargs: Additional arguments passed to DynamicCache.__init__
        """
        # Initialize base class first (this sets up key_cache and value_cache as empty dicts)
        super().__init__(**kwargs)
        
        self.quantized_cache = quantized_cache
        self.model = model
        # Map layer_idx to (layer_name, proj_type) for k_proj and v_proj
        self._layer_mapping = {}
        
        # Track the last sequence length for each layer to know if cache was just updated
        # This helps us determine if we should exclude the last token
        self._last_seq_lengths = {}  # (layer_idx, proj_type) -> seq_len

        # Cached attention modules (rotary_emb) per layer, built once in _build_layer_mapping().
        # Avoids repeated O(num_modules) named_modules() traversal at every decode step.
        self._attention_modules = {}  # layer_idx -> attention module with .rotary_emb
        
        # Replace key_cache and value_cache with our reconstructing dicts FIRST
        # These will reconstruct BA(x) on-the-fly when accessed
        self.key_cache = ReconstructingDict(quantized_cache, model, self._layer_mapping, 'k')
        self.value_cache = ReconstructingDict(quantized_cache, model, self._layer_mapping, 'v')
        
        # Now build the layer mapping (which will update the ReconstructingDict instances)
        if model is not None:
            self._build_layer_mapping()
    
    def _build_layer_mapping(self):
        """Build mapping from layer_idx to (layer_name, proj_type)."""
        from palu.model.modules.svd_linear import HeadwiseLowRankModule

        # Clear existing mappings
        self._layer_mapping = {}
        self._attention_modules = {}

        for name, module in self.model.named_modules():
            if isinstance(module, HeadwiseLowRankModule):
                try:
                    parts = name.split('.')
                    if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
                        layer_idx = int(parts[2])
                        if 'k_proj' in name:
                            self._layer_mapping[(layer_idx, 'k')] = (name, module)
                        elif 'v_proj' in name:
                            self._layer_mapping[(layer_idx, 'v')] = (name, module)
                except (ValueError, IndexError):
                    pass
            # Cache attention modules with rotary_emb to avoid repeated named_modules() traversal.
            for prefix in ("model.layers.", "model.model.layers."):
                if name.startswith(prefix) and name.endswith(".self_attn"):
                    try:
                        layer_idx = int(name.split('.')[2])
                        # Prefer .inner when PaluLlamaAttention wraps the original attention
                        attn = module
                        if hasattr(module, "inner") and hasattr(module.inner, "rotary_emb"):
                            attn = module.inner
                        if hasattr(attn, "rotary_emb"):
                            self._attention_modules[layer_idx] = attn
                    except (ValueError, IndexError):
                        pass

        # Update ReconstructingDict layer mappings (only if they're already ReconstructingDict instances)
        if hasattr(self, 'key_cache') and isinstance(self.key_cache, ReconstructingDict):
            self.key_cache.layer_mapping = self._layer_mapping
            self.key_cache._attention_modules = self._attention_modules
        if hasattr(self, 'value_cache') and isinstance(self.value_cache, ReconstructingDict):
            self.value_cache.layer_mapping = self._layer_mapping
            self.value_cache._attention_modules = self._attention_modules
    
    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs=None,
    ):
        """
        Update cache and return the FULL sequence (cached + new) for attention computation.
        
        CRITICAL: Transformers expects update() to return the FULL sequence (cached previous tokens + new token),
        not just the new token! The returned key_states and value_states are used directly for attention.
        
        Flow:
        1. HeadwiseLowRankModule.forward() updates QuantizedCache with full sequence (cached + new)
        2. Transformers calls update() with new token's key/value states
        3. We reconstruct the FULL sequence from QuantizedCache and return it
        4. Transformers uses the returned full sequence for attention computation
        
        We do NOT store dense values - only quantized integers in QuantizedCache.
        Reconstruction happens on-the-fly from QuantizedCache.
        
        Args:
            key_states: New token's key states from transformers - shape (batch, num_heads, 1, head_dim)
            value_states: New token's value states from transformers - shape (batch, num_heads, 1, head_dim)
            layer_idx: Layer index
            cache_kwargs: Optional cache kwargs (unused, but required by transformers interface)
            
        Returns:
            Tuple of (full_key_states, full_value_states) - FULL sequence (cached + new)
            Shape: (batch, num_heads, total_seq_len, head_dim) where total_seq_len includes all cached tokens + new token
        """
        # Debug logging for layer 0 only to keep output manageable
        debug_layer = (layer_idx == 0)
        
        # Reduced logging - only log on first update
        if debug_layer and not hasattr(self, '_logged_update_input'):
            # Update input logging removed for cleaner output
            pass
        
        # IMPORTANT: Transformers calls update() with the NEW token's key/value states.
        # We need to ensure the cache contains the FULL sequence so transformers can access it.
        #
        # The flow is:
        # 1. HeadwiseLowRankModule.forward() updates QuantizedCache with full sequence (cached + new)
        # 2. Transformers calls update() with new token's key/value states
        # 3. We reconstruct the FULL sequence from QuantizedCache and store it in the cache
        # 4. We return the new token's values as-is (transformers expects this)
        # 5. When transformers accesses the cache via __getitem__(), it gets the FULL sequence
        #
        # CRITICAL: We need to store the full sequence in the cache during update() so transformers
        # can access it. Even though __getitem__() reconstructs on-the-fly, storing here ensures
        # the cache is "updated" and transformers will use it.
        
        # CRITICAL: We need to reconstruct FRESH from QuantizedCache, not use stale cached values.
        # If we call self.key_cache[layer_idx], it will use the stale value from _reconstructed_cache.
        # Instead, we directly call _get_quantized_ba() to reconstruct fresh from QuantizedCache.
        
        # CRITICAL: We need to store the full sequence in the cache so transformers sees it as "updated".
        # However, we must reconstruct FRESH from QuantizedCache to ensure we have the latest data.
        # Even though __getitem__() always reconstructs fresh, storing here ensures transformers
        # sees the cache as updated and will use it.
        #
        # The flow is:
        # 1. HeadwiseLowRankModule.forward() updates QuantizedCache with full sequence (cached + new)
        # 2. Transformers calls update() with new token's key/value states
        # 3. We reconstruct the FULL sequence fresh from QuantizedCache and store it
        # 4. We return the new token's values as-is (transformers expects this)
        # 5. When transformers accesses the cache via __getitem__(), it reconstructs fresh (but we also store for visibility)
        # 6. Transformers uses the full sequence for attention computation
        
        # CRITICAL INSIGHT: We should NOT store the full reconstructed sequence in _reconstructed_cache!
        # That defeats the purpose of compression. Instead:
        # 1. Only store quantized integers in QuantizedCache (already done by HeadwiseLowRankModule.forward())
        # 2. When transformers accesses the cache via __getitem__(), reconstruct BA(x) on-the-fly from quantized integers
        # 3. Do NOT store anything in _reconstructed_cache - always reconstruct fresh from QuantizedCache
        #
        # The flow is:
        # 1. HeadwiseLowRankModule.forward() updates QuantizedCache with quantized integers (full sequence)
        # 2. Transformers calls update() with new token's key/value states
        # 3. We return the new token's values as-is (transformers expects this)
        # 4. When transformers accesses the cache via __getitem__(), we reconstruct BA(x) from quantized integers
        # 5. Transformers uses the reconstructed full sequence for attention computation
        #
        # We do NOT need to store anything in _reconstructed_cache because __getitem__() always reconstructs fresh.
        # This ensures we only store compressed data (quantized integers) and nothing dense.
        
        # CRITICAL FIX: Transformers expects update() to return the FULL sequence (cached + new),
        # not just the new token! The returned key_states and value_states are used directly for attention.
        #
        # Flow:
        # 1. HeadwiseLowRankModule.forward() updates QuantizedCache with full sequence (cached + new)
        # 2. Transformers calls update() with new token's key/value states
        # 3. We reconstruct the FULL sequence from QuantizedCache and return it
        # 4. Transformers uses the returned full sequence for attention computation
        
        # Reconstruct full sequence for keys
        full_key_states = None
        if key_states is not None:
            # Prompt phase: key_states has multiple tokens (e.g. 7). Return as-is — no concat with cache.
            # Decode phase: key_states has 1 token. Get previous from cache (exclude_last_token) and concat.
            new_tokens_len = key_states.shape[2]
            if new_tokens_len > 1:
                full_key_states = key_states
            else:
                # Decode: 1 new token. Reconstruct previous from cache, apply RoPE, concat with new token.
                previous_key_states_from_cache = self.key_cache._get_quantized_ba(layer_idx, debug_layer=debug_layer, exclude_last_token=True)

                if previous_key_states_from_cache is None:
                    full_key_states = key_states
                else:
                    # We have previous tokens from cache (without RoPE)
                    # Apply RoPE to previous tokens, then concatenate with new token (already has RoPE from transformers)
                    try:
                        # Use cached attention module to avoid repeated named_modules() traversal.
                        attention_module = self._attention_modules.get(layer_idx)
                        if attention_module is not None and hasattr(attention_module, 'rotary_emb'):
                            prev_seq_len = previous_key_states_from_cache.shape[2]
                            position_ids_prev = torch.arange(prev_seq_len, device=previous_key_states_from_cache.device).unsqueeze(0)

                            try:
                                cos, sin = attention_module.rotary_emb(previous_key_states_from_cache, position_ids=position_ids_prev)
                            except (TypeError, ValueError):
                                try:
                                    cos, sin = attention_module.rotary_emb(previous_key_states_from_cache, seq_len=prev_seq_len)
                                except (TypeError, ValueError):
                                    cos, sin = attention_module.rotary_emb(previous_key_states_from_cache)

                            # Normalize cos/sin to (batch, seq_len, head_dim) so that
                            # apply_rotary_pos_emb's unsqueeze(1) gives (batch, 1, seq_len, head_dim)
                            # which broadcasts correctly against (batch, num_heads, seq_len, head_dim).
                            # Old-style rotary_emb(seq_len=N) returns 2D (seq_len, dim) or 4D (1,1,seq,dim).
                            if cos.dim() == 2:  # (seq_len, dim) → (1, seq_len, dim)
                                cos = cos.unsqueeze(0)
                                sin = sin.unsqueeze(0)
                            elif cos.dim() == 4:  # (1, 1, seq_len, dim) → (1, seq_len, dim)
                                cos = cos.squeeze(1)
                                sin = sin.squeeze(1)

                            if HAS_ROPE_UTILITY:
                                try:
                                    previous_key_states_rope, _ = apply_rotary_pos_emb(
                                        previous_key_states_from_cache,
                                        previous_key_states_from_cache,
                                        cos,
                                        sin,
                                        position_ids_prev
                                    )
                                    full_key_states = torch.cat([previous_key_states_rope, key_states], dim=2)
                                except Exception as rope_error:
                                    if debug_layer:
                                        import traceback
                                        logger.error(f"Failed to apply RoPE: {rope_error}")
                                        logger.error(traceback.format_exc())
                                    raise rope_error
                            else:
                                full_key_states = torch.cat([previous_key_states_from_cache, key_states], dim=2)
                        else:
                            full_key_states = torch.cat([previous_key_states_from_cache, key_states], dim=2)
                    except Exception as e:
                        # Fallback: concatenate without RoPE
                        full_key_states = torch.cat([previous_key_states_from_cache, key_states], dim=2)

        # Reconstruct previous tokens for values.
        # In LLaMA, values do NOT use RoPE (only keys and queries do), so no position-embedding fix is needed.
        # We must return the same sequence length as keys (cached_prev + new) so attention gets aligned K and V.
        full_value_states = None
        if value_states is not None:
            # Prompt phase: value_states has multiple tokens — return as-is. Decode: 1 token — concat with cache.
            new_val_len = value_states.shape[2]
            if new_val_len > 1:
                full_value_states = value_states
            else:
                previous_value_states_from_cache = self.value_cache._get_quantized_ba(layer_idx, debug_layer=debug_layer, exclude_last_token=True)
                if previous_value_states_from_cache is None:
                    full_value_states = value_states
                else:
                    prev_v = previous_value_states_from_cache
                    if prev_v.device != value_states.device or prev_v.dtype != value_states.dtype:
                        prev_v = prev_v.to(device=value_states.device, dtype=value_states.dtype)
                    full_value_states = torch.cat([prev_v, value_states], dim=2)
        
        # Ensure key and value sequence lengths match (required for correct attention)
        if full_key_states is not None and full_value_states is not None:
            k_len = full_key_states.shape[2]
            v_len = full_value_states.shape[2]
            if k_len != v_len:
                if debug_layer:
                    logger.warning(
                        f"[CACHE UPDATE Layer {layer_idx}] Key/value seq length mismatch: key={k_len}, value={v_len}. "
                        f"Trimming to min to avoid attention errors."
                    )
                min_len = min(k_len, v_len)
                full_key_states = full_key_states[:, :, :min_len, :].contiguous()
                full_value_states = full_value_states[:, :, :min_len, :].contiguous()
        
        # Always log a few cache length updates for layer 0 to help debug position/cache issues.
        if layer_idx == 0 and full_key_states is not None:
            k_len = full_key_states.shape[2]
            v_len = full_value_states.shape[2] if full_value_states is not None else None
            if not hasattr(self, "_debug_update_count"):
                self._debug_update_count = 0
            self._debug_update_count += 1
            if self._debug_update_count <= 5:  # first 5 updates only
                logger.info(
                    f"[CACHE DEBUG] update() layer 0: full_key_seq_len={k_len}, full_value_seq_len={v_len} "
                    f"(update #{self._debug_update_count})"
                )
        # Reduced logging - only log on first return
        if debug_layer and not hasattr(self, '_logged_return'):
            self._logged_return = True
            if full_key_states is not None:
                logger.debug(
                    f"[TRANSFORMERS UPDATE Layer {layer_idx}]: "
                    f"Returning k: shape={full_key_states.shape}, v: shape={full_value_states.shape if full_value_states is not None else None}"
                )
        
        # CRITICAL: Transformers' DynamicCache.update() stores the returned values in key_cache[layer_idx] and value_cache[layer_idx].
        # Even though we've replaced key_cache/value_cache with ReconstructingDict, we should still store the full sequence
        # so that if transformers accesses the cache later, it gets the correct values.
        # However, since we're using ReconstructingDict which reconstructs on-the-fly, we don't need to store here.
        # The parent class will try to store, but ReconstructingDict.__setitem__() will handle it.
        # Actually, let's NOT call the parent's update() because it might interfere with our logic.
        # Instead, we'll just return the values and let transformers use them directly.
        
        # Track _seen_tokens like DynamicCache.update() does — some HF versions use this
        # directly (not via get_seq_length()) for cache_position / causal mask computation.
        # Without this, _seen_tokens stays at 0 and every decode token gets position 0.
        if hasattr(self, '_seen_tokens') and layer_idx == 0:
            if key_states is not None:
                self._seen_tokens += key_states.shape[-2]

        # Return the FULL sequence (cached + new), not just the new token
        return (full_key_states, full_value_states)
    
    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Get sequence length from _seen_tokens (kept in sync by update()) or QuantizedCache.
        Some HF versions read _seen_tokens directly; keeping it authoritative avoids mismatches."""
        # Prefer _seen_tokens when available — it is incremented by our update() override
        # and matches what DynamicCache.get_seq_length() returns in the parent class.
        if hasattr(self, '_seen_tokens') and self._seen_tokens > 0:
            return self._seen_tokens
        if layer_idx is not None:
            for (idx, proj_type), (layer_name, _) in self._layer_mapping.items():
                if idx != layer_idx:
                    continue
                cache_key = _cache_key_for_layer_name(layer_name)
                # Try normalized key first (pre-wrap), then raw layer_name (post-wrap e.g. .inner)
                seq_len = self.quantized_cache.get_seq_length(cache_key)
                if seq_len <= 0 and cache_key != layer_name:
                    seq_len = self.quantized_cache.get_seq_length(layer_name)
                if seq_len > 0:
                    if layer_idx == 0:
                        logger.debug(f"[CACHE DEBUG] get_seq_length(layer_idx={layer_idx}) -> {seq_len} (from layer lookup)")
                    return seq_len
        
        # Fallback: max over all layers (e.g. before any layer has run, or no match)
        fallback = max(self.quantized_cache.seq_lengths.values()) if self.quantized_cache.seq_lengths else 0
        if layer_idx == 0:
            logger.debug(f"[CACHE DEBUG] get_seq_length(layer_idx={layer_idx}) -> {fallback} (fallback)")
        return fallback
