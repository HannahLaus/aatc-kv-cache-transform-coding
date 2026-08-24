"""
Qwen2-specific sliding-window KV cache integration.

Keeps the most recent `recent_tokens` and the first `attention_sink_tokens`
of the cache at full 16-bit precision, requantizing older tokens down to their
base bitwidths in steps as the window slides.
"""

import os
from typing import Optional, Dict, Tuple, Callable

import torch
import torch.nn as nn

from ..modules.svd_linear import QuantizedCache, HeadwiseLowRankModule, _to_cache_quant_dtype

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Global flag to enable detailed debug logging in hot paths.
_ADAPTIVE_DEBUG = os.environ.get("PALU_ADAPTIVE_DEBUG", "0") == "1"

# One-time debug: sliding-step requant (layer 0) to trace garbled output past sink+recent window
_sliding_requant_debug_logged: set = set()


def _bitwidths_summary(bw: torch.Tensor) -> str:
    """Summary of bitwidths tensor for logging."""
    b = bw.detach().float()
    flat = b.flatten()
    n = flat.numel()
    f5 = flat[:5].tolist() if n >= 5 else flat.tolist()
    l5 = flat[-5:].tolist() if n >= 5 else flat.tolist()
    return (
        f"shape={tuple(bw.shape)} min={b.min().item():.1f} max={b.max().item():.1f} mean={b.mean().item():.2f} "
        f"first5={[round(x, 1) for x in f5]} last5={[round(x, 1) for x in l5]}"
    )


# Cache/scales_dict use uniform scale (x = Q*scale). dequantize_with_per_dim_bitwidths expects alpha (stepsize = 2*alpha/2^b).
# For 8-bit: alpha = scale * 2^7. Use this when passing scales from cache/scales_dict to dequantize_with_per_dim_bitwidths.
_UNIFORM_SCALE_TO_ALPHA = 2.0 ** 7


def _scales_tensor_summary(s: torch.Tensor) -> str:
    """Summary of scales tensor (no base; symmetric)."""
    s = s.detach().float().flatten()
    n = s.numel()
    f3 = s[:3].tolist() if n >= 3 else s.tolist()
    l3 = s[-3:].tolist() if n >= 3 else s.tolist()
    return (
        f"shape numel={n} min={s.min().item():.6f} max={s.max().item():.6f} mean={s.mean().item():.6f} "
        f"first3={[round(x, 5) for x in f3]} last3={[round(x, 5) for x in l3]}"
    )


def _safe_key_cache_get(key_cache, layer_idx: int):
    """Get key_cache[layer_idx] without using 'in' on a list/tuple of tensors (avoids 'Boolean value of Tensor is ambiguous')."""
    if key_cache is None:
        return None
    if isinstance(key_cache, dict):
        return key_cache.get(layer_idx)
    if isinstance(key_cache, (list, tuple)) and not isinstance(key_cache, torch.Tensor):
        if 0 <= layer_idx < len(key_cache):
            return key_cache[layer_idx]
        return None
    return None


class PaluQwen2Attention(nn.Module):
    """
    Lightweight wrapper around a Qwen2 attention module.

    Delegates the attention computation to the wrapped module (`inner`), which can
    use fast SDPA / FlashAttention, and triggers the sliding-window requantize step
    at the right point in the decode loop.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        run_sliding_step_if_needed: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self.run_sliding_step_if_needed = run_sliding_step_if_needed

        # Mirror key attributes expected from Qwen2Attention / Qwen2SdpaAttention
        self.hidden_size = getattr(inner, "hidden_size", None)
        self.num_heads = getattr(inner, "num_heads", None)
        self.head_dim = getattr(inner, "head_dim", None)
        self.num_key_value_heads = getattr(inner, "num_key_value_heads", self.num_heads)

        self.q_proj = inner.q_proj
        self.k_proj = inner.k_proj
        self.v_proj = inner.v_proj
        self.o_proj = inner.o_proj
        self.rotary_emb = inner.rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        # Run sliding requantize step before appending when cache has exactly (sink + recent_tokens)
        # tokens, so we truncate the older half of the recent window at the right time (not after append).
        if past_key_value is not None and self.run_sliding_step_if_needed is not None:
            try:
                self.run_sliding_step_if_needed()
            except Exception as e:
                logger.warning(f"[PaluQwen2Attention] run_sliding_step_if_needed failed: {e}")
        # Forward position_embeddings and cache_position explicitly so they are never dropped (HF inner
        # needs them for RoPE and causal mask). Use a copy of kwargs so we don't mutate the caller's dict.
        inner_kwargs = dict(kwargs)
        position_embeddings = inner_kwargs.pop("position_embeddings", None)
        cache_position = inner_kwargs.pop("cache_position", None)
        if position_embeddings is not None:
            inner_kwargs["position_embeddings"] = position_embeddings
        if cache_position is not None:
            inner_kwargs["cache_position"] = cache_position
        # Log position/cache shapes for layer 0 once to help debug RoPE / causal masking.
        if self.layer_idx == 0 and not getattr(self, "_debug_pos_logged", False):
            self._debug_pos_logged = True
            pe_shape = tuple(position_embeddings[0].shape) if position_embeddings and len(position_embeddings) else None
            cp_shape = tuple(cache_position.shape) if cache_position is not None else None
            pid_shape = tuple(position_ids.shape) if position_ids is not None else None
            logger.info(
                f"[CACHE DEBUG] Layer 0 attention: position_embeddings cos.shape={pe_shape}, "
                f"cache_position.shape={cp_shape}, position_ids.shape={pid_shape}"
            )
        # Run the original HF attention (this will use SDPA / FlashAttention if enabled)
        attn_output, _, present_key_value = self.inner(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,  # keep fast path
            use_cache=use_cache,
            **inner_kwargs,
        )

        # We mimic standard Qwen2Attention output signature: (attn_output, attn_weights, present_key_value)
        return attn_output, None, present_key_value

class Qwen2AdaptiveQuantizationManager:
    """
    Manager for the sliding-window KV cache in Qwen2 models.

    This class:
    1. Patches HeadwiseLowRankModule.forward() to automatically use QuantizedCache
    2. Wraps Qwen2Attention so the sliding step fires at the right point in decode
    3. Requantizes tokens leaving the recent window down to their base bitwidths
    """
    
    def __init__(
        self,
        model,
        quantized_cache: Optional[QuantizedCache] = None,
        recent_tokens: int = 0,
        attention_sink_tokens: int = 0,
        past_key_values: Optional[torch.Tensor] = None,
        sliding_step_size: Optional[int] = None,
    ):
        """
        Initialize the sliding-window KV cache manager for a Qwen2 model.

        Args:
            model: PaluQwen2ForCausalLM model instance
            quantized_cache: Optional QuantizedCache instance (creates new one if None)
            recent_tokens: Sliding recent window: last N tokens at 16 bits; when full,
                quantize the older sliding_step_size tokens down to their base
                bitwidths. 0 = disable.
            attention_sink_tokens: First N tokens kept at 16 bits (full precision).
                0 = disable; 8 = typical sink.
            past_key_values: Optional reference to transformers' past_key_values cache
                (updated when requantizing cached latents).
            sliding_step_size: How many tokens to requantize each time the recent
                window fills. Default: recent_tokens // 2 (halve the window). Set to a
                small value (e.g. 16) to keep the window tightly between
                (recent_tokens - sliding_step_size) and recent_tokens.
        """
        self.model = model
        self.quantized_cache = quantized_cache or QuantizedCache()
        self.recent_tokens = max(0, recent_tokens)
        # How many tokens to requantize each sliding step. Default: half the window.
        self.sliding_step_size = max(1, sliding_step_size) if sliding_step_size is not None \
            else max(1, recent_tokens // 2)
        self.quantized_cache.recent_tokens = self.recent_tokens
        self.attention_sink_tokens = max(0, attention_sink_tokens)
        self.quantized_cache.attention_sink_tokens = self.attention_sink_tokens
        self.past_key_values = past_key_values  # Reference to transformers' cache (can be updated later)

        # (Unused — budget is now computed from actual stored bitwidths each check.)

        # Store original forward methods and attention wrappers
        self.original_forwards = {}
        self.attention_hooks = []  # kept for API compatibility (no longer used)
        self.wrapped_attentions = []
        # Cache for {module_name: HeadwiseLowRankModule} — built once after all wrapping is done
        # so patched_forward and _run_sliding_recent_quantize_step avoid O(N) named_modules() scans.
        self._module_cache: dict = {}
        # Pre-bound forward methods {original_name: bound_method} — avoids types.MethodType() on
        # every forward call (built alongside _module_cache after wrapping is complete).
        self._bound_forward_cache: dict = {}

        # Prefix for layer names as seen by named_modules() (e.g. "model" or "model.model")
        self._layer_name_prefix: Optional[str] = None

        # Patch HeadwiseLowRankModule forward methods
        self._patch_projection_modules()

        # Wrap attention layers so the sliding step fires at the right point in decode
        self._wrap_attention_layers()

        # Build module cache now that all wrapping is complete (stable names)
        self._build_module_cache()

    
    def update_past_key_values(self, past_key_values):
        """Update the reference to transformers' past_key_values cache."""
        self.past_key_values = past_key_values
    
    def _patch_projection_modules(self):
        """Patch HeadwiseLowRankModule.forward() to automatically use QuantizedCache."""
        import re as _re
        # Initialize class-level tracking dicts once here so patched_forward closures
        # don't repeat hasattr/setdefault checks on every forward call.
        if not hasattr(Qwen2AdaptiveQuantizationManager, '_called_layers'):
            Qwen2AdaptiveQuantizationManager._called_layers = set()
        if not hasattr(Qwen2AdaptiveQuantizationManager, '_call_count'):
            Qwen2AdaptiveQuantizationManager._call_count = {}
        patched_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, HeadwiseLowRankModule):
                # Determine projection type from name
                proj_type = None
                if 'k_proj' in name:
                    proj_type = 'k'
                elif 'v_proj' in name:
                    proj_type = 'v'
                
                if proj_type is not None:
                    # CRITICAL: Don't capture 'module' in the closure - look it up dynamically instead
                    # This ensures that if the module is replaced, we always use the current module from the model
                    # Store original forward by NAME, not by module object (since modules can be replaced)
                    if name not in self.original_forwards:
                        # Store original as-is (bound method, closure, or plain function).
                        # Do NOT extract __func__ — accelerate may store new_forward as a
                        # plain closure (no __self__) whose module is captured in __closure__.
                        # Extracting __func__ and re-wrapping with MethodType would add an
                        # extra positional arg and trigger "multiple values for argument" errors.
                        self.original_forwards[name] = module.forward
                    
                    # Create patched forward that automatically passes cache
                    # Use default argument trick to ensure each closure captures its own copy
                    # This is critical: default arguments are evaluated at function definition time,
                    # so each closure gets its own copy of the values
                    def patched_forward(hidden_states, cache_for_quantization=None, layer_name=None, proj_type=None, 
                                      _captured_name=name, _captured_proj_type=proj_type, **kwargs):
                        # Look up the current module from the pre-built cache (O(1)) instead of
                        # scanning named_modules() on every forward call.  The cache stores both
                        # the pre-wrap name and the post-wrap .self_attn.inner. alias, so a single
                        # get() handles both cases.
                        current_module = self._module_cache.get(_captured_name)
                        if current_module is None:
                            logger.error(f"Could not find module {_captured_name} in model hierarchy!")
                            raise RuntimeError(f"Module {_captured_name} not found and no original forward available")
                        
                        # Use the pre-bound forward method built once in _build_module_cache.
                        original_forward = self._bound_forward_cache.get(_captured_name)
                        if original_forward is None:
                            logger.warning(f"No pre-bound forward for {_captured_name}, falling back to current forward")
                            original_forward = current_module.forward
                        
                        # Log which module object is actually being called
                        current_module_id = id(current_module)
                        
                        # Always use our cache and layer info from default arguments
                        # Ignore any cache_for_quantization, layer_name, proj_type passed by caller
                        # and use the ones from default arguments
                        # One-time log per layer (first few layers only) — runs at most once per name
                        if _captured_name not in Qwen2AdaptiveQuantizationManager._called_layers:
                            Qwen2AdaptiveQuantizationManager._called_layers.add(_captured_name)
                            m = _re.search(r'layers\.(\d+)', _captured_name or "")
                            if m and int(m.group(1)) in (0, 1, 2):
                                logger.info(
                                    f"[FORWARD CALL] {_captured_name}[{_captured_proj_type}]: "
                                    f"Using current_module_id={id(current_module)} (looked up dynamically from model)"
                                )
                        
                        # Extract layer_idx from layer_name (e.g., "model.layers.0.self_attn.k_proj" -> 0)
                        layer_idx = None
                        if _captured_name:
                            try:
                                # Extract number between "model.layers." and ".self_attn"
                                parts = _captured_name.split('.')
                                if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
                                    layer_idx = int(parts[2])
                            except (ValueError, IndexError):
                                pass
                        
                        output = original_forward(
                            hidden_states,
                            cache_for_quantization=self.quantized_cache,
                            layer_name=_captured_name,  # Use the default argument, not the parameter
                            proj_type=_captured_proj_type,  # Use the default argument, not the parameter
                            past_key_values=self.past_key_values,  # Pass transformers' cache to update it
                            layer_idx=layer_idx  # Pass layer index for updating past_key_values
                        )
                        
                        # Log what we return to transformers for layer 0 k_proj
                        # Forward return logging removed for cleaner output
                        
                        return output
                    
                    # PATCH logging removed for cleaner output
                    module.forward = patched_forward
                    patched_count += 1
                    # Patched forward logging removed for cleaner output
                    
                    if proj_type in ('v', 'k') and (self.recent_tokens > 0 or self.attention_sink_tokens > 0):
                        # Sliding window / attention sink: set the window params
                        # so that HeadwiseLowRankModule.forward keeps recent/sink tokens at 16 bits.
                        module.recent_tokens = self.recent_tokens
                        module.attention_sink_tokens = self.attention_sink_tokens
        
        logger.info(f"Total modules patched: {patched_count} (expected: 64 for 32 layers * 2 projections)")
    
    def _wrap_attention_layers(self):
        """
        Replace each layer's self-attention with PaluQwen2Attention, which
        delegates to the original attention module and triggers the sliding-window
        requantize step at the right point in the decode loop.
        """
        # Support both model.model.layers (e.g. PaluQwen2ForCausalLM) and model.layers (e.g. Qwen2Model / single "model" prefix)
        layers_seq = getattr(self.model, "model", None)
        if layers_seq is not None:
            layers_seq = getattr(layers_seq, "layers", None)
        if layers_seq is None:
            layers_seq = getattr(self.model, "layers", None)
        if layers_seq is None:
            logger.warning("[ATTN WRAP] No model.model.layers or model.layers found; skipping attention wrap")
            return
        num_layers = len(layers_seq)
        logger.info(f"[ATTN WRAP] Wrapping attention layers for {num_layers} layers")

        for layer_idx, layer in enumerate(layers_seq):
            attention = layer.self_attn

            # Avoid double-wrapping
            if isinstance(attention, PaluQwen2Attention):
                continue

            # Only wrap modules that look like Qwen2 attention (have q_proj/k_proj and rotary_emb)
            if not (hasattr(attention, "q_proj") and hasattr(attention, "k_proj") and hasattr(attention, "rotary_emb")):
                continue

            if layer_idx in [0, 1, 2]:
                logger.info(f"[ATTN WRAP] Wrapping attention for layer {layer_idx} ({attention.__class__.__name__})")

            wrapped = PaluQwen2Attention(
                inner=attention,
                layer_idx=layer_idx,
                run_sliding_step_if_needed=self._maybe_run_sliding_step_before_append if (layer_idx == 0 and self.recent_tokens > 0) else None,
            )
            wrapped.eval()  # New submodules default to training=True
            layer.self_attn = wrapped
            self.wrapped_attentions.append((layer, attention))
    
    def _build_module_cache(self) -> None:
        """Build lookup caches for O(1) access in hot paths.

        _module_cache: {name: HeadwiseLowRankModule} — registered under both the
        actual post-wrap name and the pre-wrap alias so patched_forward closures
        (which capture the pre-wrap name) always get a hit.

        _bound_forward_cache: {original_name: bound_method} — pre-binds each
        original forward function to its module so patched_forward avoids calling
        types.MethodType() on every token.
        """
        import types as _types
        self._module_cache = {}
        self._bound_forward_cache = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, HeadwiseLowRankModule):
                continue
            self._module_cache[name] = module
            # Register under the alternate wrapping path so both pre- and post-wrap
            # names resolve to the same module object.
            if ".self_attn.inner." in name:
                alt = name.replace(".self_attn.inner.", ".self_attn.", 1)
                self._module_cache[alt] = module
            elif ".self_attn." in name:
                alt = name.replace(".self_attn.", ".self_attn.inner.", 1)
                self._module_cache[alt] = module
            else:
                alt = None

        # Pre-bind original forward functions to their (post-wrap) modules.
        # original_forwards keys use the pre-wrap name; _module_cache[pre-wrap name]
        # already resolves to the correct post-wrap module object via the alias above.
        for orig_name, func in self.original_forwards.items():
            target_module = self._module_cache.get(orig_name)
            if target_module is not None and callable(func):
                # Bound methods (MethodType) and closures already carry their module reference.
                # Wrapping them again with MethodType would inject an extra positional arg and
                # cause "multiple values for argument" errors (seen with accelerate closures).
                # Only unbound, non-closure plain functions need explicit binding.
                is_bound = isinstance(func, _types.MethodType) or hasattr(func, '__self__')
                is_closure = isinstance(func, _types.FunctionType) and getattr(func, '__closure__', None)
                if is_bound or is_closure:
                    self._bound_forward_cache[orig_name] = func
                else:
                    self._bound_forward_cache[orig_name] = _types.MethodType(func, target_module)

        # Pre-build ordered K/V entry lists and stacked tensors for vectorized sliding step.
        # Canonical (non-.inner.) names only, sorted by layer index so layer 0 is first.
        import re as _re_sliding
        k_entries, v_entries, seen = [], [], set()
        for name in sorted(self._module_cache.keys(),
                           key=lambda n: int(m.group(1)) if (m := _re_sliding.search(r'layers\.(\d+)', n)) else 999):
            if ".self_attn.inner." in name or "self_attn" not in name:
                continue
            mod = self._module_cache[name]
            if "k_proj" in name and (name, "k") not in seen:
                seen.add((name, "k")); k_entries.append((name, mod))
            elif "v_proj" in name and (name, "v") not in seen:
                seen.add((name, "v")); v_entries.append((name, mod))
        self._sliding_k_entries = k_entries
        self._sliding_v_entries = v_entries

        def _try_stack(entries, attr):
            ts = []
            for _, m in entries:
                t = getattr(m, attr, None)
                if t is None:
                    return None
                if t.device.type == "meta":
                    return None  # offloaded to meta device (device_map="auto") — skip vectorization
                ts.append(t.detach().cpu().float())
            try: return torch.stack(ts) if ts else None
            except RuntimeError: return None  # different shapes → can't stack

        self._k_fc_scales = _try_stack(k_entries, "channelwise_scalings")  # [L_k, D] or None
        self._v_fc_scales = _try_stack(v_entries, "channelwise_scalings")  # [L_v, D] or None
        self._k_base_bw   = _try_stack(k_entries, "base_bitwidths")        # [L_k, D] or None
        self._v_base_bw   = _try_stack(v_entries, "base_bitwidths")        # [L_v, D] or None

        # Vectorization only works when all modules use "factored" scaling and stacking succeeded.
        self._can_vectorize_sliding = (
            self._k_fc_scales is not None and self._v_fc_scales is not None
            and self._k_base_bw is not None and self._v_base_bw is not None
            and all(getattr(m, "scaling_type", None) == "factored"
                    for _, m in k_entries + v_entries)
        )

    def _get_layer_name_prefix(self) -> str:
        """Resolve the actual prefix used by named_modules() for cache lookups (e.g. 'model' or 'model.model')."""
        if self._layer_name_prefix is not None:
            return self._layer_name_prefix
        for name, module in self.model.named_modules():
            if not isinstance(module, HeadwiseLowRankModule):
                continue
            if "layers." in name and "self_attn" in name and "v_proj" in name:
                # e.g. "model.model.layers.0.self_attn.inner.v_proj" -> prefix "model.model"
                # e.g. "layers.0.self_attn.inner.v_proj" -> prefix ""
                idx = name.find("layers.")
                self._layer_name_prefix = name[:idx].rstrip(".") if idx >= 0 else "model"
                return self._layer_name_prefix
        self._layer_name_prefix = "model"
        return self._layer_name_prefix

    def _run_sliding_step_vectorized(self, start: int, end: int, seq_len: int) -> bool:
        """
        Vectorized sliding step: batches all-layer dequant+requant into 2 GPU calls (K, V).

        Instead of a Python loop that launches 64 small GPU kernels sequentially, we:
          1. Stack chunks from all 32 K layers → [32, B, S, D]; same for V.
          2. One batched dequant, one batched requant per projection type.
          3. Write-back loop (dict writes only, no GPU ops).

        Only works for 'factored' scaling with uniform num_dims across layers.
        Returns True on success, False to signal fallback to the per-layer loop.
        """
        if not getattr(self, "_can_vectorize_sliding", False):
            return False

        chunk_len = end - start

        def _process(entries, fc_scales_cpu, base_bw_cpu, proj_type):
            # ── Phase 1: collect (read-only; abort if any layer is missing data) ──────
            full_qs, full_scales, old_bws = [], [], []
            layer_names = []
            device = None
            for layer_name, _ in entries:
                cached_q   = self.quantized_cache.get_quantized(layer_name, proj_type)
                cached_sc  = self.quantized_cache.get_scales(layer_name, proj_type)
                old_bw     = self.quantized_cache.get_bitwidths(layer_name, proj_type)
                if cached_q is None or cached_sc is None or old_bw is None:
                    return False
                if not cached_q.is_floating_point():
                    cached_q = cached_q.float()
                # Normalise scales to [B, T, 1] (factored has 1 group per token)
                if cached_sc.dim() == 2:
                    cached_sc = cached_sc.unsqueeze(-1)
                if cached_sc.dim() != 3 or cached_sc.shape[1] < seq_len or old_bw.shape[0] < seq_len:
                    return False
                if device is None:
                    device = cached_q.device
                full_qs.append(cached_q)
                full_scales.append(cached_sc)   # keep full tensor; chunk slice in Phase 2
                old_bws.append(old_bw)          # keep full tensor; chunk slice in Phase 2
                layer_names.append(layer_name)

            if not layer_names:
                return True

            L = len(layer_names)
            # ── Phase 2: batched dequant → requant ──────────────────────────────────
            chunks     = torch.stack([fq[:, start:end, :] for fq in full_qs])  # [L,B,S,D]
            old_sc_ch  = torch.stack([fs[:, start:end, :] for fs in full_scales])  # [L,B,S,1]
            old_bw_ch  = torch.stack([bw[start:end, :]   for bw in old_bws])   # [L,S,D]
            fc   = fc_scales_cpu[:L].to(device).float()     # [L,D]
            bbase = base_bw_cpu[:L].to(device).float()      # [L,D]

            fc_e  = fc[:, None, None, :]                    # [L,1,1,D]
            alpha = old_sc_ch[:, :, :, 0:1]                 # [L,B,S,1]
            # Use the stored bitwidth for dequant — per-dim recent tokens are
            # genuinely quantized at 16 bits (large q values), so clamping to
            # base_bw here would multiply large q by a huge stepsize and produce
            # astronomically wrong latents (the "protection marker" clamp was
            # added for the uniform path where tokens ARE stored at base_bw
            # precision but labeled 16; it is wrong for per-dim).
            bw_e  = old_bw_ch[:, None, :, :].float()        # [L,1,S,D]
            stepsize = 2.0 * fc_e * alpha / (2.0 ** bw_e)
            latents  = chunks.float() * stepsize             # [L,B,S,D]
            if not torch.isfinite(latents).all():
                return False

            base_e    = bbase[:, None, None, :]              # [L,1,1,D]
            lat_norm  = latents / fc_e.clamp(min=1e-8)
            new_alpha = lat_norm.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [L,B,S,1]
            new_step  = 2.0 * fc_e * new_alpha / (2.0 ** base_e)
            new_q     = (latents / new_step.clamp(min=1e-12)).round()
            new_q     = new_q.clamp(-2.0 ** (base_e - 1), 2.0 ** (base_e - 1) - 1)
            new_sc    = new_alpha.squeeze(-1).clamp(min=1e-6, max=1e4)  # [L,B,S]
            if not torch.isfinite(new_q).all() or not torch.isfinite(new_sc).all():
                return False

            # ── Phase 3: write-back (Python loop but no GPU ops) ────────────────────
            for i, layer_name in enumerate(layer_names):
                fq = full_qs[i]           # [B, T, D] float, in-place target
                fs = full_scales[i]       # [B, T, 1]
                new_bw = old_bws[i].clone()  # [T, D] — only clone bitwidths, not full Q
                new_bw[start:end, :] = bbase[i].unsqueeze(0).expand(chunk_len, -1).to(new_bw.dtype)

                fq[:, start:end, :] = new_q[i].to(fq.dtype)
                fs[:, start:end, 0] = new_sc[i].to(fs.dtype)

                n_bits = int(new_bw.max().item()) if new_bw.numel() else 8
                stored = _to_cache_quant_dtype(fq.detach().float(), n_bits)
                self.quantized_cache.quantized_latents[layer_name][proj_type] = stored
                self.quantized_cache.update_bitwidths(layer_name, proj_type, new_bw)
                self.quantized_cache.scales[layer_name][proj_type] = fs

                alt = QuantizedCache._alt_cache_key(layer_name)
                if alt and alt != layer_name:
                    self.quantized_cache.quantized_latents.setdefault(alt, {})[proj_type] = stored
                    self.quantized_cache.scales.setdefault(alt, {})[proj_type] = fs
                    if layer_name in self.quantized_cache.seq_lengths:
                        self.quantized_cache.seq_lengths[alt] = self.quantized_cache.seq_lengths[layer_name]
            return True

        # Process K and V; if either fails, return False → fall back to per-layer loop.
        if not _process(self._sliding_k_entries, self._k_fc_scales, self._k_base_bw, "k"):
            return False
        if not _process(self._sliding_v_entries, self._v_fc_scales, self._v_base_bw, "v"):
            return False
        return True

    def _maybe_run_sliding_step_before_append(self) -> None:
        """
        Run the sliding requantize step when the recent window is full (n_recent_full == recent_tokens)
        and we have at least (sink + recent_tokens) tokens, *before* we append the next token.

        Flow (e.g. sink=8, recent_tokens=16, sliding_step_size=8 [default half]):
        1. Everything stays at 16 bits until we have 8+16=24 tokens.
        2. At 24 tokens the recent window (last 16) is full → quantize [8..15] to base; n_recent_full=8. Append → 25.
        3. As we append, n_recent_full grows 9, 10, … 16. At 32 tokens n_recent_full=16 again → quantize [16..23]
           to base; n_recent_full=8. Repeat every time the recent window fills.

        With sliding_step_size=4 the window stays tighter (12–16 instead of 8–16):
        2. At 24 tokens → quantize [8..11] to base; n_recent_full=12. Append → 25.
        3. Grows 13 … 16. At 28 tokens → quantize [12..15] to base; n_recent_full=12. Repeat.

        Call this at the start of the first attention layer's forward (layer 0) in decode.
        """
        recent_tokens = self.quantized_cache.get_recent_tokens()
        if recent_tokens <= 0:
            return
        sink = self.attention_sink_tokens
        target_len = sink + recent_tokens
        n_recent_full = self.quantized_cache.get_n_recent_full()
        if n_recent_full != recent_tokens:
            return
        seq_len = 0
        for layer_name in getattr(self.quantized_cache, "quantized_latents", {}) or {}:
            for proj_type in ("k", "v"):
                q = self.quantized_cache.get_quantized(layer_name, proj_type)
                if q is not None:
                    l = q.shape[1]
                    seq_len = l if seq_len == 0 else min(seq_len, l)
        if seq_len < target_len:
            return
        self._run_sliding_recent_quantize_step()

    def _run_sliding_recent_quantize_step(self) -> None:
        """
        When the recent window is full (n_recent_full == recent_tokens), quantize the oldest
        sliding_step_size tokens in the window to base bitwidths, then set
        n_recent_full = recent_tokens - sliding_step_size.

        Example with recent_tokens=128, sliding_step_size=16:
          window oscillates 112–128; only 16 tokens are requantized per step.
        Example with recent_tokens=16, sliding_step_size=8 (default half):
          window oscillates 8–16; 8 tokens requantized per step.
        """
        recent_tokens = self.quantized_cache.get_recent_tokens()
        if recent_tokens <= 0:
            return
        sink = getattr(self, "attention_sink_tokens", 0)
        target_len = sink + recent_tokens
        n_recent_full = self.quantized_cache.get_n_recent_full()
        if n_recent_full != recent_tokens:
            return
        # Get seq_len as the minimum across all layers.
        # _maybe_run_sliding_step_before_append fires inside PaluQwen2Attention for
        # layer 0, after layer 0's k/v projections have already written the current
        # token to QuantizedCache.  Layer 0 therefore has N+1 tokens while layers
        # 1-31 still have N.  Taking the minimum avoids an off-by-one that would
        # cause the per-layer loop to index out of bounds on the shorter tensors.
        seq_len = 0
        for layer_name in getattr(self.quantized_cache, "quantized_latents", {}) or {}:
            for proj_type in ("k", "v"):
                q = self.quantized_cache.get_quantized(layer_name, proj_type)
                if q is not None:
                    l = q.shape[1]
                    seq_len = l if seq_len == 0 else min(seq_len, l)
        # Run when we have at least target_len tokens (e.g. 24, 32, 40, …). Prevents running from
        # the sliding step with the wrong length (e.g. 16 -> chunk [0:8] trashing sink).
        if seq_len < target_len:
            return
        if seq_len < recent_tokens:
            return
        step_size = min(self.sliding_step_size, recent_tokens)
        start = seq_len - recent_tokens
        end = start + step_size
        if end <= start:
            return
        chunk_len = end - start

        # Fast path: vectorize all-layer dequant+requant into 2 GPU calls instead of 64.
        # Falls back to the per-layer loop below if any layer is incompatible.
        if self._run_sliding_step_vectorized(start, end, seq_len):
            if _ADAPTIVE_DEBUG:
                logger.info(f"[SLIDING_VECTORIZED] step [{start}:{end}] seq_len={seq_len} step_size={step_size}")
            self.quantized_cache.set_n_recent_full(recent_tokens - step_size)
            return

        # Per-layer fallback loop (handles non-factored scaling, shape mismatches, etc.)
        # De-duplicate keys: QuantizedCache mirrors data under both the pre-wrap name
        # (e.g. model.layers.0.self_attn.k_proj) and the post-wrap alias
        # (model.layers.0.self_attn.inner.k_proj).  Iterating over all keys would
        # process each physical layer TWICE, causing an extra quantize→dequantize→
        # requantize cycle that compounds noise and garbles output.  Canonicalize
        # to the non-.inner. form so each layer is visited exactly once.
        _seen_canonical = set()
        for layer_name in list(getattr(self.quantized_cache, "quantized_latents", {}) or {}):
            if "self_attn" not in layer_name:
                continue
            if "v_proj" in layer_name:
                proj_type = "v"
            elif "k_proj" in layer_name:
                proj_type = "k"
            else:
                continue
            canonical = layer_name.replace(".self_attn.inner.", ".self_attn.", 1)
            dedup_key = (canonical, proj_type)
            if dedup_key in _seen_canonical:
                continue
            _seen_canonical.add(dedup_key)
            cached_q = self.quantized_cache.get_quantized(layer_name, proj_type)
            old_bw = self.quantized_cache.get_bitwidths(layer_name, proj_type)
            if cached_q is None or old_bw is None:
                continue
            module = self._module_cache.get(layer_name)
            if module is None:
                continue
            if not cached_q.is_floating_point():
                cached_q = cached_q.float()
            batch_size, _, num_dims = cached_q.shape
            actual_q_len = cached_q.shape[1]
            # Skip layers that don't have enough tokens to cover the chunk.
            if actual_q_len < end:
                continue
            # Clamp old_bw to actual_q_len; mismatches arise when layer 0 is one token
            # ahead and the vectorized path previously stored a sliced bw tensor.
            if old_bw.shape[0] > actual_q_len:
                old_bw = old_bw[:actual_q_len, :]
            elif old_bw.shape[0] < actual_q_len:
                continue  # bitwidths shorter than q — something went wrong; skip
            cached_scales = self.quantized_cache.get_scales(layer_name, proj_type)
            cached_scales_dict = self.quantized_cache.get_scales_dict(layer_name, proj_type)
            scales_tensor = None
            if cached_scales is not None and cached_scales.numel() >= batch_size * actual_q_len:
                if cached_scales.dim() == 2:
                    scales_tensor = cached_scales[:batch_size, :actual_q_len].to(cached_q.device)
                elif cached_scales.dim() == 3:
                    scales_tensor = cached_scales[:, :actual_q_len, :].to(cached_q.device)
            if scales_tensor is None and cached_scales_dict:
                first_grp = next(iter(cached_scales_dict.values()), None)
                if first_grp is not None:
                    sc, _ = first_grp
                    if sc.numel() >= batch_size * actual_q_len:
                        scales_tensor = sc.view(batch_size, -1)[:batch_size, :actual_q_len].to(cached_q.device)
            if scales_tensor is None:
                scales_tensor = torch.ones(batch_size, actual_q_len, device=cached_q.device, dtype=cached_q.dtype)
            new_bitwidths = old_bw.clone()
            if getattr(module, "base_bitwidths", None) is not None and module.base_bitwidths.numel() == num_dims:
                base_expanded = module.base_bitwidths.unsqueeze(0).expand(chunk_len, -1).to(
                    device=new_bitwidths.device, dtype=new_bitwidths.dtype
                )
            else:
                n_bits = 8
                if getattr(module, "latent_quantizer", None) is not None and hasattr(module.latent_quantizer, "n_bits"):
                    n_bits = module.latent_quantizer.n_bits
                base_expanded = torch.full((chunk_len, num_dims), n_bits, dtype=new_bitwidths.dtype, device=new_bitwidths.device)
            new_bitwidths[start:end, :] = base_expanded
            uniform_skip_scale_correction = False
            try:
                # Dequantize the chunk then re-quantize to bitwidth file (base_bitwidths) so values
                # and scales match; reader uses standard dequantize_with_per_dim_bitwidths.
                use_requant = (
                    cached_scales is not None
                    and hasattr(module, "dequantize_with_per_dim_bitwidths")
                    and hasattr(module, "quantize_with_per_dim_bitwidths")
                    and scales_tensor.shape[1] == actual_q_len
                )
                if use_requant:
                    cached_q_chunk = cached_q[:, start:end, :].contiguous()
                    scales_chunk = scales_tensor[:, start:end, :].contiguous()
                    old_bw_chunk = old_bw[start:end, :].contiguous()
                    # Do NOT clamp old_bw_chunk to base_bw here.  Per-dim recent tokens
                    # are genuinely quantized at 16 bits (large q values); clamping to
                    # base_bw multiplies those large integers by a huge stepsize (2α/2^2
                    # instead of 2α/2^16) producing ~32000× inflated latents that corrupt
                    # the cache.  Use the stored bw as-is for correct dequant.
                    # Avoid NaN/Inf: clamp scales to finite positive so dequant doesn't blow up
                    scales_chunk_safe = scales_chunk.clamp(min=1e-8, max=1e4)
                    latent_chunk = module.dequantize_with_per_dim_bitwidths(
                        cached_q_chunk, scales_chunk_safe, bitwidths_per_token=old_bw_chunk
                    )
                    if not torch.isfinite(latent_chunk).all():
                        use_requant = False
                    if use_requant:
                        new_q_chunk, new_scales_chunk = module.quantize_with_per_dim_bitwidths(
                            latent_chunk, bitwidths_per_token=new_bitwidths[start:end, :]
                        )
                        if not torch.isfinite(new_q_chunk).all() or not torch.isfinite(new_scales_chunk).all():
                            use_requant = False
                if use_requant:
                    # Use the scale that matches the requantized Q (do NOT max with original scale or magnitude is wrong)
                    min_scale = 1e-6
                    new_scales_chunk = new_scales_chunk.to(scales_chunk.dtype).clamp(min=min_scale, max=1e4)
                    new_scales_chunk = torch.where(
                        torch.isfinite(new_scales_chunk), new_scales_chunk, scales_chunk.clamp(min=min_scale)
                    )
                    new_q_chunk = torch.where(torch.isfinite(new_q_chunk), new_q_chunk, torch.zeros_like(new_q_chunk))
                    # In-place update — no full-tensor clone needed; _to_cache_quant_dtype
                    # creates a new storage tensor anyway for the final write-back.
                    cached_q[:, start:end, :] = new_q_chunk.to(cached_q.dtype)
                    scales_tensor[:, start:end, :] = new_scales_chunk.to(scales_tensor.dtype)
                    truncated_q, truncated_scales = cached_q, scales_tensor
                    # Debug: compare latent vs requant magnitude (layer 0 only, once per proj) to trace garbled output
                    if "layers.0" in layer_name and "self_attn" in layer_name and proj_type in ("k", "v"):
                        key = ("sliding_requant", layer_name, proj_type)
                        if key not in _sliding_requant_debug_logged:
                            _sliding_requant_debug_logged.add(key)
                            l0 = latent_chunk[0, 0, :].float()
                            l1 = latent_chunk[0, -1, :].float()
                            q0 = new_q_chunk[0, 0, :].float()
                            sc0 = new_scales_chunk[0, 0, :].float()
                            logger.info(
                                f"[SLIDING_DEBUG] Layer 0 {proj_type} requant chunk [{start}:{end}]: "
                                f"latent tok0 min={l0.min().item():.4f} max={l0.max().item():.4f} mean={l0.mean().item():.4f} | "
                                f"latent tok{-1} min={l1.min().item():.4f} max={l1.max().item():.4f} | "
                                f"Q tok0 min={q0.min().item():.1f} max={q0.max().item():.1f} | "
                                f"scale tok0 min={sc0.min().item():.6f} max={sc0.max().item():.6f} mean={sc0.mean().item():.6f}"
                            )
                else:
                    # For the uniform path, tokens are quantized at n_bits (e.g. 4) but stored
                    # with bitwidths=16 as a protection marker for the recent window. If we
                    # right-shift a 4-bit integer by (16-4)=12 it becomes 0, zeroing the cache
                    # and garbling output. Clamp old_bw to n_bits_actual so bit_reduction=0
                    # (integers are already at the target precision; only the label changes).
                    n_bits_actual = None
                    if hasattr(module, "latent_quantizer") and module.latent_quantizer is not None:
                        n_bits_actual = module.latent_quantizer.n_bits
                    uniform_skip_scale_correction = False
                    old_bw_for_trunc = old_bw
                    if (n_bits_actual is not None
                            and int(old_bw[start:end].max().item()) > n_bits_actual):
                        old_bw_for_trunc = old_bw.clone()
                        old_bw_for_trunc[start:end] = n_bits_actual
                        uniform_skip_scale_correction = True
                    truncated_q, truncated_scales = module.truncate_quantized_values(
                        cached_q, scales_tensor, old_bw_for_trunc, new_bitwidths
                    )
                    if "layers.0" in layer_name and "self_attn" in layer_name and proj_type in ("k", "v"):
                        key = ("sliding_truncate", layer_name, proj_type)
                        if key not in _sliding_requant_debug_logged:
                            _sliding_requant_debug_logged.add(key)
                            logger.info(
                                f"[SLIDING_DEBUG] Layer 0 {proj_type} {'no-op (protection marker→base)' if uniform_skip_scale_correction else 'TRUNCATION'} for chunk [{start}:{end}]"
                            )
            except Exception as e:
                logger.warning(f"Sliding recent truncate failed for {layer_name}[{proj_type}]: {e}")
                continue
            n_bits_storage = int(new_bitwidths.max().item()) if new_bitwidths.numel() else 8
            truncated_q_stored = _to_cache_quant_dtype(truncated_q.detach().float(), n_bits_storage)
            self.quantized_cache.quantized_latents[layer_name][proj_type] = truncated_q_stored
            self.quantized_cache.update_bitwidths(layer_name, proj_type, new_bitwidths)
            if cached_scales is not None:
                self.quantized_cache.scales[layer_name][proj_type] = truncated_scales
            # Mirror to alt key so reader using .inner. vs no .inner. sees same data (per-dim path)
            alt = QuantizedCache._alt_cache_key(layer_name)
            if alt and alt != layer_name:
                if alt not in self.quantized_cache.quantized_latents:
                    self.quantized_cache.quantized_latents[alt] = {}
                self.quantized_cache.quantized_latents[alt][proj_type] = truncated_q_stored
                if cached_scales is not None:
                    if alt not in self.quantized_cache.scales:
                        self.quantized_cache.scales[alt] = {}
                    self.quantized_cache.scales[alt][proj_type] = truncated_scales
                # Keep seq_lengths in sync so get_seq_length(alt) returns correct length
                if layer_name in self.quantized_cache.seq_lengths:
                    self.quantized_cache.seq_lengths[alt] = self.quantized_cache.seq_lengths[layer_name]
            # When cache uses scales_dict (uniform path), scale was for original bitwidth; after truncation
            # we must scale up so (Q_trunc + base) * scale_new = correct magnitude. Multiply scale by 2^bit_reduction
            # for positions [start:end] so dequantize_latent_from_integers gives correct values.
            # Skip when uniform_skip_scale_correction: integers were already at n_bits precision
            # (old_bw was a protection marker, not actual precision), so no scale adjustment is needed.
            if cached_scales is None and cached_scales_dict and batch_size > 0 and not uniform_skip_scale_correction:
                bit_red = int(round((old_bw[start:end, :].float().mean() - new_bitwidths[start:end, :].float().mean()).item()))
                bit_red = max(0, min(bit_red, 16))
                if bit_red > 0:
                    mult = float(2 ** bit_red)
                    flat_start = batch_size * start
                    flat_end = batch_size * end
                    updated_dict = {}
                    for group_idx, (sc, base) in cached_scales_dict.items():
                        if sc is None or sc.numel() < flat_end:
                            updated_dict[group_idx] = (sc, base)
                            continue
                        sc = sc.to(cached_q.device).float().clone()
                        if sc.dim() == 2:
                            sc[flat_start:flat_end, :] *= mult
                        else:
                            sc_flat = sc.view(-1, 1)
                            if sc_flat.shape[0] >= flat_end:
                                sc_flat[flat_start:flat_end, :] *= mult
                        updated_dict[group_idx] = (sc, base)
                    self.quantized_cache.scales_dict[layer_name][proj_type] = updated_dict
                    alt = QuantizedCache._alt_cache_key(layer_name)
                    if alt and alt != layer_name:
                        if alt not in self.quantized_cache.scales_dict:
                            self.quantized_cache.scales_dict[alt] = {}
                        self.quantized_cache.scales_dict[alt][proj_type] = updated_dict
        self.quantized_cache.set_n_recent_full(recent_tokens - step_size)

    def reset_cache(self):
        """Reset the quantized cache between samples
        (keeps hooks/patches in place so the manager can be reused across prompts)."""
        self.quantized_cache = QuantizedCache()
        self.quantized_cache.recent_tokens = self.recent_tokens
        self.quantized_cache.attention_sink_tokens = self.attention_sink_tokens
        self.past_key_values = None

    def remove_hooks(self):
        """Remove all wrappers/hooks and restore original forward methods and attention modules."""
        # Log summary of which layers were called - only show summary to reduce noise
        if hasattr(Qwen2AdaptiveQuantizationManager, '_call_count') and Qwen2AdaptiveQuantizationManager._call_count:
            total_calls = sum(Qwen2AdaptiveQuantizationManager._call_count.values())
            logger.info(f"Summary: {len(Qwen2AdaptiveQuantizationManager._call_count)} unique layers called, {total_calls} total calls")
            # Only show first few layers
            for i, (cache_key, count) in enumerate(sorted(Qwen2AdaptiveQuantizationManager._call_count.items())):
                if i < 3:
                    logger.info(f"  {cache_key}: {count} calls")
                elif i == 3:
                    logger.info(f"  ... (and {len(Qwen2AdaptiveQuantizationManager._call_count) - 3} more layers)")
                    break
            # Reset for next run
            Qwen2AdaptiveQuantizationManager._call_count = {}
            Qwen2AdaptiveQuantizationManager._called_layers = set()
        
        # Unwrap attention modules back to the original implementations
        for layer, original_attention in self.wrapped_attentions:
            try:
                if isinstance(layer.self_attn, PaluQwen2Attention):
                    layer.self_attn = original_attention
            except Exception as e:
                logger.warning(f"[REMOVE HOOKS] Failed to restore original attention for layer: {e}")
        self.wrapped_attentions.clear()
        
        # Restore original forward methods
        # Since we now store by name (string), we need to look up modules by name
        for name, original_forward_func in self.original_forwards.items():
            # Find the module by name
            module = None
            for n, m in self.model.named_modules():
                if n == name:
                    module = m
                    break
            
            if module is not None:
                import types
                is_closure = (isinstance(original_forward_func, types.FunctionType)
                              and getattr(original_forward_func, '__closure__', None))
                if (isinstance(original_forward_func, types.FunctionType)
                        and not is_closure
                        and not isinstance(original_forward_func, types.MethodType)):
                    # Plain unbound function with no closure — needs explicit binding
                    module.forward = types.MethodType(original_forward_func, module)
                else:
                    # Bound method or closure — already carries its module reference
                    module.forward = original_forward_func
            else:
                logger.warning(f"[REMOVE HOOKS] Could not find module {name} to restore original forward")
        
        self.original_forwards.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the quantized cache."""
        stats = {
            "num_layers": len(self.quantized_cache.quantized_latents),
            "total_memory_mb": 0,
            "layers": {}
        }
        
        for layer_name, proj_dict in self.quantized_cache.quantized_latents.items():
            layer_stats = {}
            for proj_type, quantized in proj_dict.items():
                if quantized is not None:
                    num_elements = quantized.numel()
                    # Estimate memory (assuming int8 for quantized values)
                    memory_bytes = num_elements * 1  # 1 byte per quantized value
                    memory_mb = memory_bytes / (1024 * 1024)
                    layer_stats[proj_type] = {
                        "shape": list(quantized.shape),
                        "memory_mb": memory_mb,
                        "num_elements": num_elements,
                    }
                    stats["total_memory_mb"] += memory_mb
            
            if layer_stats:
                stats["layers"][layer_name] = layer_stats
        
        return stats
