"""Pure-PyTorch KVQuant KV cache (no CUDA kernel).

Faithful port of the simulated-quantization algorithm from
SqueezeAILab/KVQuant (`quant/kvquant/simquant_module_quantizer.py`),
adapted into a streaming HuggingFace ``DynamicCache`` so it can be dropped
into ``model.generate()`` exactly like ``KIVIQuantizedCache`` in
``kivi_cache.py``.

KVQuant pillars implemented here
--------------------------------
  * Per-channel Key quantization (one scale/zero-point + outlier threshold
    per (kv_head, head_dim) channel, calibrated offline and held fixed).
  * Per-token Value quantization (min/max computed dynamically per token).
  * Non-Uniform Quantization (NUQ): a learned look-up table of 2**nbits
    signposts in the normalized [-1, 1] range (offline Fisher-weighted
    k-means; see ``run_kvquant_calibration.py``).  Falls back to fixed
    NormalFloat (NF) signposts when no calibrated LUT is supplied.
  * Dense-and-Sparse decomposition: the top ``(1 - sparsity_threshold)``
    fraction of magnitude outliers are kept in fp16 and added back after
    dequantization.
  * Attention-sink awareness: the first ``num_sink_tokens`` tokens are kept
    in fp16, plus an optional recent fp16 residual window.

Pre-RoPE keys (faithful to the paper)
-------------------------------------
KVQuant quantizes Keys *pre-RoPE*.  HuggingFace applies RoPE before
``past_key_values.update()`` sees the keys, so we invert it inside the cache:
the incoming post-RoPE keys are un-rotated to the pre-RoPE frame (where the
per-channel outlier structure is clean and position-independent), quantized
there, and the reconstruction is re-rotated back to post-RoPE for attention.
We reuse the exact cos/sin the model used (passed via ``cache_kwargs``), so
Llama-3.1's custom RoPE scaling is handled automatically — no attention-forward
patching required.  Set ``pre_rope=False`` to quantize post-RoPE instead (for
ablation; must be paired with a post-RoPE-calibrated quantizer).  Calibration
must produce pre-RoPE statistics — see ``run_kvquant_calibration.py``.
"""

import math
import torch
from torch.distributions import Normal
from transformers import DynamicCache

from kv_effective_bits import kvquant_effective_bits, format_breakdown

# Print the effective-bits breakdown once per unique config (not per sample).
_EFF_BITS_LOGGED = set()


# ── NUQ signpost / look-up-table machinery ──────────────────────────────────

def build_nf_signposts(bits: int) -> torch.Tensor:
    """NormalFloat signposts in [-1, 1] (KVQuant ``nf_nuq`` fallback).

    Direct port of the NF construction in ``QuantLinearSim.__init__``.
    """
    dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    num_pos = (2 ** (bits - 1)) + 1
    num_neg = (2 ** (bits - 1))

    offsets = [0.5 * (1 / 32 + 1 / 30), 1 - 0.5 * (1 / 32 + 1 / 30)]

    list1 = [offsets[0]]
    spacing = (0.5 - offsets[0]) / (2 ** (bits - 1) - 1)
    add = offsets[0]
    for _ in range(num_neg - 1):
        add += spacing
        list1.append(add)

    list2 = []
    spacing = (offsets[1] - 0.5) / (2 ** (bits - 1))
    add = 0.5
    for _ in range(num_pos - 1):
        list2.append(add)
        add += spacing
    list2.append(offsets[-1])

    neg = [dist.icdf(torch.tensor([v])).item() for v in list1]
    pos = [dist.icdf(torch.tensor([v])).item() for v in list2]

    rangeval = abs(neg[0]) - abs(neg[-1])
    off = abs(neg[-1])
    neg = [(v + off) / rangeval for v in neg]

    rangeval = abs(pos[-1]) - abs(pos[0])
    off = abs(pos[0])
    pos = [(v - off) / rangeval for v in pos]
    del pos[0]

    signposts = neg + pos
    assert len(signposts) == 2 ** bits
    return torch.tensor(signposts, dtype=torch.float32)


def round_to_lut(x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Round every element of ``x`` to the nearest value in ``lut``.

    Vectorized equivalent of KVQuant's ``round_to_nearest_pole_sim`` using
    boundary bucketization (O(N log L) instead of O(N*L) memory).
    """
    sorted_lut, _ = torch.sort(lut.to(x.device, x.dtype))
    # midpoints between adjacent signposts are the decision boundaries
    boundaries = (sorted_lut[1:] + sorted_lut[:-1]) / 2
    idx = torch.bucketize(x, boundaries)
    return sorted_lut[idx]


# ── RoPE un-rotate / re-rotate (for pre-RoPE key quantization) ────────────────
#
# KVQuant quantizes Keys *pre-RoPE*.  HF applies RoPE before the cache sees the
# keys, so we invert it inside the cache: un-rotate the incoming post-RoPE keys
# to the pre-RoPE frame (where per-channel outlier structure is clean and stable
# across positions), quantize there, then re-rotate the reconstruction back to
# post-RoPE for attention.  We reuse the exact cos/sin the model used (passed in
# cache_kwargs), so Llama-3.1's custom RoPE scaling is handled automatically.

def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope(k, cos, sin):
    """k: [B, H, S, D]; cos/sin: [B, S, D].  Standard HF RoPE."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return k * cos + _rotate_half(k) * sin


def unapply_rope(k, cos, sin):
    """Inverse rotation (rotate by -theta): flips the sign of the sin term."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return k * cos - _rotate_half(k) * sin


# ── Core quantize-then-dequantize (returns reconstructed fp16) ───────────────
#
# Like KVQuant's QuantLinearSim.forward and the existing KIVI cache, this is a
# *simulation*: we reconstruct the dequantized tensor that attention actually
# consumes.  Attention output is identical whether we store the compact
# (indices + scales + sparse outliers) form and dequantize on read, or store
# the reconstructed fp16 directly — so we store the reconstruction.


def _nuq_recon(inp, offset, rangeval, lut, outlier_mask,
               normscale=None, normoffset=None):
    """NUQ reconstruction in the normalized frame (port of quant_fn_nuq_recon).

    ``inp``/``offset``/``rangeval`` already broadcast-compatible.  Outliers
    (where ``outlier_mask``) are preserved in fp16.  When ``normscale`` is given,
    applies KVQuant's Q-Norm (affine moment-matching) to the quantized values
    before un-normalizing.
    """
    centered = inp - offset
    outliers = centered * outlier_mask if outlier_mask is not None else None
    if outliers is not None:
        centered = centered - outliers

    normalized = centered / rangeval.clamp(min=1e-8)
    q = round_to_lut(normalized.flatten(), lut).reshape(inp.shape)
    if normscale is not None:                       # Q-Norm
        q = q * normscale + normoffset
    recon = q * rangeval

    if outliers is not None:
        recon = torch.where(outlier_mask, torch.zeros_like(recon), recon)
        recon = recon + outliers

    recon = recon + offset
    return torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)


def _cap_outlier_mask(kf, offset, rangeval, cap_frac):
    """Per-token fixed-count key outlier mask (KVQuant cap_outliers, keys only).

    Instead of a magnitude threshold (variable #outliers/token), keep the
    ``ceil(cap_frac * H*D)`` most extreme channels *per token* (by per-channel
    normalized magnitude).  kf/offset/rangeval shaped for [B,H,S,D].
    """
    B, H, S, D = kf.shape
    normed = ((kf - offset) / rangeval.clamp(min=1e-8)).abs()
    normed_t = normed.permute(0, 2, 1, 3).reshape(B, S, H * D)   # per token
    n_cap = max(1, math.ceil(cap_frac * H * D))
    n_cap = min(n_cap, H * D)
    idx = normed_t.topk(n_cap, dim=-1).indices
    mask_t = torch.zeros_like(normed_t, dtype=torch.bool)
    mask_t.scatter_(-1, idx, True)
    return mask_t.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()


def quantize_keys(k, layer_q, lut, include_sparse, dtype,
                  cap_outliers=-1.0, normscale=None, normoffset=None):
    """Per-channel key quantization with static calibrated stats.

    k: [B, H, S, D], in whatever frame the cache supplies (pre-RoPE by default).
    ``layer_q`` holds per-(H,D) ``thr_upper``/``thr_lower`` (calibrated outlier
    thresholds → range/zero-point).  Outliers (per channel) are isolated to fp16.
    ``cap_outliers`` > 0 switches to a fixed #outliers-per-token policy.
    Returns fp16 recon.
    """
    B, H, S, D = k.shape
    kf = k.float()

    thr_u = layer_q["k_thr_upper"].to(kf.device).view(1, H, 1, D)
    thr_l = layer_q["k_thr_lower"].to(kf.device).view(1, H, 1, D)
    offset = (thr_u + thr_l) / 2
    rangeval = (thr_u - thr_l) / 2

    outlier_mask = None
    if include_sparse:
        if cap_outliers and cap_outliers > 0:
            outlier_mask = _cap_outlier_mask(kf, offset, rangeval, cap_outliers)
        else:
            outlier_mask = (kf > thr_u) | (kf < thr_l)

    recon = _nuq_recon(kf, offset, rangeval, lut, outlier_mask, normscale, normoffset)
    return recon.to(dtype)


def quantize_values(v, lut, include_sparse, sparsity_threshold, dtype,
                    normscale=None, normoffset=None):
    """Per-token value quantization with dynamic (per-token) min/max.

    v: [B, H, S, D].  KVQuant pools the value features per *token* across all
    kv heads, so we compute stats over the flattened (H*D) feature vector.
    Returns fp16 recon.
    """
    B, H, S, D = v.shape
    vf = v.float().permute(0, 2, 1, 3).reshape(B, S, H * D)  # [B, S, H*D]

    outlier_mask = None
    if include_sparse:
        t = 1 - ((1 - sparsity_threshold) / 2)
        thr_u = torch.quantile(vf, t, dim=-1, keepdim=True)
        thr_l = torch.quantile(vf, 1 - t, dim=-1, keepdim=True)
        outlier_mask = (vf >= thr_u) | (vf <= thr_l)
        outliers = vf * outlier_mask
        median = torch.median(vf, dim=-1, keepdim=True).values
        tmp = vf - outliers + median * outlier_mask  # recenter to avoid skew
        maxval = tmp.amax(dim=-1, keepdim=True)
        minval = tmp.amin(dim=-1, keepdim=True)
    else:
        maxval = vf.amax(dim=-1, keepdim=True)
        minval = vf.amin(dim=-1, keepdim=True)

    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2

    recon = _nuq_recon(vf, offset, rangeval, lut, outlier_mask, normscale, normoffset)
    recon = recon.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()
    return recon.to(dtype)


# ── Quantizer file loading (cached) ──────────────────────────────────────────

_QUANTIZER_CACHE = {}


def load_quantizer(path):
    """Load a calibrated quantizer ``.pt`` (from run_kvquant_calibration.py).

    Returns None for a falsy path (→ NormalFloat fallback).  Cached so the
    same file is only read from disk once per process.
    """
    if not path:
        return None
    q = _QUANTIZER_CACHE.get(path)
    if q is None:
        q = torch.load(path, map_location="cpu")
        _QUANTIZER_CACHE[path] = q
    return q


# ── Streaming cache ──────────────────────────────────────────────────────────

class KVQuantizedCache(DynamicCache):
    """KVQuant-style KV cache (pure PyTorch, no CUDA kernel).

    Same plug-in contract as ``KIVIQuantizedCache``: construct one per sample
    and pass as ``past_key_values`` to ``model.generate()``.

    Parameters
    ----------
    nbits : quantization bits (2-5; LUT must have 2**nbits entries).
    quantizer : dict produced by ``run_kvquant_calibration.py`` (or None to
        fall back to NormalFloat signposts + dynamic key thresholds).
    include_sparse : enable dense-and-sparse outlier handling.
    sparsity_threshold : keep top/bottom ((1-thr)/2) fraction as fp16 outliers
        (default 0.99 → ~1% of values kept in fp16).
    num_sink_tokens : keep first N tokens in fp16 (attention sink).
    residual_length : keep most recent N tokens in fp16 (0 = quantize all).
    group_size : flush granularity during decode (must divide residual_length).
    pre_rope : quantize keys in the pre-RoPE frame (faithful KVQuant, default).
        Requires a pre-RoPE-calibrated quantizer and cos/sin in cache_kwargs.
    cap_outliers : keys only; >0 keeps a fixed fraction of outliers per token
        (KVQuant cap_outliers) instead of threshold-based dense-and-sparse.

    Q-Norm (per-layer affine moment correction) is applied automatically when
    the calibrated quantizer carries ``k_normscale``/``v_normscale`` (produced by
    ``run_kvquant_calibration.py --norm``).
    """

    def __init__(self, nbits=4, quantizer=None, include_sparse=True,
                 sparsity_threshold=0.99, num_sink_tokens=5,
                 residual_length=0, group_size=32, pre_rope=True,
                 cap_outliers=-1.0):
        super().__init__()
        self.nbits = nbits
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.num_sink_tokens = num_sink_tokens
        self.residual_length = residual_length
        self.group_size = max(1, group_size)
        self.pre_rope = pre_rope
        # cap_outliers > 0: keep a fixed fraction of outliers per token for keys
        # (KVQuant's cap_outliers; -1 = threshold-based dense-and-sparse).
        self.cap_outliers = cap_outliers
        if residual_length > 0 and residual_length % self.group_size != 0:
            raise ValueError(
                f"residual_length ({residual_length}) must be a multiple of "
                f"group_size ({self.group_size}).")

        self._quantizer = quantizer or {}
        # Fallback NF signposts (used when a layer/proj has no calibrated LUT)
        self._nf_lut = build_nf_signposts(nbits)

        # Warn if the calibrated quantizer was produced in a different RoPE frame
        # than the cache will run in (pre-RoPE vs post-RoPE quantizers are not
        # interchangeable — using the wrong one yields garbage).
        cfg = self._quantizer.get("config")
        if cfg is not None and "pre_rope" in cfg and cfg["pre_rope"] != self.pre_rope:
            print(f"  [KVQUANT-WARN] quantizer was calibrated with pre_rope="
                  f"{cfg['pre_rope']} but cache pre_rope={self.pre_rope}. "
                  f"Re-calibrate to match, or set the flag accordingly.", flush=True)

        self._seen_tokens = 0
        # Per-layer token counts (like HF DynamicCache).  Needed because Qwen2's
        # old-style attention computes RoPE per layer and queries
        # get_seq_length(layer_idx) *before* that layer's update() — a global
        # counter bumped at layer 0 would make layers 1..N double-count within a
        # single prefill (kv_seq_len = cur + cur → cos twice as long → crash).
        self._layer_seen = {}
        self._sink_k, self._sink_v = [], []
        self._chunks_k, self._chunks_v = [], []
        self._res_k, self._res_v = [], []
        # Accumulated cos/sin (the model's, via cache_kwargs) for re-rotating the
        # reconstructed pre-RoPE keys back to post-RoPE.  Shared across layers
        # (RoPE depends only on position), so built once at layer 0.
        self._cos, self._sin = None, None
        self._rope_warned = False
        self._dbg_prefill_logged = False

    # -- LUT / quantizer lookup -------------------------------------------------
    def _lut(self, layer_idx, proj):
        layer_q = self._quantizer.get("layers", {}).get(layer_idx)
        if layer_q is not None and f"{proj}_lut" in layer_q:
            return layer_q[f"{proj}_lut"].float()
        return self._nf_lut

    def _layer_q(self, layer_idx):
        return self._quantizer.get("layers", {}).get(layer_idx)

    def _q_keys(self, k, layer_idx, dtype):
        layer_q = self._layer_q(layer_idx)
        if layer_q is not None and "k_thr_upper" in layer_q:
            ns, no = (layer_q.get("k_normscale"), layer_q.get("k_normoffset"))
            return quantize_keys(k, layer_q, self._lut(layer_idx, "k"),
                                 self.include_sparse, dtype,
                                 cap_outliers=self.cap_outliers,
                                 normscale=ns, normoffset=no)
        # No calibrated per-channel key stats: fall back to per-token dynamic
        # (same path as values) so the cache still functions uncalibrated.
        return quantize_values(k, self._lut(layer_idx, "k"),
                               self.include_sparse, self.sparsity_threshold, dtype)

    def _q_vals(self, v, layer_idx, dtype):
        layer_q = self._layer_q(layer_idx)
        ns = layer_q.get("v_normscale") if layer_q is not None else None
        no = layer_q.get("v_normoffset") if layer_q is not None else None
        return quantize_values(v, self._lut(layer_idx, "v"),
                               self.include_sparse, self.sparsity_threshold, dtype,
                               normscale=ns, normoffset=no)

    # -- DynamicCache interface -------------------------------------------------
    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Report tokens already stored for THIS layer (before the current step),
        # matching DynamicCache semantics: a not-yet-populated layer has 0 tokens.
        # Must NOT fall back to the global _seen_tokens — during prefill that is
        # already bumped by layer 0, so layers 1..N would double-count
        # (kv_seq_len = cur + cur → cos twice as long → crash).
        return self._layer_seen.get(layer_idx, 0)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Resolve cos/sin for pre-RoPE handling (the model's own, via cache_kwargs).
        cos = sin = None
        if self.pre_rope:
            if cache_kwargs is not None:
                cos, sin = cache_kwargs.get("cos"), cache_kwargs.get("sin")
            if cos is None or sin is None:
                if not self._rope_warned:
                    self._rope_warned = True
                    print("  [KVQUANT-WARN] pre_rope=True but cos/sin missing from "
                          "cache_kwargs — falling back to post-RoPE quantization "
                          "(transformers too old?).", flush=True)
                self.pre_rope = False
            else:
                # Normalize cos/sin to (batch, seq_len, head_dim) so that
                # (un)apply_rope's unsqueeze(1) → (batch, 1, seq_len, head_dim)
                # broadcasts against (batch, num_heads, seq_len, head_dim), and the
                # per-step accumulation cat(..., dim=1) grows the seq dim.  Old-style
                # Qwen2/Llama rotary_emb passes 2D (seq_len, dim) or 4D (1,1,seq,dim).
                if cos.dim() == 2:      # (seq_len, dim) → (1, seq_len, dim)
                    cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
                elif cos.dim() == 4:    # (1, 1, seq_len, dim) → (1, seq_len, dim)
                    cos, sin = cos.squeeze(1), sin.squeeze(1)

                # Old-style Qwen2/Llama rotary_emb(seq_len=kv_seq_len) returns cos/sin
                # for the FULL sequence (positions 0..kv_seq_len-1), but key_states here
                # is only the current step's q_len token(s).  Keep only the last q_len
                # positions — otherwise unapply_rope broadcasts key_states up to the full
                # length (k_in becomes kv_seq_len long → K/V length mismatch in attention),
                # and the layer-0 cat accumulation double-counts.  New-style position_embeddings
                # already have length q_len, so this is a no-op there.
                q_len = key_states.shape[-2]
                if cos.shape[1] > q_len:
                    cos, sin = cos[:, -q_len:, :], sin[:, -q_len:, :]

        # Advance this layer's token count for the next step's RoPE length query.
        self._layer_seen[layer_idx] = self._layer_seen.get(layer_idx, 0) + key_states.shape[-2]

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            seq = key_states.shape[-2]
            if self.pre_rope:  # accumulate cos/sin once per step (shared across layers)
                self._cos = cos if self._cos is None else torch.cat([self._cos, cos], dim=1)
                self._sin = sin if self._sin is None else torch.cat([self._sin, sin], dim=1)
            if seq > 1 and not self._dbg_prefill_logged:
                self._dbg_prefill_logged = True
                n_sink = min(self.num_sink_tokens, seq)
                n_q = max(0, seq - n_sink - self.residual_length)
                calib = "calibrated NUQ" if self._quantizer.get("layers") else "NF (uncalibrated)"
                frame = "pre-RoPE" if self.pre_rope else "post-RoPE"
                print(f"  [KVQUANT-DBG] prefill: {seq} tokens — sink={n_sink} FP16, "
                      f"{n_q} quantized at {self.nbits}-bit ({calib}, {frame} keys, "
                      f"sparse={self.include_sparse}), residual={seq - n_sink - n_q} FP16",
                      flush=True)
                H, D = key_states.shape[1], key_states.shape[-1]
                # Dedup on the quantization config only (not seq) so this prints
                # once per run, not once per variable-length sample.
                cfg_key = (self.nbits, self.include_sparse, round(self.sparsity_threshold, 4),
                           self.num_sink_tokens, self.residual_length, self.pre_rope, H, D)
                if cfg_key not in _EFF_BITS_LOGGED:
                    _EFF_BITS_LOGGED.add(cfg_key)
                    bd = kvquant_effective_bits(
                        self.nbits, seq, H, D,
                        include_sparse=self.include_sparse,
                        sparsity_threshold=self.sparsity_threshold,
                        num_sink_tokens=self.num_sink_tokens,
                        residual_length=self.residual_length)
                    print("  " + format_breakdown("KVQUANT", bd, seq, H * D), flush=True)

        dtype = key_states.dtype
        # Keys are stored/quantized in the pre-RoPE frame; values are unaffected
        # by RoPE so they pass through unchanged.
        k_in = unapply_rope(key_states, cos, sin) if self.pre_rope else key_states

        def _rerotate(keys_pre):
            """Re-apply RoPE to an assembled pre-RoPE key tensor (positions 0..L-1)."""
            if not self.pre_rope:
                return keys_pre
            L = keys_pre.shape[-2]
            return apply_rope(keys_pre, self._cos[:, :L, :], self._sin[:, :L, :])

        if len(self._sink_k) == layer_idx:
            # ---- Prefill ----
            S = key_states.shape[-2]
            n_sink = min(self.num_sink_tokens, S)
            self._sink_k.append(k_in[:, :, :n_sink, :].clone())
            self._sink_v.append(value_states[:, :, :n_sink, :].clone())

            rest_k = k_in[:, :, n_sink:, :]
            rest_v = value_states[:, :, n_sink:, :]
            n_quant = max(0, rest_k.shape[-2] - self.residual_length)

            cks, cvs = [], []
            if n_quant > 0:
                cks.append(self._q_keys(rest_k[:, :, :n_quant, :].contiguous(), layer_idx, dtype))
                cvs.append(self._q_vals(rest_v[:, :, :n_quant, :].contiguous(), layer_idx, dtype))
            self._chunks_k.append(cks)
            self._chunks_v.append(cvs)

            res_k = rest_k[:, :, n_quant:, :]
            res_v = rest_v[:, :, n_quant:, :]
            self._res_k.append(res_k.clone() if res_k.shape[-2] > 0 else None)
            self._res_v.append(res_v.clone() if res_v.shape[-2] > 0 else None)

            # Prefill attention uses the original full-precision K/V (quantization
            # only affects what is *stored* for later decode steps), matching KIVI.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return key_states, value_states

        # ---- Decode ----  (assemble pre-RoPE parts, then re-rotate keys)
        parts_k, parts_v = [], []
        if self._sink_k[layer_idx].shape[-2] > 0:
            parts_k.append(self._sink_k[layer_idx])
            parts_v.append(self._sink_v[layer_idx])
        parts_k.extend(self._chunks_k[layer_idx])
        parts_v.extend(self._chunks_v[layer_idx])
        if self._res_k[layer_idx] is not None:
            parts_k.append(self._res_k[layer_idx])
            parts_v.append(self._res_v[layer_idx])
        parts_k.append(k_in)
        parts_v.append(value_states)

        keys_out = _rerotate(torch.cat(parts_k, dim=-2))
        vals_out = torch.cat(parts_v, dim=-2)

        if len(self.key_cache) == layer_idx:
            self.key_cache.append(keys_out)
            self.value_cache.append(vals_out)
        else:
            self.key_cache[layer_idx] = keys_out
            self.value_cache[layer_idx] = vals_out

        # Append new (pre-RoPE) token to residual; flush oldest group_size tokens when full.
        res_k = self._res_k[layer_idx]
        new_k = torch.cat([res_k, k_in], dim=-2) if res_k is not None else k_in
        new_v = torch.cat([self._res_v[layer_idx], value_states], dim=-2) if res_k is not None else value_states

        if self.residual_length == 0:
            self._chunks_k[layer_idx].append(self._q_keys(new_k.contiguous(), layer_idx, dtype))
            self._chunks_v[layer_idx].append(self._q_vals(new_v.contiguous(), layer_idx, dtype))
            self._res_k[layer_idx] = None
            self._res_v[layer_idx] = None
        elif new_k.shape[-2] > self.residual_length:
            gs = self.group_size
            self._chunks_k[layer_idx].append(self._q_keys(new_k[:, :, :gs, :].contiguous(), layer_idx, dtype))
            self._chunks_v[layer_idx].append(self._q_vals(new_v[:, :, :gs, :].contiguous(), layer_idx, dtype))
            self._res_k[layer_idx] = new_k[:, :, gs:, :]
            self._res_v[layer_idx] = new_v[:, :, gs:, :]
        else:
            self._res_k[layer_idx] = new_k
            self._res_v[layer_idx] = new_v

        return keys_out, vals_out
