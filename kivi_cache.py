"""Shared KIVI KV cache implementation.

KIVI-style asymmetric min-max uniform quantization:
  Keys:   per-channel  — groups of group_size along the sequence dimension
  Values: per-token    — groups of group_size along the head_dim dimension

Recent residual_length tokens and the first num_sink_tokens are kept in FP16.
Chunks are quantized exactly once and never re-quantized.
"""

import torch
from transformers import DynamicCache

from kv_effective_bits import kivi_effective_bits, format_breakdown

# Print the effective-bits breakdown once per unique config (not per sample).
_EFF_BITS_LOGGED = set()


def _kivi_quantize_keys(x: torch.Tensor, bits: int, group_size: int):
    """x: [B, H, S, D] → (q_u8, scale_f16, zero_f16, orig_shape, seq_pad)"""
    B, H, S, D = x.shape
    seq_pad = (-S) % group_size
    if seq_pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_pad))
    S2 = S + seq_pad
    xp = x.float().permute(0, 1, 3, 2).reshape(B * H * D, S2 // group_size, group_size)
    mn = xp.amin(dim=-1, keepdim=True)
    mx = xp.amax(dim=-1, keepdim=True)
    scale = (mx - mn) / (2 ** bits - 1)
    q = ((xp - mn) / scale.clamp(min=1e-8)).round_().clamp_(0, 2 ** bits - 1).to(torch.uint8)
    return q, scale.to(torch.float16), mn.to(torch.float16), (B, H, S, D), seq_pad


def _kivi_dequantize_keys(q, scale, zero, orig_shape, seq_pad, dtype=torch.float16):
    B, H, S, D = orig_shape
    S2 = S + seq_pad
    xf = (q.float() * scale.float() + zero.float()).reshape(B, H, D, S2)
    return xf.permute(0, 1, 3, 2)[:, :, :S, :].to(dtype)


def _kivi_quantize_values(x: torch.Tensor, bits: int, group_size: int):
    """x: [B, H, S, D] → (q_u8, scale_f16, zero_f16, orig_shape, dim_pad)"""
    B, H, S, D = x.shape
    dim_pad = (-D) % group_size
    if dim_pad:
        x = torch.nn.functional.pad(x, (0, dim_pad))
    D2 = D + dim_pad
    xp = x.float().reshape(B * H * S, D2 // group_size, group_size)
    mn = xp.amin(dim=-1, keepdim=True)
    mx = xp.amax(dim=-1, keepdim=True)
    scale = (mx - mn) / (2 ** bits - 1)
    q = ((xp - mn) / scale.clamp(min=1e-8)).round_().clamp_(0, 2 ** bits - 1).to(torch.uint8)
    return q, scale.to(torch.float16), mn.to(torch.float16), (B, H, S, D), dim_pad


def _kivi_dequantize_values(q, scale, zero, orig_shape, dim_pad, dtype=torch.float16):
    B, H, S, D = orig_shape
    D2 = D + dim_pad
    xf = (q.float() * scale.float() + zero.float()).reshape(B, H, S, D2)
    return xf[:, :, :, :D].to(dtype)


class KIVIQuantizedCache(DynamicCache):
    """KIVI-style KV cache with asymmetric uniform quantization.

    Keys: per-channel (groups along sequence dim).
    Values: per-token (groups along head_dim).
    Sink tokens and the recent residual window are kept in FP16.
    """

    def __init__(self, nbits: int = 2, group_size: int = 32,
                 residual_length: int = 128, num_sink_tokens: int = 0):
        super().__init__()
        self.nbits = nbits
        self.group_size = group_size
        self.residual_length = residual_length
        self.num_sink_tokens = num_sink_tokens
        if residual_length > 0 and residual_length % group_size != 0:
            raise ValueError(
                f"residual_length ({residual_length}) must be a multiple of "
                f"group_size ({group_size}) for correct per-channel key quantization."
            )
        self._seen_tokens = 0
        self._sink_k: list = []
        self._sink_v: list = []
        self._chunks_k: list = []
        self._chunks_v: list = []
        self._res_k: list = []
        self._res_v: list = []
        self._dbg_prefill_logged = False
        self._dbg_flush_count = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            seq = key_states.shape[-2]
            if seq > 1 and not self._dbg_prefill_logged:
                self._dbg_prefill_logged = True
                n_sink = min(self.num_sink_tokens, seq)
                n_q = max(0, seq - n_sink - self.residual_length)
                n_res = seq - n_sink - n_q
                print(
                    f"  [KIVI-DBG] prefill: {seq} tokens — "
                    f"sink={n_sink} FP16, {n_q} quantized at {self.nbits}-bit, "
                    f"residual={n_res} FP16 (residual_length={self.residual_length})",
                    flush=True,
                )
                H, D = key_states.shape[1], key_states.shape[-1]
                # Dedup on the quantization config only (not seq) so this prints
                # once per run, not once per variable-length sample.
                cfg_key = (self.nbits, self.group_size, self.num_sink_tokens,
                           self.residual_length, H, D)
                if cfg_key not in _EFF_BITS_LOGGED:
                    _EFF_BITS_LOGGED.add(cfg_key)
                    bd = kivi_effective_bits(
                        self.nbits, seq, H, D,
                        group_size=self.group_size,
                        num_sink_tokens=self.num_sink_tokens,
                        residual_length=self.residual_length)
                    print("  " + format_breakdown("KIVI", bd, seq, H * D), flush=True)
            elif seq == 1 and not getattr(self, "_dbg_flush_logged", False):
                res = self._res_k[0] if self._res_k else None
                res_len = res.shape[-2] if res is not None else 0
                if res_len + 1 > self.residual_length and self.residual_length > 0:
                    self._dbg_flush_logged = True
                    print(
                        f"  [KIVI-DBG] residual full ({self.residual_length} tokens): "
                        f"flushing oldest {self.group_size} tokens → "
                        f"FP16 window stays {self.residual_length - self.group_size + 1}"
                        f"–{self.residual_length} tokens",
                        flush=True,
                    )

        if len(self._sink_k) == layer_idx:
            # Prefill
            S = key_states.shape[-2]
            n_sink = min(self.num_sink_tokens, S)
            self._sink_k.append(key_states[:, :, :n_sink, :].clone())
            self._sink_v.append(value_states[:, :, :n_sink, :].clone())

            rest_k = key_states[:, :, n_sink:, :]
            rest_v = value_states[:, :, n_sink:, :]
            n_quant = max(0, rest_k.shape[-2] - self.residual_length)

            cks, cvs = [], []
            if n_quant > 0:
                cks.append(_kivi_quantize_keys(
                    rest_k[:, :, :n_quant, :].contiguous(), self.nbits, self.group_size))
                cvs.append(_kivi_quantize_values(
                    rest_v[:, :, :n_quant, :].contiguous(), self.nbits, self.group_size))
            self._chunks_k.append(cks)
            self._chunks_v.append(cvs)

            res_k = rest_k[:, :, n_quant:, :]
            res_v = rest_v[:, :, n_quant:, :]
            self._res_k.append(res_k.clone() if res_k.shape[-2] > 0 else None)
            self._res_v.append(res_v.clone() if res_v.shape[-2] > 0 else None)
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return key_states, value_states

        # Decode
        parts_k, parts_v = [], []
        if self._sink_k[layer_idx].shape[-2] > 0:
            parts_k.append(self._sink_k[layer_idx])
            parts_v.append(self._sink_v[layer_idx])

        kv_dtype = key_states.dtype
        for ck, cv in zip(self._chunks_k[layer_idx], self._chunks_v[layer_idx]):
            parts_k.append(_kivi_dequantize_keys(*ck, dtype=kv_dtype))
            parts_v.append(_kivi_dequantize_values(*cv, dtype=kv_dtype))

        res_k = self._res_k[layer_idx]
        if res_k is not None:
            parts_k.append(res_k)
            parts_v.append(self._res_v[layer_idx])

        parts_k.append(key_states)
        parts_v.append(value_states)

        keys_out = torch.cat(parts_k, dim=-2)
        vals_out = torch.cat(parts_v, dim=-2)

        assert keys_out.shape[-2] == vals_out.shape[-2], \
            f"[KIVI] shape mismatch layer {layer_idx}: keys {keys_out.shape} vs vals {vals_out.shape}"

        # Keep parent DynamicCache lists in sync for direct list access.
        if len(self.key_cache) == layer_idx:
            self.key_cache.append(keys_out)
            self.value_cache.append(vals_out)
        else:
            self.key_cache[layer_idx] = keys_out
            self.value_cache[layer_idx] = vals_out

        # Append current token to residual; flush oldest group_size tokens when full.
        # Flushing in group_size chunks keeps per-channel key quantization correct
        # and maintains a large FP16 window (residual_length - group_size + 1 .. residual_length).
        new_k = torch.cat([res_k, key_states], dim=-2) if res_k is not None else key_states
        new_v = torch.cat([self._res_v[layer_idx], value_states], dim=-2) if res_k is not None else value_states
        if self.residual_length == 0:
            self._chunks_k[layer_idx].append(
                _kivi_quantize_keys(new_k.contiguous(), self.nbits, self.group_size))
            self._chunks_v[layer_idx].append(
                _kivi_quantize_values(new_v.contiguous(), self.nbits, self.group_size))
            self._res_k[layer_idx] = None
            self._res_v[layer_idx] = None
        elif new_k.shape[-2] > self.residual_length:
            self._chunks_k[layer_idx].append(
                _kivi_quantize_keys(new_k[:, :, :self.group_size, :].contiguous(), self.nbits, self.group_size))
            self._chunks_v[layer_idx].append(
                _kivi_quantize_values(new_v[:, :, :self.group_size, :].contiguous(), self.nbits, self.group_size))
            self._res_k[layer_idx] = new_k[:, :, self.group_size:, :]
            self._res_v[layer_idx] = new_v[:, :, self.group_size:, :]
        else:
            self._res_k[layer_idx] = new_k
            self._res_v[layer_idx] = new_v

        return keys_out, vals_out
