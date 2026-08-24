"""Effective bits-per-element accounting for KV-cache quantization methods.

Nominal bitwidth (e.g. "2-bit") understates real memory because methods store
extra fp16 metadata and keep some values in full precision.  This module makes
the *effective* bits/element explicit so KIVI and KVQuant can be compared at
matched memory rather than matched nominal bits.

Components modelled (all per stored KV element, averaged over keys+values):

  dense_index   nominal bits — the quantized index for every element.
  scale         fp16 scale/zero metadata, amortized over its sharing group:
                  KIVI    : per-group (group_size) for both K and V  → 2*16/group_size
                  KVQuant : per-token (V) over head_dim*num_kv_heads  → 2*16/F
                            (K uses a *static* calibrated per-channel scale that
                             ships in the quantizer file ≈ model metadata; by
                             default it is NOT counted as per-sequence cost.
                             Pass count_key_scale=True to amortize it over seq_len.)
  sparse        KVQuant dense-and-sparse: a (1 - sparsity_threshold) fraction of
                values kept in fp16 plus an index.  Cost = frac * (16 + index_bits),
                index_bits defaults to 32 (CSR/CSC int32 index, faithful to the
                KVQuant reference kernels; pass a smaller value for a tight pack).
  fp16_window   attention-sink + recent-residual tokens kept in fp16, amortized
                over the sequence: (n_sink + n_residual)/seq_len * 16.

Assumptions are deliberately simple and documented; a packed deployment kernel
may store metadata/indices tighter.  Numbers are estimates for comparison, not
exact on-disk sizes.
"""

import argparse
import math


def _fp16_window_bits(num_sink_tokens, residual_length, seq_len):
    if seq_len <= 0:
        return 0.0
    n_sink = min(num_sink_tokens, seq_len)
    n_res = min(residual_length, max(seq_len - n_sink, 0))
    return (n_sink + n_res) / seq_len * 16.0


def kvquant_effective_bits(nbits, seq_len, num_kv_heads, head_dim,
                           include_sparse=True, sparsity_threshold=0.99,
                           num_sink_tokens=5, residual_length=0,
                           count_key_scale=False, index_bits=None):
    """Effective bits/element for the KVQuant cache.  Returns a breakdown dict."""
    F = num_kv_heads * head_dim
    out = {}
    out["dense_index"] = float(nbits)
    # value: per-token scale + zero (offset+range), fp16 each, shared over F dims
    value_scale = 2 * 16.0 / F
    # key: static calibrated per-channel scale (ships in quantizer file).
    key_scale = (2 * 16.0 / seq_len) if (count_key_scale and seq_len > 0) else 0.0
    out["scale"] = (value_scale + key_scale) / 2.0  # averaged over K and V
    if include_sparse:
        frac = max(0.0, 1.0 - sparsity_threshold)  # top+bottom fraction kept fp16
        # KVQuant stores outliers in CSR/CSC with int32 indices (faithful to the
        # reference kernels), so default to 32-bit index, not a tight ceil(log2 F).
        idx = index_bits if index_bits is not None else 32.0
        out["sparse"] = frac * (16.0 + idx)
    else:
        out["sparse"] = 0.0
    out["fp16_window"] = _fp16_window_bits(num_sink_tokens, residual_length, seq_len)
    out["total"] = out["dense_index"] + out["scale"] + out["sparse"] + out["fp16_window"]
    return out


def kivi_effective_bits(nbits, seq_len, num_kv_heads, head_dim,
                        group_size=32, num_sink_tokens=0, residual_length=128):
    """Effective bits/element for the KIVI cache.  Returns a breakdown dict."""
    out = {}
    out["dense_index"] = float(nbits)
    # per-group scale + zero, fp16 each, shared over group_size elements (both K and V)
    out["scale"] = 2 * 16.0 / group_size
    out["sparse"] = 0.0
    out["fp16_window"] = _fp16_window_bits(num_sink_tokens, residual_length, seq_len)
    out["total"] = out["dense_index"] + out["scale"] + out["sparse"] + out["fp16_window"]
    return out


def palu_effective_bits(mean_quant_bits, latent_dim, num_kv_heads, head_dim,
                        num_groups, num_sink_tokens=0, residual_length=0,
                        seq_len=8192, symmetric=True, count_channel_scale=False):
    """Effective bits/element for the PALU per-dim / uniform / adaptive cache.

    PALU stores a low-rank latent of dimension ``latent_dim = sum(ranks)`` instead
    of the original KV dimension ``D_orig = num_kv_heads*head_dim``.  The quantized
    index cost is therefore amortized over the *original* elements via the rank
    ratio.

      ratio        = latent_dim / D_orig
      dense_index  = ratio * mean_quant_bits
      scale        = n_meta*16*num_groups / D_orig   (per group per token)
                       symmetric (factored/kvtc) : n_meta = 1  (only token alpha,
                                                   no zero-point)
                       asymmetric (tokenwise sym=False) : n_meta = 2 (scale + base)
      channel_scale= static per-dim fc_scale from calibration; counted only if
                     count_channel_scale=True, amortized over seq:
                       ratio * 16 / seq_len   (one fp16 per latent dim).
      fp16_window  = ratio * (sink + recent-residual tokens kept fp16) / seq * 16
                     (window keeps the latent z in fp16 → scaled by rank ratio).
    """
    D_orig = num_kv_heads * head_dim
    ratio = latent_dim / D_orig
    n_meta = 1 if symmetric else 2
    out = {}
    out["dense_index"] = ratio * float(mean_quant_bits)
    out["scale"] = n_meta * 16.0 * num_groups / D_orig
    if count_channel_scale and seq_len > 0:
        # static per-dim channel scale (factored/kvtc), amortized over the sequence
        out["scale"] += ratio * 16.0 / seq_len
    out["sparse"] = 0.0
    # window tokens keep the *latent* z in fp16 → 16 bits per latent dim, so the
    # per-original-element cost is scaled by the rank ratio (only latent is stored).
    out["fp16_window"] = ratio * _fp16_window_bits(num_sink_tokens, residual_length, seq_len)
    out["total"] = out["dense_index"] + out["scale"] + out["sparse"] + out["fp16_window"]
    out["ratio"] = ratio
    return out


def palu_effective_bits_from_file(bitwidth_file, num_kv_heads=8, head_dim=128,
                                  num_sink_tokens=0, residual_length=0, seq_len=8192,
                                  symmetric=True, count_channel_scale=False,
                                  num_groups_override=None):
    """Load a per-dim bitwidth .pt file and compute K, V, and mean effective bits.

    num_groups_override: number of factored head groups (token-adaptive alpha is
        stored once per group per token).  The bit-allocation file's own head
        grouping may differ from the scaling grouping, so pass the model's actual
        head-group count (e.g. 2 for G=2 → 4 heads/group on Llama-3.1-8B).
    """
    import torch
    d = torch.load(bitwidth_file, map_location="cpu", weights_only=False)

    def _proj_stats(alloc):
        total_dims = 0
        total_bits = 0.0
        num_groups = 0
        for _layer, heads in alloc.items():
            num_groups = max(num_groups, len(heads))  # latent groups per layer
            for _h, t in heads.items():
                t = t.float()
                total_dims += t.numel()
                total_bits += float(t.sum())
        n_layers = len(alloc)
        latent_dim = total_dims / n_layers          # latent dim per layer per token
        mean_bits = total_bits / total_dims         # mean quant bits per latent dim
        return latent_dim, mean_bits, num_groups

    res = {}
    for proj, key in (("k", "bit_alloc_k"), ("v", "bit_alloc_v")):
        if key not in d:
            continue
        latent_dim, mean_bits, num_groups = _proj_stats(d[key])
        if num_groups_override is not None:
            num_groups = num_groups_override
        res[proj] = palu_effective_bits(
            mean_bits, latent_dim, num_kv_heads, head_dim, num_groups,
            num_sink_tokens=num_sink_tokens, residual_length=residual_length,
            seq_len=seq_len, symmetric=symmetric,
            count_channel_scale=count_channel_scale)
    if "k" in res and "v" in res:
        res["mean"] = {kk: (res["k"][kk] + res["v"][kk]) / 2.0 for kk in res["k"]}
    return res


def format_breakdown(method, bd, seq_len, kv_dim):
    parts = [f"index {bd['dense_index']:.2f}", f"scale {bd['scale']:.3f}"]
    if bd["sparse"] > 0:
        parts.append(f"sparse {bd['sparse']:.3f}")
    if bd["fp16_window"] > 0:
        parts.append(f"fp16-window {bd['fp16_window']:.3f}")
    return (f"[{method}] effective bits ≈ {bd['total']:.2f}/elem "
            f"({' + '.join(parts)}) @ seq={seq_len}, kv_dim={kv_dim}")


def main():
    p = argparse.ArgumentParser(description="KV-cache effective bits/element estimator")
    p.add_argument("--method", choices=["kvquant", "kivi", "aatc"], required=True,
                   help="aatc reads a per-channel bitwidth .pt from run_bit_allocation.py; "
                        "kivi/kvquant take a nominal --nbits.")
    p.add_argument("--nbits", type=int, default=2)
    p.add_argument("--seqlen", type=int, default=8192)
    p.add_argument("--num_kv_heads", type=int, default=8, help="Llama-3.1-8B: 8")
    p.add_argument("--head_dim", type=int, default=128, help="Llama-3.1-8B: 128")
    p.add_argument("--num_sink_tokens", type=int, default=None)
    p.add_argument("--residual_length", type=int, default=None)
    # kvquant
    p.add_argument("--sparsity_threshold", type=float, default=0.99)
    p.add_argument("--no_sparse", action="store_true")
    p.add_argument("--count_key_scale", action="store_true")
    # kivi
    p.add_argument("--group_size", type=int, default=32)
    # aatc
    p.add_argument("--bitwidth_file", type=str, default=None,
                   help="aatc: per-channel bitwidth .pt file from run_bit_allocation.py")
    p.add_argument("--asymmetric", action="store_true",
                   help="aatc: model asymmetric quant (scale+zero, 2 fp16/group). "
                        "Default is symmetric (factored/kvtc: 1 fp16/group).")
    p.add_argument("--count_channel_scale", action="store_true",
                   help="aatc: amortize the static per-dim fc_scale over the sequence.")
    p.add_argument("--num_groups", type=int, default=None,
                   help="aatc: number of factored head groups (token alpha is stored "
                        "once per group per token). E.g. 2 for G=2 (4 heads/group).")
    args = p.parse_args()

    if args.method == "aatc":
        if args.bitwidth_file is None:
            p.error("--bitwidth_file is required for method 'aatc'")
        res = palu_effective_bits_from_file(
            args.bitwidth_file, num_kv_heads=args.num_kv_heads, head_dim=args.head_dim,
            num_sink_tokens=args.num_sink_tokens if args.num_sink_tokens is not None else 0,
            residual_length=args.residual_length if args.residual_length is not None else 0,
            seq_len=args.seqlen, symmetric=not args.asymmetric,
            count_channel_scale=args.count_channel_scale,
            num_groups_override=args.num_groups)
        kv_dim = args.num_kv_heads * args.head_dim
        for proj in ("k", "v", "mean"):
            if proj in res:
                print(format_breakdown(f"aatc-{proj}", res[proj], args.seqlen, kv_dim))
        return

    if args.method == "kvquant":
        bd = kvquant_effective_bits(
            args.nbits, args.seqlen, args.num_kv_heads, args.head_dim,
            include_sparse=not args.no_sparse,
            sparsity_threshold=args.sparsity_threshold,
            num_sink_tokens=args.num_sink_tokens if args.num_sink_tokens is not None else 5,
            residual_length=args.residual_length if args.residual_length is not None else 0,
            count_key_scale=args.count_key_scale,
        )
    else:
        bd = kivi_effective_bits(
            args.nbits, args.seqlen, args.num_kv_heads, args.head_dim,
            group_size=args.group_size,
            num_sink_tokens=args.num_sink_tokens if args.num_sink_tokens is not None else 0,
            residual_length=args.residual_length if args.residual_length is not None else 128,
        )
    print(format_breakdown(args.method, bd, args.seqlen, args.num_kv_heads * args.head_dim))


if __name__ == "__main__":
    main()
