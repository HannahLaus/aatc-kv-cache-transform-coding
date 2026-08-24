"""Offline calibration for the KVQuant KV cache (pure PyTorch).

Produces the quantizer file consumed by ``KVQuantizedCache`` in
``kvquant_cache.py``.  Faithful to SqueezeAILab/KVQuant's calibration
(`quant/llama_simquant.py` + `simquant_module_quantizer.py`):

  1. Run the model over a small calibration set and capture the **pre-RoPE**
     Key/Value activations — the outputs of k_proj / v_proj (before the rotary
     embedding), which is the frame KVQuantizedCache quantizes in.
  2. Compute **Fisher information** as the squared gradient of the LM loss
     w.r.t. those Key/Value activations (per element).
  3. For each layer:
       * Keys  — per-channel (per kv_head, head_dim) outlier thresholds
         (range/zero-point); pool normalized non-outlier values across all
         channels and fit **Fisher-weighted k-means** → NUQ look-up table.
       * Values — per-token (dynamic) normalization; pool normalized
         non-outlier values and fit Fisher-weighted k-means → NUQ LUT.
  4. Save ``{"config": ..., "layers": {i: {k_thr_upper, k_thr_lower,
     k_lut, v_lut}}}`` to a ``.pt`` file.

Example
-------
    CUDA_VISIBLE_DEVICES=0 python run_kvquant_calibration.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --nbits 4 --nsamples 8 --seqlen 512 \
        --sparsity_threshold 0.99 \
        --output kvquant_quantizers_llama31_8b_4bit.pt
"""

import argparse
import math

import torch
from datasets import load_dataset

from utils import load_model_and_tokenizer
from kvquant_cache import round_to_lut

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── Pre-RoPE K/V capture via k_proj / v_proj forward hooks ───────────────────

def register_capture_hooks(model, captured):
    """Hook k_proj/v_proj outputs (pre-RoPE) on every decoder layer.

    KVQuant calibrates on the *pre-RoPE* key activations, which are exactly the
    outputs of k_proj (before the rotary embedding is applied) — the same frame
    KVQuantizedCache quantizes in.  ``retain_grad`` lets us read their gradient
    (→ Fisher) after backward.  Stores into ``captured[layer_idx] = {'k','v'}``.
    """
    handles = []
    for li, layer in enumerate(model.model.layers):
        def mk(idx, which):
            def hook(_module, _inp, out):
                if out.requires_grad:
                    out.retain_grad()
                captured.setdefault(idx, {})[which] = out
            return hook
        handles.append(layer.self_attn.k_proj.register_forward_hook(mk(li, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(mk(li, "v")))
    return handles


# ── Calibration data ─────────────────────────────────────────────────────────

def get_calibration_inputs(tokenizer, nsamples, seqlen, seed=0, dataset="wikitext2"):
    """Sample ``nsamples`` ``seqlen``-token windows (GPTQ/KVQuant style)."""
    if dataset == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
        ids = enc.input_ids[0]
        g = torch.Generator().manual_seed(seed)
        max_start = ids.numel() - seqlen - 1
        starts = torch.randint(0, max_start, (nsamples,), generator=g)
        return [ids[s:s + seqlen].unsqueeze(0) for s in starts]

    if dataset == "c4":
        # One shard of C4 (≈300 MB) — avoids downloading the full corpus.
        # Matches KVQuant/GPTQ: random documents, random seqlen-token windows.
        import random
        data = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train")
        rng = random.Random(seed)
        samples = []
        while len(samples) < nsamples:
            row = data[rng.randint(0, len(data) - 1)]
            enc = tokenizer(row["text"], return_tensors="pt")
            if enc.input_ids.shape[1] <= seqlen:
                continue
            j = rng.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            samples.append(enc.input_ids[:, j:j + seqlen])
        return samples

    raise ValueError(f"Unknown calibration dataset: {dataset!r} (use wikitext2 or c4).")


# ── k-means LUT fitting ──────────────────────────────────────────────────────

def fit_lut(values, weights, n_clusters, max_points=200_000, seed=0):
    """Fisher-weighted 1-D k-means → sorted centroids (the NUQ signposts)."""
    values = values.reshape(-1).float().cpu().numpy()
    weights = weights.reshape(-1).float().cpu().numpy() if weights is not None else None
    if values.shape[0] > max_points:
        import numpy as np
        rng = np.random.default_rng(seed)
        idx = rng.choice(values.shape[0], size=max_points, replace=False)
        values = values[idx]
        weights = weights[idx] if weights is not None else None
    import numpy as np
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto", max_iter=50)
    km.fit(values.reshape(-1, 1), sample_weight=weights)
    centroids = np.sort(km.cluster_centers_.reshape(-1))
    return torch.tensor(centroids, dtype=torch.float32)


def fit_qnorm(norm_vals, inlier_mask, lut):
    """KVQuant Q-Norm: affine (scale, offset) matching the quantized inliers'
    mean/std to the original.  Computed on the normalized inlier values."""
    inl = norm_vals[inlier_mask].float()
    if inl.numel() == 0:
        return 1.0, 0.0
    m1 = inl.mean()
    stdev1 = ((inl - m1) ** 2).mean().sqrt()
    q = round_to_lut(inl, lut)                       # quantize the inliers
    m2 = q.mean()
    stdev2 = ((q - m2) ** 2).mean().sqrt()
    if stdev2 <= 0:
        return 1.0, 0.0
    normscale = (stdev1 / stdev2).item()
    normoffset = (-m2 * normscale + m1).item()
    return normscale, normoffset


def calibrate_layer(k_acts, k_fish, v_acts, v_fish, nbits,
                    include_sparse, sparsity_threshold, qnorm=False):
    """Return per-layer quantizer dict from captured activations + Fisher.

    k_acts/k_fish : [N, H, D]  (pre-RoPE keys, per-channel)
    v_acts/v_fish : [T, F]     (values flattened to per-token feature vectors)
    """
    n_clusters = 2 ** nbits
    t = 1 - ((1 - sparsity_threshold) / 2) if include_sparse else 1.0

    # ---- Keys: per-channel (H, D) thresholds + shared LUT ----
    N, H, D = k_acts.shape
    flat = k_acts.reshape(N, H * D)
    thr_u = torch.quantile(flat.float(), t, dim=0).reshape(H, D)
    thr_l = torch.quantile(flat.float(), 1 - t, dim=0).reshape(H, D)
    zp = (thr_u + thr_l) / 2
    rng = ((thr_u - thr_l) / 2).clamp(min=1e-8)

    k_norm = (k_acts - zp.unsqueeze(0)) / rng.unsqueeze(0)
    k_inlier = k_norm.abs() <= 1.0
    k_lut = fit_lut(k_norm[k_inlier], (k_fish[k_inlier] if k_fish is not None else None), n_clusters)

    # ---- Values: per-token dynamic normalization, shared LUT ----
    if include_sparse:
        vu = torch.quantile(v_acts.float(), t, dim=-1, keepdim=True)
        vl = torch.quantile(v_acts.float(), 1 - t, dim=-1, keepdim=True)
        vmask = (v_acts >= vu) | (v_acts <= vl)
        outliers = v_acts * vmask
        median = torch.median(v_acts, dim=-1, keepdim=True).values
        tmp = v_acts - outliers + median * vmask
        vmax = tmp.amax(dim=-1, keepdim=True)
        vmin = tmp.amin(dim=-1, keepdim=True)
    else:
        vmax = v_acts.amax(dim=-1, keepdim=True)
        vmin = v_acts.amin(dim=-1, keepdim=True)
    voff = (vmax + vmin) / 2
    vrng = ((vmax - vmin) / 2).clamp(min=1e-8)
    v_norm = (v_acts - voff) / vrng
    v_inlier = v_norm.abs() <= 1.0
    v_lut = fit_lut(v_norm[v_inlier], (v_fish[v_inlier] if v_fish is not None else None), n_clusters)

    out = {
        "k_thr_upper": thr_u.cpu(),
        "k_thr_lower": thr_l.cpu(),
        "k_lut": k_lut.cpu(),
        "v_lut": v_lut.cpu(),
    }
    if qnorm:
        ks, ko = fit_qnorm(k_norm, k_inlier, k_lut)
        vs, vo = fit_qnorm(v_norm, v_inlier, v_lut)
        out.update(k_normscale=ks, k_normoffset=ko, v_normscale=vs, v_normoffset=vo)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

@torch.enable_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--nbits", type=int, default=4, choices=[2, 3, 4, 5])
    p.add_argument("--nsamples", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", choices=["wikitext2", "c4"], default="wikitext2",
                   help="Calibration corpus (paper uses wikitext2; c4 is broader).")
    p.add_argument("--sparsity_threshold", type=float, default=0.99,
                   help="Keep top/bottom (1-thr)/2 as fp16 outliers (0.99 ≈ 1%% total).")
    p.add_argument("--no_sparse", action="store_true", help="Disable dense-and-sparse.")
    p.add_argument("--no_fisher", action="store_true", help="Unweighted k-means.")
    p.add_argument("--norm", action="store_true",
                   help="Q-Norm: store per-layer affine (scale, offset) so the "
                        "quantized values match the original mean/std.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for NUQ k-means calibration "
                           "(`pip install scikit-learn`).")

    include_sparse = not args.no_sparse
    model, tokenizer = load_model_and_tokenizer(args.model)
    model.config.use_cache = True

    # Build the autograd graph cheaply: freeze everything, enable grad only on
    # the input embedding so backward can reach the retained K/V activations
    # without storing grads for every weight.
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    if not args.no_fisher:
        model.get_input_embeddings().weight.requires_grad_(True)

    inputs = get_calibration_inputs(tokenizer, args.nsamples, args.seqlen,
                                    args.seed, args.dataset)
    device = next(model.parameters()).device

    # k_proj / v_proj output dims (GQA: num_key_value_heads, not attention heads)
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)

    captured = {}
    handles = register_capture_hooks(model, captured)

    # Accumulate per-layer captured activations (+ Fisher) on CPU.
    k_acts, k_fish, v_acts, v_fish = {}, {}, {}, {}

    try:
        for si, ids in enumerate(inputs):
            ids = ids.to(device)
            captured.clear()
            out = model(ids, use_cache=False)

            if not args.no_fisher:
                logits = out.logits[:, :-1, :].float()
                labels = ids[:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                model.zero_grad(set_to_none=True)
                loss.backward()

            for li, cap in captured.items():
                k = cap["k"]   # [B, S, n_kv*head_dim]  (pre-RoPE)
                v = cap["v"]   # [B, S, n_kv*head_dim]
                B, S, _ = k.shape
                kc = k.detach().float().reshape(B * S, n_kv, head_dim)   # [N, H, D]
                vc = v.detach().float().reshape(B * S, n_kv * head_dim)  # [T, F]
                k_acts.setdefault(li, []).append(kc.cpu())
                v_acts.setdefault(li, []).append(vc.cpu())
                if not args.no_fisher and k.grad is not None:
                    kg = (k.grad.detach().float() ** 2).reshape(B * S, n_kv, head_dim)
                    vg = (v.grad.detach().float() ** 2).reshape(B * S, n_kv * head_dim)
                    k_fish.setdefault(li, []).append(kg.cpu())
                    v_fish.setdefault(li, []).append(vg.cpu())
            del out
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  captured sample {si + 1}/{len(inputs)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    print("Fitting NUQ look-up tables (Fisher-weighted k-means)...", flush=True)
    layers = {}
    for li in sorted(k_acts):
        ka = torch.cat(k_acts[li], dim=0)             # [N, H, D]
        va = torch.cat(v_acts[li], dim=0)             # [T, F]
        kf = torch.cat(k_fish[li], dim=0) if li in k_fish else None
        vf = torch.cat(v_fish[li], dim=0) if li in v_fish else None
        layers[li] = calibrate_layer(ka, kf, va, vf, args.nbits,
                                     include_sparse, args.sparsity_threshold,
                                     qnorm=args.norm)
        print(f"  layer {li}: k_lut={layers[li]['k_lut'].numel()} levels, "
              f"v_lut={layers[li]['v_lut'].numel()} levels", flush=True)

    quantizer = {
        "config": {
            "model": args.model,
            "nbits": args.nbits,
            "include_sparse": include_sparse,
            "sparsity_threshold": args.sparsity_threshold,
            "fisher": not args.no_fisher,
            "qnorm": args.norm,
            "dataset": args.dataset,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "pre_rope": True,  # keys calibrated in the pre-RoPE frame (k_proj output)
        },
        "layers": layers,
    }
    torch.save(quantizer, args.output)
    print(f"Saved KVQuant quantizers for {len(layers)} layers → {args.output}")


if __name__ == "__main__":
    main()
