import math
import os
import torch
import torch.nn as nn
import re
from typing import Optional, Any, Tuple, Dict
from .quant import Quantizer
from .hadamard_utils import apply_hadamard
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    # Fallback for older transformers versions
    DynamicCache = None

# One-time log for pre-cache A(x) magnitude (layer 0 only, to compare with dequant in cache wrapper)
_pre_cache_magnitude_logged: set = set()
# Debug: log bitwidth source when seq_len is 18-20 (layer 0) to trace why token 9 can be 8-bit before 24 tokens
_per_dim_bitwidth_debug_logged: set = set()
# One-time trace: writer scale/bitwidth path for k vs v (debug key-specific issues)
_writer_scale_trace_logged: set = set()

def _per_head_whiten_decomposition_from_weight(weight, scaling_diag_matrix, rank, uneven_split=False):
    original_dtype = weight.dtype
    try:
        scaling_diag_matrix = scaling_diag_matrix.to(weight.device)
    except AttributeError:
        raise FileExistsError("Cache may not be loaded correctly")
    
    # Get the inverse of scaling_diag_matrix
    try:
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))
    except Exception as e:
        # If inversion fails, try with regularization
        logger.warning(f"Failed to invert scaling_diag_matrix: {e}, trying with regularization")
        reg = 1e-6 * torch.eye(scaling_diag_matrix.shape[0], device=scaling_diag_matrix.device, dtype=torch.float32)
        scaling_diag_matrix_reg = scaling_diag_matrix.to(torch.float32) + reg
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix_reg)
        del reg, scaling_diag_matrix_reg

    # Multiply scaling_diag_matrix to weight matrix
    W_scale = torch.matmul(weight.to(torch.float32), scaling_diag_matrix.to(torch.float32))
    
    # Check condition number to detect ill-conditioned matrices
    try:
        # Compute condition number (ratio of largest to smallest singular value)
        # For efficiency, we'll just try SVD and catch the error
        U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
    except RuntimeError as e:
        if "failed to converge" in str(e) or "ill-conditioned" in str(e) or "error code: 14" in str(e):
            logger.warning(f"SVD failed due to ill-conditioned matrix, adding regularization: {e}")
            # Add small regularization to make the matrix better conditioned
            # Regularize by adding a small multiple of identity to W_scale^T W_scale
            # This is equivalent to regularizing the covariance
            reg_amount = 1e-6
            max_reg = 1e-2
            U, S, Vt = None, None, None
            
            while reg_amount <= max_reg and U is None:
                try:
                    # Regularize by adding small value to diagonal
                    # For rectangular matrices, add to the smaller dimension
                    min_dim = min(W_scale.shape[0], W_scale.shape[1])
                    reg_diag = reg_amount * torch.eye(min_dim, device=W_scale.device, dtype=W_scale.dtype)
                    
                    if W_scale.shape[0] == W_scale.shape[1]:
                        # Square matrix: add directly
                        W_scale_reg = W_scale + reg_diag
                    elif W_scale.shape[0] < W_scale.shape[1]:
                        # Tall matrix: add to top-left block
                        W_scale_reg = W_scale.clone()
                        W_scale_reg[:, :min_dim] += reg_diag
                    else:
                        # Wide matrix: add to top-left block
                        W_scale_reg = W_scale.clone()
                        W_scale_reg[:min_dim, :] += reg_diag
                    
                    U, S, Vt = torch.linalg.svd(W_scale_reg, full_matrices=False)
                    logger.info(f"SVD succeeded with regularization {reg_amount:.2e}")
                    W_scale = W_scale_reg  # Use regularized version
                    del W_scale_reg, reg_diag
                    break
                except RuntimeError as e2:
                    logger.debug(f"Regularization {reg_amount:.2e} failed: {e2}, trying larger regularization...")
                    reg_amount *= 10.0
            
            if U is None:
                # Last resort: try double precision
                logger.warning(f"All regularization attempts failed, trying double precision")
                try:
                    W_scale_double = W_scale.double()
                    U, S, Vt = torch.linalg.svd(W_scale_double, full_matrices=False)
                    U, S, Vt = U.float(), S.float(), Vt.float()
                    logger.info("SVD succeeded with double precision")
                except RuntimeError:
                    # Final fallback: use very large regularization
                    logger.warning(f"Double precision failed, using large regularization as final fallback")
                    min_dim = min(W_scale.shape[0], W_scale.shape[1])
                    reg_diag = 1e-2 * torch.eye(min_dim, device=W_scale.device, dtype=W_scale.dtype)
                    if W_scale.shape[0] == W_scale.shape[1]:
                        W_scale_reg = W_scale + reg_diag
                    elif W_scale.shape[0] < W_scale.shape[1]:
                        W_scale_reg = W_scale.clone()
                        W_scale_reg[:, :min_dim] += reg_diag
                    else:
                        W_scale_reg = W_scale.clone()
                        W_scale_reg[:min_dim, :] += reg_diag
                    U, S, Vt = torch.linalg.svd(W_scale_reg, full_matrices=False)
                    W_scale = W_scale_reg
        else:
            raise
    
    V = torch.matmul(Vt, scaling_matrix_inv)
    
    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    
    if uneven_split:
        # Uneven split: L = U @ S, R = V
        Sigma = torch.diag(S)
        L = torch.matmul(U, Sigma).to(original_dtype)
        R = V.to(original_dtype)
    else:
        # Even split: L = U @ sqrt(S), R = sqrt(S) @ V
        sqrtSigma = torch.sqrt(torch.diag(S))
        L = torch.matmul(U, sqrtSigma).to(original_dtype)
        R = torch.matmul(sqrtSigma, V).to(original_dtype)
    
    return L, R

def _per_head_decomposition_from_weight(weight, rank):
    original_dtype = weight.dtype
    # Get weight matrix decomposed
    U, S, Vt = torch.linalg.svd(weight.to(torch.float32), full_matrices=False)

    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]

    sqrtSigma = torch.sqrt(torch.diag(S))
    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, Vt).to(original_dtype)
    assert torch.allclose(torch.matmul(L, R), weight, atol=1e-3), "SVD decomposition failed"
    return L, R


def _to_cache_quant_dtype(t: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Cast quantized latents to int8 or int16 for cache storage (bitwidths at most 16).
    - n_bits <= 8: clamp to [-128, 127] and cast to torch.int8
    - n_bits > 8: clamp to [-32768, 32767] and cast to torch.int16
    """
    if n_bits <= 8:
        t = t.clamp(-128, 127).to(torch.int8)
    else:
        t = t.clamp(-32768, 32767).to(torch.int16)
    return t


class QuantizedCache:
    """
    Cache for storing quantized A(x) values, scales, and bitwidths.
    This allows truncation of cached quantized values as the sliding window advances.
    
    What we store:
    - QuantA(x): The quantized representation of A(x) = VT(x)
      After quantization: A(x) → QuantA(x) → dequantize → A'(x) → BA'(x)
      We cache QuantA(x) so we can truncate it to lower bitwidths without re-quantizing.
      
      Storage dtype: int8 (n_bits <= 8) or int16 (n_bits > 8). Readers must convert to float
      before dequantization (e.g. .float()).
    
    Structure:
    - quantized_latents[layer_name]: Dict[str, torch.Tensor] - QuantA(x) values per layer
      Format: {"k": tensor, "v": tensor} for key and value projections
      Shape: (batch_size, seq_len, num_dims)
      Dtype: torch.int8 (n_bits <= 8) or torch.int16 (n_bits > 8)
      Values: Integer values in range determined by bitwidth
    - scales[layer_name]: Dict[str, torch.Tensor] - scales (alpha) per token
      Shape: (batch_size, seq_len) or (batch_size, seq_len, num_groups)
      Used for dequantization: A'(x) = QuantA(x) * stepsize, where stepsize = 2*alpha / 2^b
    - bitwidths[layer_name]: Dict[str, torch.Tensor] - current bitwidths per token/dim
      Shape: (seq_len, num_dims) - bitwidth for each token and dimension (can be 0-16 bits)
    """
    
    def __init__(self):
        self.quantized_latents = {}  # layer_name -> {"k": tensor, "v": tensor} - actual quantized integers
        self.scales_dict = {}  # layer_name -> {"k": dict, "v": dict} - scales_dict from quantize_latent_to_integers
        self.bitwidths = {}  # layer_name -> {"k": tensor, "v": tensor}  # (seq_len, num_dims)
        self.seq_lengths = {}  # layer_name -> int
        # Keep old scales for backward compatibility during transition
        self.scales = {}  # layer_name -> {"k": tensor, "v": tensor} - deprecated, use scales_dict instead
        # Per-dim path: store channelwise_scaling (SC) so wrapper can restore magnitude when dequantizing
        self.channelwise_scaling = {}  # layer_name -> {"k": tensor, "v": tensor}  # (num_dims,) or None
        # Sliding recent window: last N tokens at 16 bits; when full, quantize older N/2 to base bits
        self.recent_tokens = 0  # 0 = disabled; when > 0, last recent_tokens (or N/2 after step) at 16 bits
        self.attention_sink_tokens = 0  # first N tokens at 16 bits; used with recent_tokens to enforce no 8-bit before sliding step
        self.n_recent_full = 0  # current number of tokens at 16 bits at the end (between recent_tokens//2 and recent_tokens)
    
    def update(
        self,
        layer_name: str,
        proj_type: str,  # "k" or "v"
        quantized_latents: torch.Tensor,  # (batch_size, seq_len, num_dims) - actual quantized integers
        scales_dict: dict = None,  # Dict mapping group_idx to (scales, base) tuples from quantize_latent_to_integers
        bitwidths: torch.Tensor = None,  # (seq_len, num_dims) or (num_dims,) - will be expanded
        scales: torch.Tensor = None,  # Deprecated: kept for backward compatibility
        channelwise_scaling: torch.Tensor = None,  # (num_dims,) - SC used at quantize; apply after dequant to restore magnitude
        n_bits: Optional[int] = None,  # Used to store as int8 (<=8) or int16 (>8). Default 8.
    ):
        """
        Update cache with quantized values for a specific layer and projection type.
        
        Args:
            layer_name: Full layer name (e.g., "model.layers.0.self_attn.k_proj")
            proj_type: "k" for key or "v" for value
            quantized_latents: Quantized A(x) values (stored as int8 or int16)
            scales: Scales (alpha) per token
            bitwidths: Bitwidths per dimension (will be expanded to per-token if needed)
            n_bits: Bitwidth for cache storage dtype (int8 if <=8, int16 if >8). Default 8.
        """
        cache_n_bits = n_bits if n_bits is not None else 8
        # Cast to int8 or int16 for storage (readers will .float() before dequant)
        quantized_latents = _to_cache_quant_dtype(quantized_latents.detach().float(), cache_n_bits)
        if layer_name not in self.quantized_latents:
            self.quantized_latents[layer_name] = {}
            self.scales_dict[layer_name] = {}
            self.scales[layer_name] = {}  # Deprecated
            self.bitwidths[layer_name] = {}
            self.channelwise_scaling[layer_name] = {}
        
        batch_size, seq_len, num_dims = quantized_latents.shape
        
        # Store quantized latents
        if proj_type in self.quantized_latents[layer_name]:
            # Check if shapes match (except sequence dimension)
            existing = self.quantized_latents[layer_name][proj_type]
            existing_shape = existing.shape
            new_shape = quantized_latents.shape
            
            # Debug: log shapes for troubleshooting (only for attention layers, reduce noise)
            if existing_shape[0] != batch_size or existing_shape[2] != num_dims:
                # Only log if it's an attention layer to reduce noise
                if 'self_attn' in layer_name:
                    logger.debug(
                        f"Cache shape mismatch for {layer_name}[{proj_type}]: "
                        f"existing={existing_shape}, new={new_shape}, "
                        f"batch_match={existing_shape[0]==batch_size}, dims_match={existing_shape[2]==num_dims}. "
                        f"Will replace cache instead of concatenating."
                    )
            
            # Check all dimensions match except sequence dimension (dim=1)
            if (existing_shape[0] == new_shape[0] and  # batch_size matches
                existing_shape[2] == new_shape[2] and  # num_dims matches
                len(existing_shape) == len(new_shape) == 3):  # both are 3D
                # Determine if we should concatenate or replace:
                # - If new_seq_len == 1: token-by-token generation, concatenate
                # - If new_seq_len > 1: prompt processing (all tokens at once), replace
                new_seq_len = new_shape[1]
                existing_seq_len = existing_shape[1]
                
                # QuantizedCache update logging removed for cleaner output
                
                if new_seq_len == 1:
                    # Token-by-token generation: concatenate
                    # CRITICAL: Only concatenate if the existing cache has a reasonable sequence length
                    # The existing cache should have (prompt_len + num_generated_tokens) tokens
                    # Sanity check: if cache is way too large, something is wrong - replace instead of concatenate
                    # Also check: existing_seq_len should be close to what we expect (prompt + generated tokens)
                    # Upper bound to detect real anomalies (e.g. double-appends).
                    # Must be large enough for long-context workloads
                    # (prompts can be 40k+ tokens, plus generated tokens).
                    max_expected_tokens = 131072
                    if existing_seq_len > max_expected_tokens:
                        logger.warning(
                            f"Cache size anomaly for {layer_name}[{proj_type}]: "
                            f"existing_seq_len={existing_seq_len} > {max_expected_tokens}. Replacing instead of concatenating."
                        )
                        self.quantized_latents[layer_name][proj_type] = quantized_latents
                    else:
                        # existing_seq_len is reasonable (between 1 and max_expected_tokens)
                        # This is likely fine - concatenate normally
                        try:
                            concatenated = torch.cat([existing, quantized_latents], dim=1)
                            self.quantized_latents[layer_name][proj_type] = concatenated
                        except RuntimeError as e:
                            logger.error(
                                f"Failed to concatenate cache for {layer_name}[{proj_type}]: {e}. "
                                f"existing={existing_shape}, new={new_shape}. Replacing cache."
                            )
                            self.quantized_latents[layer_name][proj_type] = quantized_latents
                else:
                    # Prompt processing (multiple tokens at once): replace
                    # This happens during initial prompt processing where we process all tokens together
                    # Replace cache instead of concatenating
                    self.quantized_latents[layer_name][proj_type] = quantized_latents
            else:
                # Shape mismatch - replace instead of concatenate
                # Only log warnings for attention layers to reduce noise
                is_layer_0_cache = 'layers.0' in layer_name if layer_name else False
                if 'self_attn' in layer_name:
                    logger.debug(
                        f"Shape mismatch in cache update for {layer_name}[{proj_type}]: "
                        f"existing={existing_shape}, new={new_shape}. Replacing cache."
                    )
                self.quantized_latents[layer_name][proj_type] = quantized_latents
        else:
            self.quantized_latents[layer_name][proj_type] = quantized_latents
        
        # Store scales_dict (new approach - stores actual quantization scales per group)
        if scales_dict is not None:
            if layer_name not in self.scales_dict:
                self.scales_dict[layer_name] = {}
            
            # For scales_dict, we need to handle concatenation carefully
            # scales_dict is a dict mapping group_idx -> (scales, base) where scales/base are (batch*seq, 1)
            # When concatenating, we need to concatenate the scales and base tensors along the batch*seq dimension
            if proj_type in self.scales_dict[layer_name]:
                existing_scales_dict = self.scales_dict[layer_name][proj_type]
                # Concatenate scales and base for each group
                new_seq_len = quantized_latents.shape[1]
                existing_seq_len = existing.shape[1] if proj_type in self.quantized_latents[layer_name] else 0
                
                if new_seq_len == 1 and existing_seq_len > 0:
                    # Token-by-token generation: concatenate scales_dict
                    concatenated_scales_dict = {}
                    for group_idx in scales_dict:
                        if group_idx in existing_scales_dict:
                            existing_scales, existing_base = existing_scales_dict[group_idx]
                            new_scales, new_base = scales_dict[group_idx]
                            # Concatenate along the batch*seq dimension (dim=0)
                            concatenated_scales = torch.cat([existing_scales, new_scales], dim=0)
                            concatenated_base = torch.cat([existing_base, new_base], dim=0)
                            concatenated_scales_dict[group_idx] = (concatenated_scales, concatenated_base)
                        else:
                            concatenated_scales_dict[group_idx] = scales_dict[group_idx]
                    self.scales_dict[layer_name][proj_type] = concatenated_scales_dict
                else:
                    # Prompt processing or first time: replace
                    self.scales_dict[layer_name][proj_type] = scales_dict
            else:
                # First time storing for this proj_type
                self.scales_dict[layer_name][proj_type] = scales_dict
        
        # Store scales (deprecated - kept for backward compatibility)
        if scales is not None:
            existing_scales = None
            if layer_name in self.scales and proj_type in self.scales[layer_name]:
                existing_scales = self.scales[layer_name][proj_type]
            if existing_scales is None:
                # First time or no existing: store directly
                self.scales[layer_name][proj_type] = scales
            elif existing_scales.shape[0] == batch_size:
                # Check if we can concatenate along sequence dimension (dim=1)
                # Scales can be (batch, seq) or (batch, seq, groups)
                if len(existing_scales.shape) == len(scales.shape):
                    # Same number of dimensions - check if we should concatenate or replace
                    if existing_scales.shape[0] == scales.shape[0]:  # batch sizes match
                        # Determine if we should concatenate or replace based on new sequence length
                        new_seq_len = scales.shape[1] if len(scales.shape) >= 2 else 1
                        
                        if new_seq_len == 1:
                            # Token-by-token generation: concatenate
                            try:
                                self.scales[layer_name][proj_type] = torch.cat(
                                    [existing_scales, scales], dim=1
                                )
                            except RuntimeError:
                                # Concatenation failed - replace
                                if 'self_attn' in layer_name:
                                    logger.debug(
                                        f"Scale concatenation failed for {layer_name}[{proj_type}]: "
                                        f"existing={existing_scales.shape}, new={scales.shape}. Replacing cache."
                                    )
                                self.scales[layer_name][proj_type] = scales
                        else:
                            # Prompt processing (multiple tokens at once): replace
                            self.scales[layer_name][proj_type] = scales
                    else:
                        # Batch size mismatch - replace
                        self.scales[layer_name][proj_type] = scales
                else:
                    # Different number of dimensions - replace
                    if 'self_attn' in layer_name:
                        logger.debug(
                            f"Scale dimension mismatch for {layer_name}[{proj_type}]: "
                            f"existing={existing_scales.shape}, new={scales.shape}. Replacing cache."
                        )
                    self.scales[layer_name][proj_type] = scales
            else:
                # Batch size mismatch - replace
                self.scales[layer_name][proj_type] = scales
        
        # Store channelwise_scaling (SC) for per-dim path so wrapper can restore magnitude after dequant
        if channelwise_scaling is not None and channelwise_scaling.numel() == num_dims:
            self.channelwise_scaling[layer_name][proj_type] = channelwise_scaling.detach().to(
                quantized_latents.device, copy=True
            )
        # One-time log for layer 0 so we see write path: storing SC vs not (explains "cache has no channelwise_scaling" at read)
        if "self_attn" in layer_name and "layers.0" in layer_name and proj_type in ("k", "v"):
            if not hasattr(QuantizedCache, "_write_sc_logged"):
                QuantizedCache._write_sc_logged = set()
            key = (0, proj_type)
            if key not in QuantizedCache._write_sc_logged:
                QuantizedCache._write_sc_logged.add(key)
                if channelwise_scaling is not None and channelwise_scaling.numel() == num_dims:
                    logger.info(
                        f"[per-dim WRITE Layer 0 {proj_type}] storing channelwise_scaling (numel={channelwise_scaling.numel()})"
                    )
                else:
                    logger.info(
                        f"[per-dim WRITE Layer 0 {proj_type}] NOT storing channelwise_scaling "
                        f"(module passed {'None' if channelwise_scaling is None else 'wrong shape'})"
                    )
                # Sanity: log stored scales (alpha) for layer 0 so we can compare with read.
                # When running with uniform bitwidths, 'scales' can legitimately be None; in that case just skip.
                if scales is not None:
                    s = scales.float()
                    logger.info(
                        f"[per-dim WRITE Layer 0 {proj_type}] stored scales (alpha) shape={tuple(scales.shape)} "
                        f"min={s.min().item():.6f} max={s.max().item():.6f} mean={s.mean().item():.6f}"
                    )
        
        # Store bitwidths - expand to per-token if needed
        if bitwidths.dim() == 1:
            # (num_dims,) -> expand to (seq_len, num_dims)
            bitwidths_expanded = bitwidths.unsqueeze(0).expand(seq_len, -1)
        else:
            # Already (seq_len, num_dims)
            bitwidths_expanded = bitwidths
        
        if proj_type in self.bitwidths[layer_name]:
            existing_bitwidths = self.bitwidths[layer_name][proj_type]
            # Ensure existing and new bitwidths are on the same device before concatenation/replacement
            if existing_bitwidths.device != bitwidths_expanded.device:
                existing_bitwidths = existing_bitwidths.to(bitwidths_expanded.device)
                self.bitwidths[layer_name][proj_type] = existing_bitwidths
            # Check if num_dims matches
            if existing_bitwidths.shape[1] == num_dims:
                # Determine if we should concatenate or replace based on new sequence length
                new_seq_len = bitwidths_expanded.shape[0]
                
                if new_seq_len == 1:
                    # Token-by-token generation: concatenate
                    self.bitwidths[layer_name][proj_type] = torch.cat(
                        [existing_bitwidths, bitwidths_expanded], dim=0
                    )
                    # Debug: when total length is 17-22 and layer 0, log existing row 9 and new row (trace 8-bit before 24)
                    new_total = existing_bitwidths.shape[0] + 1
                    if 17 <= new_total <= 22 and "layers.0" in layer_name and "self_attn" in layer_name:
                        _key = ("cache_concat", layer_name, proj_type, new_total)
                        if _key not in _per_dim_bitwidth_debug_logged:
                            _per_dim_bitwidth_debug_logged.add(_key)
                            r9 = existing_bitwidths[9].float() if existing_bitwidths.shape[0] > 9 else None
                            r9_str = f"min={r9.min().item():.1f} mean={r9.mean().item():.2f}" if r9 is not None else "n/a"
                            new_r = bitwidths_expanded[0].float()
                            new_str = f"min={new_r.min().item():.1f} mean={new_r.mean().item():.2f}"
                            logger.info(
                                f"[PER-DIM BITWIDTH DEBUG] Cache concat Layer 0 {proj_type} -> {new_total} tokens: "
                                f"existing token 9: {r9_str} | new token: {new_str}"
                            )
                else:
                    # Prompt processing (multiple tokens at once): replace
                    self.bitwidths[layer_name][proj_type] = bitwidths_expanded
            else:
                # Dimension mismatch - replace
                # Only log warnings for attention layers to reduce noise
                if 'self_attn' in layer_name:
                    logger.debug(
                        f"Bitwidth dimension mismatch for {layer_name}[{proj_type}]: "
                        f"existing dims={existing_bitwidths.shape[1]}, new dims={num_dims}. Replacing cache."
                    )
                self.bitwidths[layer_name][proj_type] = bitwidths_expanded
        else:
            self.bitwidths[layer_name][proj_type] = bitwidths_expanded
        
        # Enforce: no 8-bit before we have (sink + recent_tokens) tokens — sliding step runs at that length
        stored_bw = self.bitwidths[layer_name][proj_type]
        new_total = stored_bw.shape[0]
        sink = getattr(self, "attention_sink_tokens", 0)
        recent = getattr(self, "recent_tokens", 0)
        min_len = sink + recent
        if min_len > 0 and new_total < min_len and stored_bw.numel() > 0:
            stored_bw.fill_(16)
        
        # Update sequence length based on actual cached tensor size
        # This ensures seq_lengths matches the actual cached tensor sequence length
        if layer_name in self.quantized_latents and proj_type in self.quantized_latents[layer_name]:
            cached_tensor = self.quantized_latents[layer_name][proj_type]
            if cached_tensor is not None:
                # Use the actual sequence length from the cached tensor
                self.seq_lengths[layer_name] = cached_tensor.shape[1]
            else:
                # Fallback: use the new sequence length
                self.seq_lengths[layer_name] = seq_len
        else:
            # No cache yet, use the new sequence length
            self.seq_lengths[layer_name] = seq_len
        
        # Mirror to alternate key so reader finds data whether it uses .inner. or not
        if ".self_attn.inner." in layer_name:
            alt_key = layer_name.replace(".self_attn.inner.", ".self_attn.", 1)
        elif ".self_attn." in layer_name:
            alt_key = layer_name.replace(".self_attn.", ".self_attn.inner.", 1)
        else:
            alt_key = None
        if alt_key is not None and alt_key != layer_name:
            for attr, key_attr in [
                ("quantized_latents", "quantized_latents"),
                ("scales", "scales"),
                ("bitwidths", "bitwidths"),
                ("channelwise_scaling", "channelwise_scaling"),
                ("scales_dict", "scales_dict"),
            ]:
                d = getattr(self, attr)
                if layer_name in d and proj_type in d.get(layer_name, {}):
                    if alt_key not in d:
                        d[alt_key] = {}
                    d[alt_key][proj_type] = d[layer_name][proj_type]
            if layer_name in self.seq_lengths:
                self.seq_lengths[alt_key] = self.seq_lengths[layer_name]

        # Sliding recent window: update n_recent_full (tokens at 16 bits at the end)
        if getattr(self, "recent_tokens", 0) > 0:
            new_total = self.seq_lengths.get(layer_name, 0) or seq_len
            if seq_len > 1:
                # Prompt (multiple tokens at once)
                self.n_recent_full = min(self.recent_tokens, new_total)
            elif seq_len == 1:
                if new_total > 1:
                    # Decode: appended one token — guard against double-increment (update() is
                    # called once per (layer, proj), but n_recent_full should only advance once
                    # per actual new token, identified by a new new_total value)
                    last_incr = getattr(self, '_n_recent_full_last_seq', 0)
                    if new_total > last_incr:
                        self.n_recent_full = min(getattr(self, "n_recent_full", 0) + 1, self.recent_tokens)
                        self._n_recent_full_last_seq = new_total
                else:
                    self.n_recent_full = min(self.recent_tokens, new_total)
    
    @staticmethod
    def _alt_cache_key(layer_name: str):
        """Return alternate key (.inner. vs no .inner.) so reader finds data written under the other convention."""
        if not layer_name or ".self_attn." not in layer_name:
            return None
        if ".self_attn.inner." in layer_name:
            return layer_name.replace(".self_attn.inner.", ".self_attn.", 1)
        return layer_name.replace(".self_attn.", ".self_attn.inner.", 1)
    
    def get_seq_length(self, layer_name: str) -> int:
        """Get current sequence length for a layer."""
        out = self.seq_lengths.get(layer_name, 0)
        if out == 0:
            alt = self._alt_cache_key(layer_name)
            if alt:
                out = self.seq_lengths.get(alt, 0)
        return out

    def get_recent_tokens(self) -> int:
        """Size of sliding recent window (0 = disabled)."""
        return getattr(self, "recent_tokens", 0)

    def get_n_recent_full(self) -> int:
        """Current number of tokens at 16 bits at the end (between recent_tokens//2 and recent_tokens)."""
        return getattr(self, "n_recent_full", 0)

    def set_n_recent_full(self, n: int) -> None:
        """Set number of tokens at 16 bits at the end (after sliding quantize step)."""
        self.n_recent_full = max(0, n)

    def get_quantized(self, layer_name: str, proj_type: str) -> Optional[torch.Tensor]:
        """Get quantized latents for a layer and projection type."""
        if layer_name in self.quantized_latents and proj_type in self.quantized_latents[layer_name]:
            return self.quantized_latents[layer_name].get(proj_type)
        alt = self._alt_cache_key(layer_name)
        if alt and alt in self.quantized_latents:
            return self.quantized_latents[alt].get(proj_type)
        return None
    
    def get_scales(self, layer_name: str, proj_type: str) -> Optional[torch.Tensor]:
        """Get scales for a layer and projection type (deprecated - use get_scales_dict instead)."""
        if layer_name in self.scales and proj_type in self.scales[layer_name]:
            return self.scales[layer_name].get(proj_type)
        alt = self._alt_cache_key(layer_name)
        if alt and alt in self.scales:
            return self.scales[alt].get(proj_type)
        return None
    
    def get_scales_dict(self, layer_name: str, proj_type: str) -> Optional[dict]:
        """Get scales_dict for a layer and projection type."""
        if layer_name in self.scales_dict and proj_type in self.scales_dict[layer_name]:
            return self.scales_dict[layer_name].get(proj_type)
        alt = self._alt_cache_key(layer_name)
        if alt and alt in self.scales_dict:
            return self.scales_dict[alt].get(proj_type)
        return None
    
    def get_bitwidths(self, layer_name: str, proj_type: str) -> Optional[torch.Tensor]:
        """Get bitwidths for a layer and projection type."""
        if layer_name in self.bitwidths and proj_type in self.bitwidths[layer_name]:
            return self.bitwidths[layer_name].get(proj_type)
        alt = self._alt_cache_key(layer_name)
        if alt and alt in self.bitwidths:
            return self.bitwidths[alt].get(proj_type)
        return None
    
    def get_channelwise_scaling(self, layer_name: str, proj_type: str) -> Optional[torch.Tensor]:
        """Get channelwise_scaling (SC) for a layer and projection type. Apply after dequant to restore magnitude."""
        if layer_name in self.channelwise_scaling and proj_type in self.channelwise_scaling[layer_name]:
            return self.channelwise_scaling[layer_name].get(proj_type)
        alt = self._alt_cache_key(layer_name)
        if alt and alt in self.channelwise_scaling:
            return self.channelwise_scaling[alt].get(proj_type)
        return None
    
    def update_bitwidths(self, layer_name: str, proj_type: str, new_bitwidths: torch.Tensor):
        """Update bitwidths for a layer and projection type. Mirrors to alternate key (.inner. vs no .inner.) so both conventions see the same data."""
        if layer_name not in self.bitwidths:
            self.bitwidths[layer_name] = {}
        self.bitwidths[layer_name][proj_type] = new_bitwidths
        # Mirror so reader/writer using the other key see the same bitwidths (avoids overwriting updated bitwidths with base on next cache.update)
        if ".self_attn.inner." in layer_name:
            alt_key = layer_name.replace(".self_attn.inner.", ".self_attn.", 1)
        elif ".self_attn." in layer_name:
            alt_key = layer_name.replace(".self_attn.", ".self_attn.inner.", 1)
        else:
            alt_key = None
        if alt_key is not None and alt_key != layer_name:
            if alt_key not in self.bitwidths:
                self.bitwidths[alt_key] = {}
            self.bitwidths[alt_key][proj_type] = new_bitwidths

    def get_bits_per_token(self, layer_name: str, proj_type: str):
        """
        Return one bitwidth value per token (length = seq_len).
        For (seq_len, num_dims) bitwidths we take min across dims so each token has a single
        representative bit count (e.g. for uniform where we lower by one across channels).
        Returns None if no bitwidths stored; otherwise a list of ints of length seq_len.
        """
        bw = self.get_bitwidths(layer_name, proj_type)
        if bw is None:
            return None
        if bw.dim() == 1:
            return bw.cpu().tolist()
        # (seq_len, num_dims) -> min over dims -> (seq_len,)
        per_token = bw.min(dim=1)[0]
        return per_token.cpu().tolist()


class HeadwiseLowRankModule(nn.Module):
    """ Headwise low rank module """

    def __init__(self, ranks, in_features, out_features, bias):
        super().__init__()


        self.ranks = ranks
        self.num_groups = len(ranks)
        self.in_features = in_features
        self.out_features = out_features
        self.group_dim = out_features // self.num_groups

        if (self.group_dim * self.num_groups) != self.out_features:
            raise ValueError(
                f"out_features must be divisible by num_groups (got `out_features`: {self.out_features}"
                f" and `num_groups`: {self.num_groups})."
            )

        self.VT = nn.Linear(in_features, sum(ranks), bias=False)
        
        Us = []
        for r in ranks:
            Us.append(nn.Linear(r, self.group_dim, bias=bias))

        self.U = nn.ModuleList(Us)

        # Precompute uniform-rank flag for fast reconstruct path.
        # If all heads share the same rank we can do one batched matmul instead of a Python loop.
        self._uniform_rank = self.ranks[0] if len(set(self.ranks)) == 1 else None
        # Stacked weight buffers for the batched path — built lazily, not saved to state_dict.
        self.register_buffer('_W_batched', None, persistent=False)
        self.register_buffer('_b_batched', None, persistent=False)

        # Dequant helpers — precomputed once, reused every call.
        # _repeats: self.ranks as a long tensor for repeat_interleave (ranks never change).
        # _pow2_table: 2^0 .. 2^16 lookup so dequant uses a gather instead of elementwise pow.
        self.register_buffer('_repeats', torch.tensor(ranks, dtype=torch.long), persistent=False)
        self.register_buffer('_pow2_table', 2.0 ** torch.arange(17, dtype=torch.float32), persistent=False)

        self.quantized_latents = False
        self.latent_quantizer = None
        
        # Per-dimension quantization settings
        self.use_per_dim_quantization = False
        self.bitwidths = None  # Shape: (num_groups, group_rank) or (sum(ranks),)
        self.num_heads = None  # Number of heads (for per-head scaling)
        self.num_heads_per_group = None  # Number of heads per group
        
        # Scaling type: "tokenwise" (default), "channelwise", "channel_group", "factored", or "kvtc"
        self.scaling_type = "tokenwise"
        self.channelwise_scalings = None  # Tensor [num_dims] with max abs values per dimension
        self.channel_group_size = 64     # G: tokens per bucket for "channel_group" scaling
        self.kvtc_group_size = 64        # G: dims per adaptive-scale group for "kvtc" scaling
        
        # Channel-wise scaling (SC) for whitening (from covariance matrix eigendecomposition)
        self.channelwise_scaling = None  # Shape: (sum(ranks),) - per-dimension scaling factors SC
        
        self.base_bitwidths = None  # Base bitwidths from calibration
        # Sliding recent window: last recent_tokens at 16 bits; when full, quantize older N/2 to base (0 = disabled)
        self.recent_tokens = 0

    def forward(self, 
                hidden_states: torch.Tensor,
                cache_for_quantization: Optional['QuantizedCache'] = None,
                layer_name: Optional[str] = None,
                proj_type: Optional[str] = None,
                past_key_values: Optional[Any] = None,
                layer_idx: Optional[int] = None):
        """
        Forward pass with optional per-dimension quantization.
        
        Flow: A(x) → (optionally quantize/dequantize) → BA(x)
        This allows using standard DynamicCache while optionally quantizing A(x) before reconstruction.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, in_features)
            cache_for_quantization: Optional QuantizedCache to store quantized A(x) values
            layer_name: Full layer name (e.g., "model.layers.0.self_attn.k_proj") - required if cache_for_quantization is provided
            proj_type: "k" for key or "v" for value - required if cache_for_quantization is provided
        
        Returns:
            Full reconstructed output BA(x) of shape (batch_size, seq_len, out_features)
            If use_per_dim_quantization is enabled, A(x) is quantized/dequantized before reconstruction.
        """
        # Step 1: Project to latent space A(x)
        low_rank_latents = self.project_to_latent(hidden_states)
        
        # Resolve intended rank from config so we slice/store correctly even if checkpoint overwrote self.ranks/VT
        # Prefer model.head_wise_ranks (config) > _expected_ranks > self.ranks
        # Try both layer_name and alternate (.self_attn.inner. vs .self_attn.) so all layers get config rank
        expected_vt_output = None
        if layer_name and hasattr(self, '_model_weakref'):
            model = self._model_weakref() if self._model_weakref else None
            if model is not None and hasattr(model, 'head_wise_ranks') and isinstance(getattr(model, 'head_wise_ranks'), dict):
                ranks_for_layer = model.head_wise_ranks.get(layer_name)
                if ranks_for_layer is None:
                    alt_key = QuantizedCache._alt_cache_key(layer_name)
                    if alt_key:
                        ranks_for_layer = model.head_wise_ranks.get(alt_key)
                if ranks_for_layer is not None:
                    expected_vt_output = sum(ranks_for_layer)
        if expected_vt_output is None:
            expected_vt_output = sum(getattr(self, '_expected_ranks', self.ranks))
        actual_output = low_rank_latents.shape[2]
        vt_weight_shape = self.VT.weight.shape if hasattr(self.VT, 'weight') else 'N/A'
        
        # Get actual VT weight data shape for verification
        actual_vt_weight_shape = 'N/A'
        if hasattr(self.VT, 'weight') and hasattr(self.VT.weight, 'data'):
            actual_vt_weight_shape = self.VT.weight.data.shape
        
        # CRITICAL: Always check VT weight shape - log error if mismatch (once per module instance)
        # Use a module-specific attribute to track logging per instance
        layer_id = layer_name if layer_name else f"unknown_layer_{id(self)}"
        module_id = id(self)
        vt_id = id(self.VT) if hasattr(self, 'VT') else 'N/A'
        
        # Log module and VT instance IDs for layers 0-2 to verify they're different
        # Use regex to match exact layer numbers (0, 1, 2) not substrings (10, 20, etc.)
        is_first_few_layers = False
        if layer_id:
            match = re.search(r'layers\.(\d+)', layer_id)
            if match:
                layer_num = int(match.group(1))
                is_first_few_layers = layer_num in [0, 1, 2]
        
        # On first forward pass, check if modules are still unique and have correct ranks
        if is_first_few_layers and not hasattr(self, '_first_forward_logged'):
            self._first_forward_logged = True
            has_vt_checked = hasattr(self, '_vt_checked')
            
            # Check if module was replaced by comparing current module_id with init_module_id
            module_was_replaced = False
            if hasattr(self, '_init_module_id'):
                if module_id != self._init_module_id:
                    module_was_replaced = True
                    logger.error(
                        f"[CRITICAL MODULE REPLACEMENT] {layer_id}[{proj_type}]: "
                        f"Module was replaced during checkpoint loading! "
                        f"init_module_id={self._init_module_id} != current_module_id={module_id}. "
                        f"This means the checkpoint contains buggy shared modules that overwrote the correctly initialized ones."
                    )
            
            # CRITICAL: Check if ranks were overwritten by checkpoint (even if module ID didn't change)
            # This can happen if checkpoint loads wrong ranks into existing module objects
            ranks_were_corrupted = False
            if hasattr(self, '_expected_ranks'):
                current_ranks_tuple = tuple(self.ranks) if isinstance(self.ranks, list) else self.ranks
                expected_ranks_tuple = tuple(self._expected_ranks) if isinstance(self._expected_ranks, (list, tuple)) else self._expected_ranks
                if current_ranks_tuple != expected_ranks_tuple:
                    ranks_were_corrupted = True
                    logger.error(
                        f"[CRITICAL RANKS CORRUPTION] {layer_id}[{proj_type}]: "
                        f"Module ranks were overwritten by checkpoint! "
                        f"current_ranks={self.ranks} (sum={sum(self.ranks)}) != _expected_ranks={self._expected_ranks} (sum={sum(self._expected_ranks)}). "
                        f"module_id={module_id}, VT_id={vt_id}. "
                        f"This means the checkpoint overwrote the ranks attribute even though the module object wasn't replaced."
                    )
            
            # Try to get expected ranks from model config (most reliable - can't be overwritten by checkpoint)
            expected_ranks_from_config = None
            if layer_id:
                try:
                    if hasattr(self, '_model_weakref'):
                        model = self._model_weakref()
                        if model is not None and hasattr(model, 'head_wise_ranks') and isinstance(model.head_wise_ranks, dict):
                            if layer_id in model.head_wise_ranks:
                                expected_ranks_from_config = model.head_wise_ranks[layer_id]
                except Exception as e:
                    logger.debug(f"Could not access model config to get expected ranks: {e}")
            
            # Check ranks - prefer config over stored _expected_ranks (which can be overwritten)
            if expected_ranks_from_config is not None:
                # Compare against model config (most reliable)
                expected_ranks_tuple = tuple(expected_ranks_from_config) if isinstance(expected_ranks_from_config, (list, tuple)) else expected_ranks_from_config
                current_ranks_tuple = tuple(self.ranks) if isinstance(self.ranks, list) else self.ranks
                
                if current_ranks_tuple != expected_ranks_tuple:
                    # CRITICAL: Ranks don't match model config - module was replaced!
                    ranks_were_corrupted = True  # Set flag to trigger re-initialization
                    logger.error(
                        f"[CRITICAL RANKS MISMATCH] {layer_id}[{proj_type}]: "
                        f"self.ranks={self.ranks} (sum={sum(self.ranks)}) != model.head_wise_ranks[{layer_id}]={expected_ranks_from_config} (sum={sum(expected_ranks_from_config)}). "
                        f"module_id={module_id}, VT_id={vt_id}. "
                        f"This indicates the module was replaced during checkpoint loading with wrong ranks!"
                    )
                # Ranks match config and module wasn't replaced - good!
            elif hasattr(self, '_expected_ranks'):
                # Fallback: check against stored expected ranks (less reliable - can be overwritten)
                expected_ranks_tuple = tuple(self._expected_ranks) if isinstance(self._expected_ranks, (list, tuple)) else self._expected_ranks
                current_ranks_tuple = tuple(self.ranks) if isinstance(self.ranks, list) else self.ranks
                
                if current_ranks_tuple != expected_ranks_tuple:
                    # CRITICAL: Ranks don't match expected - module was replaced or corrupted
                    logger.error(
                        f"[CRITICAL RANKS MISMATCH] {layer_id}[{proj_type}]: "
                        f"self.ranks={self.ranks} (sum={sum(self.ranks)}) != _expected_ranks={self._expected_ranks} (sum={sum(self._expected_ranks)}). "
                        f"module_id={module_id}, VT_id={vt_id}. "
                        f"This indicates the module was replaced during checkpoint loading or _expected_ranks was corrupted."
                    )
                # Ranks match stored value and module wasn't replaced - good!
            # First forward logging removed for cleaner output
            
            # Trigger global re-initialization check AFTER all rank checks are complete (lazy check)
            # This ensures we catch corruption detected via config check as well
            if (ranks_were_corrupted or module_was_replaced) and hasattr(self, '_model_weakref'):
                model = self._model_weakref()
                if model is not None and hasattr(model, '_reinitialize_corrupted_modules'):
                    # Use a flag to ensure we only trigger once
                    if not hasattr(model, '_reinit_triggered'):
                        model._reinit_triggered = True
                        logger.warning(
                            f"[CRITICAL] Detected corruption during forward pass. Triggering global re-initialization..."
                        )
                        reinit_count = model._reinitialize_corrupted_modules()
                        if reinit_count > 0:
                            logger.error(
                                f"[CRITICAL] Re-initialized {reinit_count} modules during forward pass. "
                                f"This should not happen - checkpoint loading should preserve module integrity."
                            )
        
        if not hasattr(self, '_vt_checked'):
            self._vt_checked = True
            # Get actual VT weight shape
            actual_vt_weight_shape = 'N/A'
            if hasattr(self.VT, 'weight') and hasattr(self.VT.weight, 'data'):
                actual_vt_weight_shape = self.VT.weight.data.shape
            
            # CRITICAL: Verify VT weight shape matches expected - ALWAYS check this
            # This check runs once per module instance (each layer has its own module)
            if isinstance(actual_vt_weight_shape, tuple) and len(actual_vt_weight_shape) == 2:
                if actual_vt_weight_shape[0] != expected_vt_output:
                    # This is a critical error - VT weights don't match expected dimensions
                    # Log this for ALL layers where there's a mismatch
                    logger.error(
                        f"[CRITICAL VT WEIGHT CHECK] {layer_id}[{proj_type}]: "
                        f"VT.weight.data.shape[0]={actual_vt_weight_shape[0]} != expected_output={expected_vt_output}! "
                        f"VT.weight.data.shape={actual_vt_weight_shape}, expected first dim={expected_vt_output}, "
                        f"ranks={self.ranks}, sum(ranks)={sum(self.ranks)}. "
                        f"This means VT was constructed with wrong output dimension or weights were loaded incorrectly."
                    )
                else:
                    # VT weights match - log for first few layers to help diagnose issues
                    # Check if this is one of the first few layers we want to log
                    should_log = False
                    if layer_id:
                        # Extract layer number and check if it's 0, 1, or 2
                        match = re.search(r'layers\.(\d+)', layer_id)
                        if match:
                            layer_num = int(match.group(1))
                            should_log = layer_num in [0, 1, 2]
                    
                    # VT INFO logging removed for cleaner output
        
        # VT INFO ALWAYS logging removed for cleaner output
        
        if actual_output != expected_vt_output:
            # Log VT weight shape to diagnose the issue
            vt_weight_shape = self.VT.weight.shape if hasattr(self.VT, 'weight') else 'N/A'
            logger.error(
                f"[CRITICAL VT VALIDATION] {layer_name}[{proj_type}]: VT output shape mismatch detected! "
                f"Expected {expected_vt_output} (sum(ranks)={sum(self.ranks)}), "
                f"but got {actual_output}. "
                f"VT weight shape: {vt_weight_shape}. "
                f"low_rank_latents shape: {low_rank_latents.shape}. "
                f"This suggests VT was constructed incorrectly or weights were loaded incorrectly."
            )
            logger.error(
                f"[CRITICAL] {layer_name}[{proj_type}]: VT output has wrong shape! "
                f"Expected {expected_vt_output} (sum(ranks)={sum(self.ranks)}), "
                f"but got {low_rank_latents.shape[2]}. "
                f"VT weight shape: {self.VT.weight.shape if hasattr(self.VT, 'weight') else 'N/A'}. "
                f"This suggests VT was constructed incorrectly or weights were loaded incorrectly."
            )
            # CRITICAL FIX: Always slice to expected dimensions before quantization
            # This ensures the cache stores the correct dimensions even if VT outputs wrong size
            if low_rank_latents.shape[2] != expected_vt_output:
                if low_rank_latents.shape[2] > expected_vt_output:
                    logger.warning(
                        f"[FIX] {layer_name}[{proj_type}]: Slicing low_rank_latents from "
                        f"{low_rank_latents.shape[2]} to {expected_vt_output} dimensions "
                        f"(sum(ranks)={sum(self.ranks)}). VT.weight.shape={self.VT.weight.shape if hasattr(self.VT, 'weight') else 'N/A'}"
                    )
                    low_rank_latents = low_rank_latents[:, :, :expected_vt_output]
                else:
                    # VT output has fewer dimensions than expected - this is a critical error
                    # This means VT was constructed incorrectly or weights were loaded incorrectly
                    raise RuntimeError(
                        f"[CRITICAL] {layer_name}[{proj_type}]: VT output has {low_rank_latents.shape[2]} dimensions "
                        f"but needs {expected_vt_output} (sum(ranks)={sum(self.ranks)}). "
                        f"VT.weight.shape={self.VT.weight.shape if hasattr(self.VT, 'weight') else 'N/A'}. "
                        f"This is a critical error - VT was constructed incorrectly or weights were loaded incorrectly. "
                        f"Cannot proceed as we cannot pad dimensions."
                    )
        
        # Store original dtype to preserve it after dequantization
        original_dtype = low_rank_latents.dtype
        
        # Store A(x) in cache if provided and quantization is not enabled (for standard Palu models)
        if cache_for_quantization is not None and not self.use_per_dim_quantization and not self.quantized_latents:
            if layer_name is not None and proj_type is not None:
                if 'self_attn' in layer_name:
                    # Use same expected rank as validation slice so we only store rank dimensions
                    to_store = low_rank_latents
                    if to_store.shape[2] > expected_vt_output:
                        to_store = low_rank_latents[:, :, :expected_vt_output].contiguous()
                    batch_size, seq_len, num_dims = to_store.shape
                    dummy_scales = torch.ones((batch_size, seq_len), device=to_store.device, dtype=to_store.dtype)
                    dummy_bitwidths = torch.full(
                        (seq_len, num_dims),
                        16,  # Full precision (16 bits is effectively float16)
                        dtype=torch.uint8,  # Only need 1 byte (values 0-16)
                        device=to_store.device
                    )
                    cache_for_quantization.update(
                        layer_name=layer_name,
                        proj_type=proj_type,
                        quantized_latents=to_store,  # Full-precision A(x), sliced to rank
                        scales=dummy_scales,
                        bitwidths=dummy_bitwidths,
                        n_bits=16,  # Store as int16 (full-precision path)
                    )
        
        # Step 2: Optionally quantize/dequantize A(x) if per-dimension quantization is enabled
        # This is similar to uniform quantization: quantize → dequantize → reconstruct
        if self.use_per_dim_quantization:
            # For debugging: keep a copy of full-precision A(x) so we can compare reconstruction
            orig_low_rank_for_debug = None
            _layer_num_dbg = layer_idx
            if _layer_num_dbg is None and layer_name:
                _m = re.search(r"layers\.(\d+)", layer_name)
                _layer_num_dbg = int(_m.group(1)) if _m else None
            if (
                _layer_num_dbg == 0
                and proj_type in ("k", "v")
                and low_rank_latents is not None
            ):
                # Use float32 for stable error measurement
                orig_low_rank_for_debug = low_rank_latents.detach().float()
            # One-time log pre-cache A(x) magnitude for layer 0 (compare with dequant in cache wrapper)
            _layer_num = layer_idx
            if _layer_num is None and layer_name:
                match = re.search(r'layers\.(\d+)', layer_name)
                _layer_num = int(match.group(1)) if match else None
            if _layer_num == 0 and proj_type in ('k', 'v'):
                mag_key = (0, proj_type)
                if mag_key not in _pre_cache_magnitude_logged:
                    _pre_cache_magnitude_logged.add(mag_key)
                    batch_size, seq_len, num_dims = low_rank_latents.shape
                    if seq_len > 0:
                        t0 = low_rank_latents[0, 0, :].float().flatten()
                        t_last = low_rank_latents[0, seq_len - 1, :].float().flatten()
                        logger.info(
                            f"[CACHE DEBUG] Layer 0 {proj_type} pre-cache A(x) magnitude: "
                            f"token 0   min={t0.min().item():.4f} max={t0.max().item():.4f} mean={t0.mean().item():.4f} std={t0.std().item():.4f}"
                        )
                        logger.info(
                            f"[CACHE DEBUG] Layer 0 {proj_type} pre-cache A(x) magnitude: "
                            f"token {seq_len - 1} min={t_last.min().item():.4f} max={t_last.max().item():.4f} mean={t_last.mean().item():.4f} std={t_last.std().item():.4f}"
                        )
            # Build current_bitwidths BEFORE quantize so we use the SAME bitwidths for quantize and for cache store.
            # Critical when attention_sink_tokens is used: sink overwrites first N tokens to 16 bits; if we quantized
            # with self.bitwidths (e.g. 8) but stored 16 for sink, dequant would use b=16 → ~256x magnitude error.
            seq_len_here = low_rank_latents.shape[1]
            num_dims_here = len(self.base_bitwidths)
            sink = getattr(self, "attention_sink_tokens", 0)
            recent_tokens = getattr(self, "recent_tokens", 0)
            # Always start from base bitwidths for all tokens.  Sink and recent-window
            # positions are promoted to 16 bits below.  Initialising to 16 instead would mean
            # self.base_bitwidths is never applied, leaving quantisation at 16-bit for every
            # token — making tokenwise appear nearly lossless while channelwise still clips
            # values exceeding the static calibration alpha, a spurious performance gap.
            base_bw = self.base_bitwidths.to(device=low_rank_latents.device, dtype=torch.uint8)
            current_bitwidths = base_bw.unsqueeze(0).expand(seq_len_here, num_dims_here).clone()
            if sink > 0 and current_bitwidths.shape[0] > 0:
                if current_bitwidths.shape[0] == 1 and cache_for_quantization is not None:
                    current_cache_len = cache_for_quantization.get_seq_length(layer_name)
                    if current_cache_len < sink:
                        current_bitwidths[:, :] = 16
                else:
                    n_sink = min(sink, current_bitwidths.shape[0])
                    current_bitwidths[:n_sink, :] = 16
            # Sliding recent window: recent tokens at 16 bits (all dims, including 0-bit
            # ones from base_bw — fair for KIVI comparison: recent tokens keep full precision;
            # 0-bit dims are only evicted when the sliding step requantizes to base_bw).
            if recent_tokens > 0 and current_bitwidths.shape[0] > 0:
                if current_bitwidths.shape[0] == 1:
                    # New token (decode): always in recent window until sliding step runs
                    current_bitwidths[:, :] = 16
                elif current_bitwidths.shape[0] > 1:
                    # Prefill or multi-token: last recent_tokens positions are the recent window.
                    # Use actual recent_tokens count (not n_recent_full which may be 0 on first prefill).
                    n_recent = min(recent_tokens, current_bitwidths.shape[0])
                    current_bitwidths[-n_recent:, :] = 16
            # When recent_tokens=0: no sliding window — truncate to base right after sink; every new token gets base
            elif recent_tokens == 0 and sink > 0 and current_bitwidths.shape[0] > 0:
                n_sink = min(sink, current_bitwidths.shape[0])
                if current_bitwidths.shape[0] == 1:
                    # Decode: new token is 16 only if still in sink (cache_len < sink), else base
                    current_cache_len = cache_for_quantization.get_seq_length(layer_name) if cache_for_quantization else 0
                    if current_cache_len >= sink:
                        base_bw = self.base_bitwidths.to(device=current_bitwidths.device, dtype=torch.uint8)
                        if base_bw.dim() == 1:
                            current_bitwidths[:, :] = base_bw.unsqueeze(0).expand(1, -1)
                        else:
                            current_bitwidths[:, :] = 8  # fallback
                elif current_bitwidths.shape[0] > n_sink:
                    base_bw = self.base_bitwidths.to(device=current_bitwidths.device, dtype=torch.uint8)
                    if base_bw.dim() == 1 and base_bw.numel() == num_dims_here:
                        current_bitwidths[n_sink:, :] = base_bw.unsqueeze(0).expand(
                            current_bitwidths.shape[0] - n_sink, -1
                        )
                    else:
                        n_bits = 8
                        if getattr(self, "latent_quantizer", None) is not None and hasattr(self.latent_quantizer, "n_bits"):
                            n_bits = self.latent_quantizer.n_bits
                        current_bitwidths[n_sink:, :] = n_bits
            # Use cached bitwidths only when we have at least (sink + recent_tokens) — i.e. we may have
            # already run the sliding step. Below that, everything must stay 16 bits (no copy from cache).
            # When recent_tokens=0 we already set sink + base above; do not overwrite with cache.
            min_len_for_truncated = (sink + (recent_tokens or 0)) if (sink or (recent_tokens or 0)) else 0
            copied_from_cache = False
            cached_bw_shape = None
            if (cache_for_quantization is not None and current_bitwidths.shape[0] > 1 and min_len_for_truncated > 0
                    and recent_tokens > 0):
                cached_bw = cache_for_quantization.get_bitwidths(layer_name, proj_type)
                if cached_bw is not None:
                    cached_bw_shape = tuple(cached_bw.shape)
                if cached_bw is not None and cached_bw.shape[0] >= min_len_for_truncated:
                    if cached_bw.shape[0] == current_bitwidths.shape[0] - 1:
                        # Decode: we have 1 new token; copy bitwidths for existing positions from cache
                        current_bitwidths[:-1, :] = cached_bw.to(current_bitwidths.device)
                        copied_from_cache = True
                    elif cached_bw.shape[0] == current_bitwidths.shape[0]:
                        # Prompt or full update: use cache as source of truth (has truncated zone from sliding step)
                        current_bitwidths = cached_bw.to(current_bitwidths.device)
                        copied_from_cache = True
            # Debug: when seq_len is 17-22 and layer 0, log why token 9 might be 8-bit (should be 16 until 24 tokens)
            if (
                layer_name is not None
                and "layers.0" in layer_name
                and "self_attn" in layer_name
                and proj_type in ("k", "v")
                and 17 <= current_bitwidths.shape[0] <= 22
            ):
                key = (layer_name, proj_type, current_bitwidths.shape[0])
                if key not in _per_dim_bitwidth_debug_logged:
                    _per_dim_bitwidth_debug_logged.add(key)
                    tok9 = current_bitwidths[9].float() if current_bitwidths.shape[0] > 9 else None
                    tok9_str = f"min={tok9.min().item():.1f} max={tok9.max().item():.1f} mean={tok9.mean().item():.2f}" if tok9 is not None else "n/a"
                    logger.info(
                        f"[PER-DIM BITWIDTH DEBUG] Layer 0 {proj_type} seq_len={current_bitwidths.shape[0]}: "
                        f"min_len_for_truncated={min_len_for_truncated} cached_bw_shape={cached_bw_shape} "
                        f"copied_from_cache={copied_from_cache} | token 9: {tok9_str}"
                    )
            if current_bitwidths.device != low_rank_latents.device:
                current_bitwidths = current_bitwidths.to(low_rank_latents.device)
            # Debug: decode path (1 token) — confirm we pass 16 for the new token (layer 0, once per proj)
            if (
                current_bitwidths.shape[0] == 1
                and layer_name is not None
                and "layers.0" in layer_name
                and "self_attn" in layer_name
                and proj_type in ("k", "v")
            ):
                _key = ("decode_1tok", layer_name, proj_type)
                if _key not in _per_dim_bitwidth_debug_logged:
                    _per_dim_bitwidth_debug_logged.add(_key)
                    b = current_bitwidths[0].float()
                    logger.info(
                        f"[PER-DIM BITWIDTH DEBUG] Layer 0 {proj_type} decode (1 token): "
                        f"min={b.min().item():.1f} max={b.max().item():.1f} mean={b.mean().item():.2f} (expect 16)"
                    )
            # Quantize using the same bitwidths we will store (critical for correct dequant magnitude)
            _token_offset = 0
            if self.scaling_type == "channel_group" and cache_for_quantization is not None and layer_name is not None:
                try:
                    _token_offset = cache_for_quantization.get_seq_length(layer_name) or 0
                except Exception:
                    _token_offset = 0
            quantized_latents, scales = self.quantize_with_per_dim_bitwidths(
                low_rank_latents, bitwidths_per_token=current_bitwidths, token_offset=_token_offset
            )
            
            # Store in cache if provided
            # Only cache attention layers (k_proj and v_proj in self_attn), not other layers like lm_head
            if cache_for_quantization is not None:
                if layer_name is None or proj_type is None:
                    logger.warning("cache_for_quantization provided but layer_name or proj_type missing. Not storing quantized values.")
                elif 'self_attn' not in layer_name:
                    # Skip non-attention layers (e.g., lm_head, mlp layers)
                    pass
                else:
                    # Use max bitwidth for cache storage dtype (int8 or int16)
                    max_bits = int(current_bitwidths.max().item()) if current_bitwidths.numel() else 8
                    # One-time trace: writer scale/bitwidth for k vs v (compare with reader SCALE_BW_TRACE)
                    if layer_name and "layers.0" in layer_name and proj_type in ("k", "v"):
                        key = (0, proj_type, "writer_trace")
                        if key not in _writer_scale_trace_logged:
                            _writer_scale_trace_logged.add(key)
                            st = getattr(self, "scaling_type", None)
                            logger.info(
                                f"[SCALE_BW_TRACE] Layer 0 {proj_type} (writer): scaling_type={st!r} "
                                f"scales.shape={tuple(scales.shape)} bitwidths.shape={tuple(current_bitwidths.shape)} "
                                f"num_dims={current_bitwidths.shape[1]}"
                            )
                    cache_for_quantization.update(
                        layer_name=layer_name,
                        proj_type=proj_type,
                        quantized_latents=quantized_latents,
                        scales=scales.float(),  # store as float32 so wrapper dequant matches forward dequant
                        bitwidths=current_bitwidths,
                        channelwise_scaling=self.channelwise_scaling if self.channelwise_scaling is not None else None,
                        n_bits=max_bits,
                    )
            
            # Dequantize back to A(x) (with quantization error); use same bitwidths as quantize.
            # CRITICAL: use float32 scales so the forward's dequant matches the cache wrapper's
            # dequant exactly.  The wrapper always reads scales as float32; if the forward uses
            # float16 scales the intermediate `2*alpha` is rounded in float16 before the division,
            # producing a slightly different stepsize.  With 16-bit quantization Q can reach 32767,
            # amplifying that tiny stepsize mismatch into a visible reconstruction error that
            # accumulates across decode steps and eventually garbles the output.
            low_rank_latents = self.dequantize_with_per_dim_bitwidths(
                quantized_latents, scales.float(), bitwidths_per_token=current_bitwidths
            )
            # Optional debug: compare reconstruction error vs a static 6-bit scheme on truncated tokens
            if (
                orig_low_rank_for_debug is not None
                and cache_for_quantization is not None
                and layer_name is not None
                and "self_attn" in layer_name
                and proj_type in ("k", "v")
            ):
                try:
                    sink = getattr(self, "attention_sink_tokens", 0)
                    recent_tokens = getattr(self, "recent_tokens", 0)
                    seq_len_dbg = low_rank_latents.shape[1]
                    # Compare as soon as we have a full window (seq_len >= sink + recent_tokens).
                    if recent_tokens > 0 and seq_len_dbg >= sink + recent_tokens:
                        n_recent_full = cache_for_quantization.get_n_recent_full()
                        start_trunc = max(0, seq_len_dbg - recent_tokens)
                        end_trunc = seq_len_dbg - max(0, n_recent_full)
                        if end_trunc > start_trunc:
                            token_indices = [start_trunc, end_trunc - 1]
                            token_indices = [i for i in token_indices if 0 <= i < seq_len_dbg]
                            if token_indices:
                                # Build a static-6bit scheme: keep 16 where current_bitwidths==16, else 6
                                static_bw = current_bitwidths.clone()
                                static_bw = static_bw.to(dtype=torch.uint8)
                                mask_non16 = static_bw != 16
                                static_bw[mask_non16] = 6
                                # Quantize/dequantize with static-6bit starting from original A(x)
                                q6, s6 = self.quantize_with_per_dim_bitwidths(
                                    orig_low_rank_for_debug.to(quantized_latents.device, dtype=quantized_latents.dtype),
                                    bitwidths_per_token=static_bw,
                                )
                                rec6 = self.dequantize_with_per_dim_bitwidths(
                                    q6, s6, bitwidths_per_token=static_bw
                                ).float()
                                # Current adaptive reconstruction (already dequantized above)
                                rec_adapt = low_rank_latents.detach().float()
                                for tok_idx in token_indices:
                                    x0 = orig_low_rank_for_debug[0, tok_idx, :].flatten()
                                    xa = rec_adapt[0, tok_idx, :].flatten()
                                    x6 = rec6[0, tok_idx, :].flatten()
                                    err_adapt = (x0 - xa).pow(2).mean().sqrt().item()
                                    err_6 = (x0 - x6).pow(2).mean().sqrt().item()
                                    bw_row = current_bitwidths[tok_idx].float()
                                    logger.info(
                                        f"[ADAPTIVE_ERROR_DEBUG] {layer_name}[{proj_type}] token {tok_idx}: "  # noqa: E501
                                        f"bw_min={bw_row.min().item():.1f} bw_max={bw_row.max().item():.1f} "  # noqa: E501
                                        f"err_adapt_L2={err_adapt:.6f} err_static6_L2={err_6:.6f}"
                                    )
                except Exception as e:
                    logger.debug(f"[ADAPTIVE_ERROR_DEBUG] comparison failed for {layer_name}[{proj_type}]: {e}")
            # Convert back to original dtype to match reconstruction matrices
            low_rank_latents = low_rank_latents.to(original_dtype)
            # Now low_rank_latents is A(x) with quantization noise, ready for reconstruction
        
        # Step 3: Handle uniform quantization (if enabled)
        elif self.quantized_latents:
            # Log for ALL layers (not just attention) to debug why only one layer is cached
            # Track which layers are using uniform quantization path
            if not hasattr(HeadwiseLowRankModule, '_uniform_quant_layers_seen'):
                HeadwiseLowRankModule._uniform_quant_layers_seen = set()
            
            # Only log the first time we see a layer_name - and only for first few layers
            if layer_name is not None:
                if layer_name not in HeadwiseLowRankModule._uniform_quant_layers_seen:
                    HeadwiseLowRankModule._uniform_quant_layers_seen.add(layer_name)
                    # Only log for first few layers to reduce noise
                    if 'layers.0' in layer_name or 'layers.1' in layer_name:
                        logger.debug(f"Uniform quantization path: layer_name={layer_name}, proj_type={proj_type}, cache_for_quantization={'provided' if cache_for_quantization is not None else 'None'}")
                # Don't log subsequent calls to reduce noise
            else:
                # Log if layer_name is None (this shouldn't happen if patching works correctly)
                # Only warn for attention layers (k_proj, v_proj) to avoid spam from other modules
                # that might have quantized_latents set but aren't patched (e.g., q_proj, o_proj)
                if not hasattr(HeadwiseLowRankModule, '_none_layer_name_warnings'):
                    HeadwiseLowRankModule._none_layer_name_warnings = set()
                # Only warn once per module instance to avoid spam
                module_id = id(self)
                if module_id not in HeadwiseLowRankModule._none_layer_name_warnings:
                    HeadwiseLowRankModule._none_layer_name_warnings.add(module_id)
                    # Only warn if this looks like an attention layer (has k_proj or v_proj in the module's name)
                    # We can't check layer_name since it's None, but we can check if this module
                    # is likely an attention layer by checking if it's in a model that has attention layers
                    # For now, just reduce the warning level to DEBUG to avoid spam
                    logger.debug(f"Uniform quantization path called with layer_name=None, proj_type={proj_type}. This may be a non-attention layer (q_proj, o_proj, etc.) that has quantized_latents set but isn't patched.")
            
            # Check if we have cached quantized values from previous tokens
            cached_quantized_integers = None
            cached_scales_dict = None
            cached_bitwidths = None
            is_layer_0 = layer_name and 'layers.0' in layer_name
            if cache_for_quantization is not None and layer_name is not None and proj_type is not None:
                cached_quantized_integers = cache_for_quantization.get_quantized(layer_name, proj_type)
                cached_scales_dict = cache_for_quantization.get_scales_dict(layer_name, proj_type)
                cached_bitwidths = cache_for_quantization.get_bitwidths(layer_name, proj_type)
                
                # Cache retrieve logging removed for cleaner output
            
            # Uniform quantization: quantize A(x) with uniform bitwidth (e.g., 8 bits)
            # Store original for cache (before quantization)
            original_latents = low_rank_latents.clone()
        
            # NEW APPROACH: Quantize to actual integers and store in cache
            # 1. Quantize A(x) to get actual quantized integers + scales_dict
            original_shape = low_rank_latents.shape
            # Debug: log shape mismatch if detected
            # VT outputs sum(ranks) dimensions, regardless of num_heads_per_group
            # The rank vector in config represents ranks per head, but VT was constructed with sum(ranks)
            expected_total_ranks = sum(self.ranks)
            if original_shape[2] != expected_total_ranks:
                logger.error(
                    f"[SHAPE MISMATCH] {layer_name}[{proj_type}]: low_rank_latents shape={original_shape}, "
                    f"but expected last dim={expected_total_ranks} (sum of ranks={sum(self.ranks)}, "
                    f"num_groups={self.num_groups}, num_heads_per_group={self.num_heads_per_group}). "
                    f"This suggests low_rank_latents has the wrong shape before quantization! "
                    f"VT output shape should be (batch, seq, {expected_total_ranks}) but got {original_shape}"
                )
                # Try to fix by slicing to expected size if we have too many dimensions
                if original_shape[2] > expected_total_ranks:
                    logger.warning(f"  Attempting to fix by slicing low_rank_latents from {original_shape[2]} to {expected_total_ranks} dimensions")
                    low_rank_latents = low_rank_latents[:, :, :expected_total_ranks]
                    original_shape = low_rank_latents.shape
                else:
                    raise RuntimeError(
                        f"Cannot fix shape mismatch: low_rank_latents has {original_shape[2]} dimensions "
                        f"but needs {expected_total_ranks} (sum of ranks={self.ranks})"
                    )
            quantized_integers_new, scales_dict_new = self.quantize_latent_to_integers(low_rank_latents)
            
            # New token A(x) logging removed for cleaner output
            
            # Verify shape is preserved
            if quantized_integers_new.shape != original_shape:
                logger.error(
                    f"Shape mismatch after quantization for {layer_name}: original={original_shape}, "
                    f"quantized={quantized_integers_new.shape}. Using original latents."
                )
                # Fallback to fake quantization
                quantized_integers_new = self.quantize_latent(low_rank_latents)
                scales_dict_new = {}
            
            # Concatenate cached + new quantized integers for full sequence
            if cached_quantized_integers is not None:
                # Validate cached values have correct shape
                # They should already be correct if we fixed low_rank_latents at the source
                expected_total_ranks = sum(self.ranks)
                if cached_quantized_integers.shape[2] != expected_total_ranks:
                    if cached_quantized_integers.shape[2] > expected_total_ranks:
                        logger.warning(
                            f"[FIX] {layer_name}[{proj_type}]: Cached quantized_integers has wrong shape. "
                            f"Slicing from {cached_quantized_integers.shape[2]} to {expected_total_ranks} dimensions. "
                            f"This suggests the cache was created before the fix."
                        )
                        cached_quantized_integers = cached_quantized_integers[:, :, :expected_total_ranks]
                    else:
                        logger.error(
                            f"[ERROR] {layer_name}[{proj_type}]: cached_quantized_integers has fewer dimensions "
                            f"({cached_quantized_integers.shape[2]}) than expected ({expected_total_ranks}). "
                            f"Cannot fix by slicing. Using only new values."
                        )
                        # Use only new values if cached values are wrong
                        cached_quantized_integers = None
                
            if cached_quantized_integers is not None:
                # Debug logging for layer 0 to verify we're using cached values
                if is_layer_0:
                    # Check if cached values have duplicate last token
                    cached_last_token = cached_quantized_integers[0, -1, :5].tolist() if cached_quantized_integers.shape[2] >= 5 else cached_quantized_integers[0, -1, :].tolist()
                    cached_second_last_token = cached_quantized_integers[0, -2, :5].tolist() if cached_quantized_integers.shape[1] >= 2 and cached_quantized_integers.shape[2] >= 5 else None
                    new_token_sample = quantized_integers_new[0, -1, :5].tolist() if quantized_integers_new.shape[2] >= 5 else quantized_integers_new[0, -1, :].tolist()
                    
                    # Cache concat debug logging removed for cleaner output
                
                # Concatenate cached integers + new integers (cache may be int8/int16; cast to float for concat with quantized_integers_new)
                if not cached_quantized_integers.is_floating_point():
                    cached_quantized_integers = cached_quantized_integers.float()
                quantized_integers_full = torch.cat([cached_quantized_integers, quantized_integers_new], dim=1)
                # Cache concat verify logging removed for cleaner output
                
                # Concatenate scales_dict (merge cached + new scales_dict)
                if cached_scales_dict is not None:
                    # Merge scales_dict: concatenate scales and base for each group
                    merged_scales_dict = {}
                    for group_idx in scales_dict_new:
                        if group_idx in cached_scales_dict:
                            cached_scales, cached_base = cached_scales_dict[group_idx]
                            new_scales, new_base = scales_dict_new[group_idx]
                            # Concatenate along batch*seq dimension (dim=0)
                            merged_scales = torch.cat([cached_scales, new_scales], dim=0)
                            merged_base = torch.cat([cached_base, new_base], dim=0)
                            merged_scales_dict[group_idx] = (merged_scales, merged_base)
                        else:
                            merged_scales_dict[group_idx] = scales_dict_new[group_idx]
                    scales_dict_full = merged_scales_dict
                else:
                    scales_dict_full = scales_dict_new
            else:
                # No cached values, use only new token
                quantized_integers_full = quantized_integers_new
                scales_dict_full = scales_dict_new
            
            # Store in cache if provided (for uniform quantization cache)
            # Only cache attention layers (k_proj and v_proj in self_attn), not other layers like lm_head
            if cache_for_quantization is not None:
                # Debug: log when cache is provided but layer_name/proj_type are missing
                if layer_name is None or proj_type is None:
                    # For uniform quantization, we might not have layer_name/proj_type passed
                    # Try to infer from the module's name if available
                    if hasattr(self, '_module_name') and hasattr(self, '_proj_type'):
                        layer_name = self._module_name
                        proj_type = self._proj_type
                    else:
                        # Only log for attention layers to reduce noise
                        if hasattr(self, '_module_name') and 'self_attn' in str(getattr(self, '_module_name', '')):
                            logger.debug(f"cache_for_quantization provided but layer_name or proj_type missing for {getattr(self, '_module_name', 'unknown')}. Not storing quantized values.")
                        layer_name = None
                        proj_type = None
                
                if layer_name is not None and proj_type is not None:
                    if 'self_attn' not in layer_name:
                        # Skip non-attention layers (e.g., lm_head, mlp layers)
                        if 'self_attn' not in str(layer_name):  # Extra check to avoid false positives
                            logger.debug(f"Skipping non-attention layer: {layer_name}[{proj_type}]")
                        pass
                    else:
                        # Debug: log which layers are being cached (track globally across all modules)
                        # Use a class-level set to track across all instances
                        if not hasattr(HeadwiseLowRankModule, '_global_cached_layers'):
                            HeadwiseLowRankModule._global_cached_layers = set()
                        
                        cache_key = f"{layer_name}[{proj_type}]"
                        # Only log first time caching to reduce noise - and only for first few layers
                        if cache_key not in HeadwiseLowRankModule._global_cached_layers:
                            HeadwiseLowRankModule._global_cached_layers.add(cache_key)
                            # Only log for first few layers
                            if 'layers.0' in cache_key or 'layers.1' in cache_key:
                                logger.debug(f"First time caching {cache_key} (total unique layers cached: {len(HeadwiseLowRankModule._global_cached_layers)})")
                        # For uniform quantization, we need to extract the bitwidth and scales
                        # The quantizer uses n_bits, but we need to create scales and bitwidths for the cache
                        if self.latent_quantizer is None:
                            logger.warning(f"latent_quantizer is None for {layer_name}[{proj_type}]. Cannot cache quantized values.")
                        else:
                            n_bits = self.latent_quantizer.n_bits
                            batch_size, seq_len, num_dims = quantized_integers_full.shape
                            
                            # Before cache store logging removed for cleaner output
                            
                            # CRITICAL: Validate that quantized_integers_full has the correct shape
                            # It should match sum(ranks) after our earlier fixes
                            # ALWAYS slice to expected dimensions before storing in cache
                            expected_total_ranks = sum(self.ranks)
                            if num_dims != expected_total_ranks:
                                logger.error(
                                    f"[CRITICAL CACHE] {layer_name}[{proj_type}]: About to store quantized_integers with wrong shape! "
                                    f"shape={quantized_integers_full.shape}, expected last dim={expected_total_ranks} "
                                    f"(sum(ranks)={sum(self.ranks)}). This should have been fixed earlier. "
                                    f"VT weight shape: {self.VT.weight.shape if hasattr(self.VT, 'weight') else 'N/A'}"
                                )
                                # Try to fix by slicing if we have too many dimensions
                                if num_dims > expected_total_ranks:
                                    logger.warning(
                                        f"  Fixing by slicing quantized_integers_full from {num_dims} to {expected_total_ranks} dimensions"
                                    )
                                    quantized_integers_full = quantized_integers_full[:, :, :expected_total_ranks]
                                    # Also need to update scales_dict to match
                                    # For now, we'll slice the scales_dict groups, but this is a hack
                                    # The real fix is to ensure VT outputs the correct dimensions
                                    num_dims = expected_total_ranks
                                else:
                                    # Cannot fix - this is a critical error
                                    # This means VT is outputting fewer dimensions than needed
                                    # This will corrupt the cache, so we MUST raise an error
                                    error_msg = (
                                        f"[CRITICAL CACHE ERROR] {layer_name}[{proj_type}]: "
                                        f"Cannot store quantized_integers: has {num_dims} dimensions "
                                        f"but needs {expected_total_ranks} (sum(ranks)={sum(self.ranks)}). "
                                        f"VT.weight.shape={self.VT.weight.shape if hasattr(self.VT, 'weight') else 'N/A'}. "
                                        f"This means VT was constructed incorrectly or weights were loaded incorrectly. "
                                        f"Aborting to prevent cache corruption."
                                    )
                                    logger.error(error_msg)
                                    raise RuntimeError(error_msg)
                            
                            # Create uniform bitwidths tensor (same bitwidth for all dimensions and tokens)
                            current_bitwidths = torch.full(
                                (seq_len, num_dims),
                                n_bits,
                                dtype=torch.uint8,  # Only need 1 byte (values 0-16)
                                device=quantized_integers_full.device
                            )
                            # Keys (k_proj): apply attention sink and sliding recent same as values
                            sink = getattr(self, "attention_sink_tokens", 0)
                            recent_tokens = getattr(self, "recent_tokens", 0)
                            # First time we quantize is when seq_len > sink + recent_tokens; otherwise all 16. Only for prompt (seq_len > 1).
                            if seq_len > 1 and (sink + (recent_tokens or 0)) > 0 and seq_len <= (sink + (recent_tokens or 0)):
                                current_bitwidths = torch.full(
                                    (seq_len, num_dims), 16, dtype=torch.uint8,
                                    device=quantized_integers_full.device
                                )
                            elif sink > 0 and current_bitwidths.shape[0] > 0:
                                n_sink = min(sink, current_bitwidths.shape[0])
                                current_bitwidths[:n_sink, :] = 16
                            if recent_tokens > 0 and cache_for_quantization is not None and current_bitwidths.shape[0] > 0:
                                n_recent_full = cache_for_quantization.get_n_recent_full()
                                if n_recent_full > 0:
                                    current_bitwidths[-n_recent_full:, :] = 16

                            # Store only the NEW token in cache (let update() handle concatenation)
                            # CRITICAL: Pass only quantized_integers_new, not quantized_integers_full!
                            # This ensures update() sees new_seq_len=1 and concatenates instead of replacing.
                            new_token_seq_len = quantized_integers_new.shape[1]  # Should be 1 during generation
                            
                            # Create bitwidths for just the new token (same idea as per-dim: use prev token's bitwidth so next step we can reduce).
                            current_cache_len = cache_for_quantization.get_seq_length(layer_name) if cache_for_quantization else 0
                            sink = getattr(self, "attention_sink_tokens", 0)
                            recent_tokens = getattr(self, "recent_tokens", 0)
                            min_len_for_truncated = sink + recent_tokens  # e.g. 24: sliding step runs at this length
                            # 16 bits if: in sink, in recent window, or cache not yet long enough for truncation (align with per-dim writer)
                            use_16_for_new = (
                                (sink > 0 and current_cache_len < sink)
                                or (recent_tokens > 0)
                                or (current_cache_len < min_len_for_truncated)
                            )
                            # When recent_tokens=0: after sink use base bitwidth for new token, not cached (cached may be 16)
                            if recent_tokens == 0 and not use_16_for_new and self.base_bitwidths is not None:
                                base_bw = self.base_bitwidths.to(quantized_integers_new.device)
                                if base_bw.dim() == 1 and base_bw.numel() == num_dims:
                                    new_token_bitwidths = base_bw.unsqueeze(0).expand(
                                        new_token_seq_len, -1
                                    ).to(torch.uint8)
                                else:
                                    new_token_bitwidths = torch.full(
                                        (new_token_seq_len, num_dims), n_bits, dtype=torch.uint8,
                                        device=quantized_integers_new.device)
                            else:
                                new_token_bitwidths = torch.full(
                                    (new_token_seq_len, num_dims),
                                    16 if use_16_for_new else n_bits,
                                    dtype=torch.uint8,
                                    device=quantized_integers_new.device
                                )
                            
                            # Cache update logging removed for cleaner output
                            
                            # For prompt (new_token_seq_len > 1): pass full sequence bitwidths with sink + recent overlay.
                            # For decode (new_token_seq_len == 1): pass only new token bitwidths (update() concatenates).
                            bitwidths_to_store = current_bitwidths if new_token_seq_len > 1 else new_token_bitwidths
                            cache_for_quantization.update(
                                layer_name=layer_name,
                                proj_type=proj_type,
                                quantized_latents=quantized_integers_new,  # NEW token only for decode; full prompt for prompt
                                scales_dict=scales_dict_new,  # NEW token's scales_dict only for decode; full for prompt
                                bitwidths=bitwidths_to_store,  # full sequence (sink + recent) for prompt; new token only for decode
                                n_bits=self.latent_quantizer.n_bits if self.latent_quantizer else 8,
                            )
                            
                            # Cache update verify logging removed for cleaner output
            
            # NEW APPROACH: Reconstruct from full cached sequence
            # 1. Dequantize from cached quantized integers (full sequence: cached + new)
            # 2. Reconstruct BA(x) from full sequence
            # 3. Update transformers' past_key_values with full reconstructed sequence
            # 4. Return only new token's output to transformers
            #
            # This ensures consistency: all BA(x) values are computed from quantized A(x),
            # and transformers' past_key_values is updated to use our quantized values.
            # Reconstruct flow debug logging removed for cleaner output
            dequantized_latents_full = self.dequantize_latent_from_integers(quantized_integers_full, scales_dict_full)
            # Dequant returns float32; reconstruct uses U[i] which may be float16 — match module dtype
            module_dtype = next((p.dtype for p in self.parameters()), torch.float32)
            if dequantized_latents_full.dtype != module_dtype:
                dequantized_latents_full = dequantized_latents_full.to(module_dtype)

            # Recent-tokens window: replace the last min(recent_tokens, len(original_latents)) positions
            # with the original FP16 latents, giving lossless reconstruction for recent tokens.
            # During prefill: original_latents covers all N new tokens → patches last N_recent positions.
            # During decode (seq_len=1): original_latents has 1 token → patches just the new token.
            # This mirrors KIVI's residual_length: body at n_bits, recent at full precision.
            _recent_t = getattr(self, "recent_tokens", 0)
            if _recent_t > 0 and original_latents is not None:
                _n_patch = min(_recent_t, original_latents.shape[1])
                if _n_patch > 0:
                    dequantized_latents_full = dequantized_latents_full.clone()
                    dequantized_latents_full[:, -_n_patch:, :] = original_latents[:, -_n_patch:, :].to(module_dtype)

            # Reconstruct BA(x) from full sequence
            outputs_full = self.reconstruct(dequantized_latents_full)
            
            # CRITICAL: Do NOT directly update past_key_values here!
            # Transformers will call update() separately with the new token's values.
            # Our QuantizedDynamicCache.update() will reconstruct the full sequence from QuantizedCache
            # when transformers calls it. This ensures consistency and avoids double updates.
            #
            # The flow is:
            # 1. We update QuantizedCache with quantized integers (full sequence) - already done above
            # 2. We return only the new token's output to transformers
            # 3. Transformers calls update() with the new token's key/value states
            # 4. Our QuantizedDynamicCache.update() reconstructs the full sequence from QuantizedCache
            # 5. Transformers accesses the cache via __getitem__() to get the full sequence
            
            # Extract only the new token's output (transformers expects only the new token)
            # The new token is at the end of the sequence
            new_token_start_idx = outputs_full.shape[1] - quantized_integers_new.shape[1]
            outputs = outputs_full[:, new_token_start_idx:, :]
            
            return outputs
        
        # Step 4: Reconstruct to BA(x) (for non-quantization path)
        # Forward reconstruct logging removed for cleaner output
        outputs = self.reconstruct(low_rank_latents)
        
        return outputs
    
    
    def project_to_latent(self, hidden_states:  torch.Tensor):
        """
            hidden_states: Tensor of shape (batch_size, seq_len, in_features)
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                "Input tensor should have dimension 3."
            )
        hidden_states = self.VT(hidden_states)
        """
            hidden_states: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        """
        return hidden_states

    def _build_batched_weight(self):
        """Stack U weights for the uniform-rank batched matmul path.

        _W_batched: (num_groups, group_dim, rank)
        _b_batched: (num_groups, group_dim) or None
        Non-persistent so they follow .to(device/dtype) without polluting the state_dict.
        """
        W = torch.stack([u.weight for u in self.U])  # (G, group_dim, rank)
        self.register_buffer('_W_batched', W, persistent=False)
        if self.U[0].bias is not None:
            b = torch.stack([u.bias for u in self.U])  # (G, group_dim)
            self.register_buffer('_b_batched', b, persistent=False)
        else:
            self._b_batched = None

    def reconstruct(self, low_rank_latents: torch.Tensor):
        """
            low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
            If num_heads_per_group is set, the rank vector represents ranks per head,
            so the actual dimension per group is ranks[i] * num_heads_per_group
        """
        batch_size, seq_len, total_dims_in_latents = low_rank_latents.shape
        expected_total_ranks = sum(self.ranks)

        if total_dims_in_latents != expected_total_ranks:
            raise ValueError(
                f"Shape mismatch in reconstruct: low_rank_latents last dimension ({total_dims_in_latents}) "
                f"does not match expected ({expected_total_ranks}).\n"
                f"  low_rank_latents shape: {low_rank_latents.shape}\n"
                f"  module ranks: {self.ranks}, module num_groups: {self.num_groups}, "
                f"num_heads_per_group: {self.num_heads_per_group}"
            )

        # Fast path: all groups share the same rank → one batched matmul.
        # (batch*seq, G, r) x (G, group_dim, r).T → (batch*seq, G, group_dim)
        if self._uniform_rank is not None:
            if self._W_batched is None:
                self._build_batched_weight()
            r = self._uniform_rank
            x = low_rank_latents.reshape(batch_size * seq_len, self.num_groups, r)
            x = x.to(self._W_batched.dtype)
            out = torch.einsum('bgr,gdr->bgd', x, self._W_batched)
            if self._b_batched is not None:
                out = out + self._b_batched
            return out.reshape(batch_size, seq_len, self.out_features)

        # Fallback: variable ranks across groups.
        outputs = []
        total_ranks = 0
        for i in range(self.num_groups):
            group_rank = self.ranks[i]
            low_rank_latent = low_rank_latents[:, :, total_ranks: total_ranks+group_rank]
            outputs.append(self.U[i](low_rank_latent))
            total_ranks += group_rank
        return torch.cat(outputs, dim=-1)
    
    
    def quantize_latent(self, low_rank_latents: torch.Tensor):
        """
        Quantize latents and return dequantized values (for immediate reconstruction).
        This is the "fake quantization" path - quantizes and immediately dequantizes.
        
            low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        Returns: Dequantized float values with quantization error
        """
        assert self.latent_quantizer is not None, "Latent quantizer is not initialized."
        original_shape = low_rank_latents.shape
        fake_quantized_low_rank_latents = []
        total_ranks = 0
        for i in range(self.num_groups):
            low_rank_latent = low_rank_latents[:, :, total_ranks: total_ranks+self.ranks[i]]
            quantized_group = self.latent_quantizer(low_rank_latent)
            # Verify shape is preserved for each group
            if quantized_group.shape != low_rank_latent.shape:
                logger.error(
                    f"Shape mismatch in quantize_latent group {i}: "
                    f"input={low_rank_latent.shape}, output={quantized_group.shape}, "
                    f"input_numel={low_rank_latent.numel()}, output_numel={quantized_group.numel()}"
                )
            fake_quantized_low_rank_latents.append(quantized_group)
            total_ranks += self.ranks[i]

        """
            fake_quantized_low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ...)
        """
        result = torch.cat(fake_quantized_low_rank_latents, dim=-1)
        # Final shape check
        if result.shape != original_shape:
            logger.error(
                f"Shape mismatch in quantize_latent: original={original_shape}, "
                f"result={result.shape}, original_numel={low_rank_latents.numel()}, "
                f"result_numel={result.numel()}"
            )
        return result
    
    def quantize_latent_to_integers(self, low_rank_latents: torch.Tensor):
        """
        Quantize latents and return actual quantized integers + scales.
        This is for storing in QuantizedCache.
        
        low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        Returns: (quantized_integers, scales_dict)
            - quantized_integers: Tensor of shape (batch_size, seq_len, r1 + r2 + ...) with integer values
            - scales_dict: Dict mapping group_idx to (scales, base) tuples
                scales: (batch_size * seq_len, 1) - scale per group
                base: (batch_size * seq_len, 1) - base offset per group
        """
        assert self.latent_quantizer is not None, "Latent quantizer is not initialized."
        from palu.model.modules.quant import quantize_tensor
        
        original_shape = low_rank_latents.shape
        batch_size, seq_len, total_dims = original_shape
        
        quantized_integers_list = []
        scales_dict = {}
        
        total_ranks = 0
        for group_idx in range(self.num_groups):
            group_rank = self.ranks[group_idx]
            low_rank_latent = low_rank_latents[:, :, total_ranks: total_ranks + group_rank]
            
            # Reshape to (batch_size * seq_len, group_rank) for quantization
            flat_latent = low_rank_latent.reshape(-1, group_rank)
            
            # Compute scales and quantize
            n_bits = self.latent_quantizer.n_bits
            group_size = self.latent_quantizer.group_size
            sym = self.latent_quantizer.sym
            clip_ratio = self.latent_quantizer.clip_ratio
            
            if n_bits >= 16:
                # No quantization — store identity scales so dequantize_latent_from_integers
                # retrieves them via scales_dict and avoids the int16-truncation fallback.
                quantized_integers = flat_latent
                scales = torch.ones(flat_latent.shape[0], 1, device=flat_latent.device, dtype=flat_latent.dtype)
                base = torch.zeros(flat_latent.shape[0], 1, device=flat_latent.device, dtype=flat_latent.dtype)
                scales_dict[group_idx] = (scales, base)
            else:
                # Compute scales
                if sym:
                    w_max = flat_latent.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
                    q_max = (2**(n_bits-1)-1)
                    q_min = (-2**(n_bits-1))
                    if clip_ratio < 1.0:
                        w_max = w_max * clip_ratio
                    scales = w_max / q_max
                    base = torch.zeros_like(scales)
                else:
                    w_max = flat_latent.amax(dim=-1, keepdim=True)
                    w_min = flat_latent.amin(dim=-1, keepdim=True)
                    q_max = (2**(n_bits)-1)
                    q_min = (0)
                    if clip_ratio < 1.0:
                        w_max *= clip_ratio
                        w_min *= clip_ratio
                    scales = (w_max-w_min).clamp(min=1e-5) / q_max
                    base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
                
                # Quantize to integers (before dequantization)
                quantized_integers = torch.clamp(torch.round(flat_latent / scales) + base, q_min, q_max) - base
                # Store scales and base for dequantization
                scales_dict[group_idx] = (scales, base)
            
            # Reshape back to (batch_size, seq_len, group_rank)
            quantized_integers_reshaped = quantized_integers.reshape(batch_size, seq_len, group_rank)
            quantized_integers_list.append(quantized_integers_reshaped)
            
            total_ranks += group_rank
        
        # Concatenate all groups
        quantized_integers_full = torch.cat(quantized_integers_list, dim=-1)
        
        return quantized_integers_full, scales_dict
    
    def dequantize_latent_from_integers(self, quantized_integers: torch.Tensor, scales_dict: dict):
        """
        Dequantize latents from quantized integers and scales.
        
        quantized_integers: Tensor of shape (batch_size, seq_len, r1 + r2 + ...) with integer values (int8/int16 from cache or float)
        scales_dict: Dict mapping group_idx to (scales, base) tuples
        Returns: Dequantized float values
        """
        if not quantized_integers.is_floating_point():
            quantized_integers = quantized_integers.float()
        original_shape = quantized_integers.shape
        batch_size, seq_len, total_dims = original_shape
        
        # Validate that quantized_integers has the correct shape
        # VT outputs sum(ranks) dimensions, regardless of num_heads_per_group
        expected_total_ranks = sum(self.ranks)
        if total_dims != expected_total_ranks:
            raise RuntimeError(
                f"Shape mismatch in dequantize_latent_from_integers: "
                f"quantized_integers has {total_dims} dimensions in last axis, "
                f"but expected {expected_total_ranks} (sum of ranks={self.ranks}). "
                f"quantized_integers shape={original_shape}, module.ranks={self.ranks}, num_groups={self.num_groups}"
            )
        
        dequantized_list = []
        total_ranks = 0
        
        for group_idx in range(self.num_groups):
            group_rank = self.ranks[group_idx]
            if total_ranks + group_rank > total_dims:
                raise RuntimeError(
                    f"Index out of bounds in dequantize_latent_from_integers for group {group_idx}: "
                    f"trying to slice [{total_ranks}:{total_ranks + group_rank}] from tensor with {total_dims} dimensions. "
                    f"module.ranks={self.ranks}, num_groups={self.num_groups}"
                )
            quantized_group = quantized_integers[:, :, total_ranks: total_ranks + group_rank]
            
            # Reshape to (batch_size * seq_len, group_rank)
            flat_quantized = quantized_group.reshape(-1, group_rank)
            
            if group_idx in scales_dict:
                scales, base = scales_dict[group_idx]
                # Ensure scales and base have the correct shape for broadcasting
                # flat_quantized has shape (batch_size * seq_len, group_rank)
                # scales and base should have shape (batch_size * seq_len, 1) for proper broadcasting
                
                # Reshape scales and base to (batch_size * seq_len, 1) if needed
                expected_size = batch_size * seq_len
                
                # Handle different possible shapes of scales/base
                if scales.dim() == 2:
                    if scales.shape[0] == expected_size and scales.shape[1] == 1:
                        # Already correct shape: (batch_size * seq_len, 1)
                        pass
                    elif scales.shape[0] == seq_len and scales.shape[1] == 1 and batch_size == 1:
                        # Shape is (seq_len, 1) which equals (batch_size * seq_len, 1) when batch_size=1
                        pass
                    elif scales.numel() == expected_size:
                        # Can reshape to (batch_size * seq_len, 1)
                        scales = scales.reshape(expected_size, 1)
                        base = base.reshape(expected_size, 1)
                    else:
                        # Shape mismatch - try to fix or raise error
                        raise RuntimeError(
                            f"Shape mismatch in dequantize_latent_from_integers for group {group_idx}: "
                            f"scales shape={scales.shape} (numel={scales.numel()}), "
                            f"base shape={base.shape} (numel={base.numel()}), "
                            f"expected_size={expected_size} (batch_size={batch_size} * seq_len={seq_len}), "
                            f"flat_quantized shape={flat_quantized.shape}"
                        )
                else:
                    # scales/base are not 2D - try to reshape
                    if scales.numel() == expected_size:
                        scales = scales.reshape(expected_size, 1)
                        base = base.reshape(expected_size, 1)
                    else:
                        raise RuntimeError(
                            f"Invalid scales/base shape for group {group_idx}: "
                            f"scales shape={scales.shape} (numel={scales.numel()}), "
                            f"base shape={base.shape} (numel={base.numel()}), "
                            f"expected_size={expected_size}"
                        )
                
                # Dequantize: x = (Q(x) + base) * scales
                # flat_quantized: (batch_size * seq_len, group_rank)
                # base, scales: (batch_size * seq_len, 1)
                # Broadcasting: (N, group_rank) + (N, 1) -> (N, group_rank)
                dequantized_flat = (flat_quantized + base) * scales
            else:
                # No quantization was applied (n_bits >= 16)
                dequantized_flat = flat_quantized
            
            # Reshape back to (batch_size, seq_len, group_rank)
            dequantized_reshaped = dequantized_flat.reshape(batch_size, seq_len, group_rank)
            dequantized_list.append(dequantized_reshaped)
            
            total_ranks += group_rank
        
        # Concatenate all groups
        dequantized_full = torch.cat(dequantized_list, dim=-1)
        
        return dequantized_full
    
    
    def configure_latent_quantizer(self, 
        n_bits: int, 
        group_size: int, 
        sym: bool,
        clip_ratio: float,
        hadamard = False
    ):
        #self.latent_quantizer = Quantizer(n_bits, group_size, sym, clip_ratio, hadamard)
        self.latent_quantizer = Quantizer(n_bits, group_size, sym, clip_ratio)
        if hadamard:
            try:
                self.fused_hadamard_matrix()
            except Exception as e:
                logger.warning(
                    f"Hadamard rotation failed for {getattr(self, 'name', repr(self))}: {e}. "
                    "Continuing without rotation — 4-bit uniform quantization may have poor quality."
                )
        self.quantized_latents = True
    
    
    def fused_hadamard_matrix(self):
        total_ranks = 0
        for i in range(self.num_groups):
            # Apply Q to VT
            VT_weight_i = self.VT.weight.data[total_ranks: total_ranks+self.ranks[i], :]
            VT_weight_i = apply_hadamard(VT_weight_i.t())
            self.VT.weight.data[total_ranks: total_ranks+self.ranks[i], :] = VT_weight_i.t()
            # Apply Q^T to U
            U_weight_i = self.U[i].weight.data
            U_weight_i = apply_hadamard(U_weight_i)
            self.U[i].weight.data = U_weight_i
            
            total_ranks += self.ranks[i]
    
    def configure_per_dim_quantization(self, bitwidths: torch.Tensor, num_heads: Optional[int] = None,
                                       scaling_type: str = "tokenwise", channelwise_scalings: Optional[torch.Tensor] = None,
                                       channel_group_size: int = 64, kvtc_group_size: int = 64):
        """
        Configure per-dimension quantization with bitwidths.

        Args:
            bitwidths: Tensor of shape (num_groups, group_rank) or (sum(ranks),)
                      containing bitwidth for each dimension
            num_heads: Optional number of heads (for per-head scaling). If None, uses per-group scaling.
            scaling_type: "tokenwise" (default), "channelwise", "channel_group", "factored", or "kvtc".
                         - "tokenwise": one scale per token (max across all dims in group)
                         - "channelwise": one static scale per dim (from calibration)
                         - "channel_group": one scale per dim per bucket of G tokens (G=channel_group_size)
                         - "factored": scale = channel_scale * token_scale, O(T+D) storage
                         - "kvtc": like factored but adaptive scale covers kvtc_group_size dims at a time
                                   (finer-grained than factored, matches KVTC paper's group microscaling)
            channelwise_scalings: Tensor [num_dims] with max absolute values per dimension.
                                Required if scaling_type in ("channelwise", "channel_group", "factored", "kvtc").
            channel_group_size: Bucket size G for "channel_group" scaling (default 64).
            kvtc_group_size: Dims per adaptive-scale group for "kvtc" scaling (default 64).
        """
        self.use_per_dim_quantization = True
        # Flatten to (sum(ranks),) if needed
        if bitwidths.dim() == 2:
            self.bitwidths = bitwidths.flatten()
        else:
            self.bitwidths = bitwidths
        
        # Store head information for per-head scaling
        self.num_heads = num_heads
        if num_heads is not None and self.num_groups > 0:
            if num_heads % self.num_groups != 0:
                raise ValueError(f"num_heads ({num_heads}) must be divisible by num_groups ({self.num_groups})")
            self.num_heads_per_group = num_heads // self.num_groups
        else:
            self.num_heads_per_group = None
        
        # Set scaling type
        _valid_scaling_types = ["tokenwise", "channelwise", "channel_group", "factored", "kvtc"]
        if scaling_type not in _valid_scaling_types:
            raise ValueError(f"scaling_type must be one of {_valid_scaling_types}, got '{scaling_type}'")
        self.scaling_type = scaling_type

        # Set channelwise scalings — required for "channelwise", "channel_group", "factored", and "kvtc"
        if scaling_type in ("channelwise", "channel_group", "factored", "kvtc"):
            if channelwise_scalings is None:
                raise ValueError(f"channelwise_scalings must be provided when scaling_type='{scaling_type}'")
            if channelwise_scalings.numel() != self.bitwidths.numel():
                raise ValueError(
                    f"channelwise_scalings size mismatch: expected {self.bitwidths.numel()} dimensions, "
                    f"got {channelwise_scalings.numel()}"
                )
            self.channelwise_scalings = channelwise_scalings.clamp(min=1e-8)
        else:
            self.channelwise_scalings = None

        # Store channel group size and kvtc group size
        self.channel_group_size = max(1, int(channel_group_size))
        self.kvtc_group_size = max(1, int(kvtc_group_size))
        
        # Store base bitwidths (used by the sliding-window requantize step)
        self.base_bitwidths = self.bitwidths.clone() if self.bitwidths is not None else None
        
        # Ensure bitwidths are on the right device (will be moved with module)
        if self.bitwidths is not None:
            self.bitwidths = self.bitwidths.to(next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))
        
        # Ensure channelwise_scalings are on the right device
        if self.channelwise_scalings is not None:
            self.channelwise_scalings = self.channelwise_scalings.to(next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))
    
    def truncate_quantized_values(
        self,
        quantized_states: torch.Tensor,
        scales: torch.Tensor,
        old_bitwidths: torch.Tensor,
        new_bitwidths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Truncate quantized values to lower bitwidths using nested quantization.
        
        Nested quantization property: lower bitwidth representations are truncations
        of higher bitwidth representations. This allows efficient bit reduction without
        full re-quantization.
        
        Args:
            quantized_states: Tensor of shape (batch_size, seq_len, num_dims) with quantized values
            scales: Tensor of shape (batch_size, seq_len) or (batch_size, seq_len, num_groups/num_heads) with scales
            old_bitwidths: Tensor of shape (seq_len, num_dims) with current bitwidths per token/dim
            new_bitwidths: Tensor of shape (seq_len, num_dims) with target bitwidths per token/dim
            
        Returns:
            Tuple of (truncated_quantized_states, scales) - scales remain unchanged
        """
        batch_size, seq_len, num_dims = quantized_states.shape
        
        # Ensure bitwidths are on the same device
        old_bitwidths = old_bitwidths.to(quantized_states.device)
        new_bitwidths = new_bitwidths.to(quantized_states.device)
        
        # Compute bit reduction per token/dimension
        bit_reduction = old_bitwidths - new_bitwidths  # (seq_len, num_dims)
        
        # Clamp to ensure we only reduce bits (not increase)
        bit_reduction = bit_reduction.clamp(min=0)
        
        # Reshape for per-token processing
        quantized_flat = quantized_states.reshape(batch_size * seq_len, num_dims)  # (batch*seq, num_dims)
        # bit_reduction is (seq_len, num_dims); expand to (batch*seq, num_dims) for batch_size >= 1
        bit_reduction_flat = bit_reduction.unsqueeze(0).expand(
            batch_size, seq_len, num_dims
        ).reshape(batch_size * seq_len, num_dims)
        
        # Truncate: quantized_new = quantized_old >> bit_reduction
        # rshift (>>) is integer-only; cast to long in case cache stored as float16/half
        quantized_flat_int = quantized_flat.long()
        truncated_flat = quantized_flat_int >> bit_reduction_flat.long()
        # 0 bits = evict for this layer: ensure truncated is 0 (rshift of signed may not zero)
        new_bw_flat = new_bitwidths.unsqueeze(0).expand(batch_size, seq_len, num_dims).reshape(batch_size * seq_len, num_dims)
        truncated_flat[new_bw_flat == 0] = 0
        # Reshape back
        truncated_states = truncated_flat.reshape(batch_size, seq_len, num_dims)
        
        # Scales remain the same (alpha doesn't change, only quantization precision)
        return truncated_states, scales
    
    def quantize_with_per_token_bitwidths(
        self,
        latent_states: torch.Tensor,
        per_token_bitwidths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize latent states using per-token, per-dimension bitwidths.
        
        Args:
            latent_states: Tensor of shape (batch_size, seq_len, num_dims)
            per_token_bitwidths: Tensor of shape (seq_len, num_dims) with bitwidth for each token/dimension
            
        Returns:
            Tuple of (quantized_states, scales) where:
            - quantized_states: Tensor of shape (batch_size, seq_len, num_dims)
            - scales: Tensor of shape (batch_size, seq_len) with alpha per token
        """
        batch_size, seq_len, num_dims = latent_states.shape
        
        # Ensure bitwidths are on the same device
        per_token_bitwidths = per_token_bitwidths.to(latent_states.device)
        
        if per_token_bitwidths.shape != (seq_len, num_dims):
            raise ValueError(
                f"per_token_bitwidths shape mismatch: expected ({seq_len}, {num_dims}), "
                f"got {per_token_bitwidths.shape}"
            )
        
        # Reshape for per-token processing
        latent_flat = latent_states.reshape(-1, num_dims)  # (batch*seq, num_dims)
        bitwidths_flat = per_token_bitwidths.unsqueeze(0).expand(batch_size, seq_len, num_dims)
        bitwidths_flat = bitwidths_flat.reshape(-1, num_dims)  # (batch*seq, num_dims)
        num_tokens = latent_flat.shape[0]
        
        # Apply channel-wise scaling if available
        if self.channelwise_scaling is not None:
            sc = self.channelwise_scaling.to(latent_flat.device)
            latent_flat = latent_flat / sc.unsqueeze(0)
        
        # Compute alpha (max absolute value) per token
        alpha = latent_flat.abs().max(dim=1, keepdim=True)[0]  # (num_tokens, 1)
        alpha = alpha.clamp(min=1e-8)
        
        # Compute stepsize for each token and dimension (use max(b,1) to avoid 2^0 in denominator for b=0)
        alpha_expanded = alpha.expand(num_tokens, num_dims)  # (num_tokens, num_dims)
        bitwidths_safe = bitwidths_flat.float().clamp(min=1)  # 0 bits -> evict, handled below
        stepsize = 2.0 * alpha_expanded / (2.0 ** bitwidths_safe)
        
        # Quantize
        quantized_flat = torch.round(latent_flat / stepsize)
        
        # Clamp to valid range based on bitwidths (only where b > 0; b=0 -> 0)
        max_val = (2.0 ** (bitwidths_safe - 1)) - 1
        min_val = -(2.0 ** (bitwidths_safe - 1))
        quantized_flat = torch.clamp(quantized_flat, min=min_val, max=max_val)
        quantized_flat[bitwidths_flat == 0] = 0  # 0 bits = evict for this layer
        
        # Reshape back
        quantized_states = quantized_flat.reshape(batch_size, seq_len, num_dims)
        scales = alpha.reshape(batch_size, seq_len)
        
        return quantized_states, scales
    
    def configure_channelwise_scaling(self, sc: torch.Tensor):
        """
        Configure channel-wise scaling (SC) for whitening before quantization.
        SC is computed from covariance matrix eigendecomposition: SC = diag(Λ^(-1/2) · U^T)
        
        Args:
            sc: Tensor of shape (sum(ranks),) containing scaling factor for each dimension
        """
        if sc.dim() > 1:
            sc = sc.flatten()
        self.channelwise_scaling = sc
        
        # Ensure SC is on the right device (will be moved with module)
        if self.channelwise_scaling is not None:
            self.channelwise_scaling = self.channelwise_scaling.to(
                next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            )
    
    def quantize_with_per_dim_bitwidths(
        self,
        latent_states: torch.Tensor,
        bitwidths_per_token: Optional[torch.Tensor] = None,
        token_offset: int = 0,
    ):
        """
        Quantize latent states (A(x)) using per-dimension bitwidths.
        
        Scaling types:
        - "tokenwise": For each token, compute alpha = max(|latent_states[token]|) as the scale
        - "channelwise": Use pre-computed scalings per dimension (from calibration data)
        
        For each dimension i, compute stepsize_i = 2*alpha_i / 2^b_i
        Quantize: Q(latent_states)[token, i] = round(latent_states[token, i] / stepsize_i)
        
        Args:
            latent_states: Tensor of shape (batch_size, seq_len, sum(ranks))
            where sum(ranks) = rank0 + rank1 + ... + rank_{num_groups-1}
            bitwidths_per_token: Optional (seq_len, num_dims) or (batch*seq_len, num_dims). When provided,
                use these bitwidths instead of self.bitwidths so quantize uses the SAME bitwidths that
                will be stored in the cache (avoids magnitude mismatch at dequant).
            
        Returns:
            Tuple of (quantized_states, scales) where:
            - quantized_states: Tensor of shape (batch_size, seq_len, sum(ranks)) with quantized values
            - scales: For tokenwise: (batch_size, seq_len, num_heads) if per-head, 
                     or (batch_size, seq_len, num_groups) if per-group
                     For channelwise: (batch_size, seq_len, num_dims) with per-dimension scales
        """
        if self.bitwidths is None and bitwidths_per_token is None:
            raise ValueError("Bitwidths not configured. Call configure_per_dim_quantization or pass bitwidths_per_token.")
        
        batch_size, seq_len, num_dims = latent_states.shape
        
        # Reshape for per-token processing: (batch_size * seq_len, num_dims)
        latent_flat = latent_states.reshape(-1, num_dims)
        num_tokens = latent_flat.shape[0]
        
        # Resolve bitwidths: use bitwidths_per_token if provided (must match what we store in cache)
        if bitwidths_per_token is not None:
            bitwidths_per_token = bitwidths_per_token.to(latent_states.device)
            if bitwidths_per_token.dim() == 2 and bitwidths_per_token.shape[1] == num_dims:
                bw_len = bitwidths_per_token.shape[0]
                if bw_len == num_tokens:
                    bitwidths_flat = bitwidths_per_token
                elif bw_len == seq_len:
                    bitwidths_flat = bitwidths_per_token.unsqueeze(0).expand(batch_size, seq_len, num_dims).reshape(num_tokens, num_dims)
                else:
                    bitwidths_flat = None
            else:
                bitwidths_flat = None
            if bitwidths_flat is None:
                bitwidths_flat = self.bitwidths.to(latent_states.device).unsqueeze(0).expand(num_tokens, num_dims)
        else:
            bitwidths_flat = self.bitwidths.to(latent_states.device).unsqueeze(0).expand(num_tokens, num_dims)
        if bitwidths_flat.shape[1] != num_dims:
            raise ValueError(
                f"Bitwidths size mismatch: expected {num_dims} dimensions, "
                f"got {bitwidths_flat.shape[1]}"
            )
        
        # Step 1: Apply channel-wise scaling SC if available (whitening before quantization)
        if self.channelwise_scaling is not None:
            sc = self.channelwise_scaling.to(latent_flat.device)
            if sc.numel() != num_dims:
                raise ValueError(
                    f"Channel-wise scaling size mismatch: expected {num_dims} dimensions, "
                    f"got {sc.numel()}"
                )
            # Apply per-dimension division: v_whitened = v / SC
            latent_flat = latent_flat / sc.unsqueeze(0)
        
        # Handle channelwise scaling
        if self.scaling_type == "channelwise":
                # Uniform quantization with channelwise scaling
                if self.channelwise_scalings is None:
                    raise ValueError("channelwise_scalings not configured. Call configure_per_dim_quantization with channelwise_scalings.")
                
                # Get pre-computed scalings per dimension
                alpha_per_dim = self.channelwise_scalings.to(latent_states.device)  # (num_dims,)
                
                # Compute stepsize for each dimension: stepsize_i = 2*alpha_i / 2^b_i
                # bitwidths_flat already (num_tokens, num_dims)
                alpha_expanded = alpha_per_dim.unsqueeze(0).expand(num_tokens, num_dims)  # (num_tokens, num_dims)
                bitwidths_expanded = bitwidths_flat
                bitwidths_safe = bitwidths_expanded.float().clamp(min=1)  # 0 bits -> evict
                stepsize = 2.0 * alpha_expanded / (2.0 ** bitwidths_safe)
                
                # Quantize
                quantized_flat = torch.round(latent_flat / stepsize)
                
                # Clamp to valid range (only where b > 0)
                max_val = (2.0 ** (bitwidths_safe - 1)) - 1
                min_val = -(2.0 ** (bitwidths_safe - 1))
                quantized_flat = torch.clamp(quantized_flat, min=min_val, max=max_val)
                quantized_flat[bitwidths_expanded == 0] = 0  # 0 bits = evict for this layer
                
                # Reshape back to original shape
                quantized_states = quantized_flat.reshape(batch_size, seq_len, num_dims)
                
                # Return scales as (batch_size, seq_len, num_dims) - one scale per dimension per token
                # For channelwise, scales are the same for all tokens (per-dimension)
                scales = alpha_per_dim.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, num_dims)
                
                return quantized_states, scales

        elif self.scaling_type == "channel_group":
            # Per-channel scale shared across a bucket of G consecutive tokens.
            # Tokens at global positions [token_offset, token_offset + seq_len) are assigned to
            # buckets b = global_pos // G.  All tokens within the same bucket share the per-channel
            # max of that bucket's tokens (computed from the current batch only).
            G = self.channel_group_size
            global_indices = token_offset + torch.arange(num_tokens, device=latent_flat.device)
            bucket_ids = global_indices // G
            # Relative bucket indices starting from 0 within this batch
            bucket_ids_rel = bucket_ids - bucket_ids[0]
            num_local_buckets = int(bucket_ids_rel.max().item()) + 1

            # Per-bucket per-channel max: (num_local_buckets, num_dims)
            bucket_scale = torch.zeros(num_local_buckets, num_dims,
                                       device=latent_flat.device, dtype=latent_flat.dtype)
            for b in range(num_local_buckets):
                mask = bucket_ids_rel == b
                bucket_scale[b] = latent_flat[mask].abs().amax(dim=0).clamp(min=1e-8)

            # Assign each token its bucket's scale: (num_tokens, num_dims)
            scales_flat = bucket_scale[bucket_ids_rel]

            bitwidths_safe = bitwidths_flat.float().clamp(min=1)
            stepsize = 2.0 * scales_flat / (2.0 ** bitwidths_safe)
            quantized_flat = torch.round(latent_flat / stepsize)
            max_val = (2.0 ** (bitwidths_safe - 1)) - 1
            min_val = -(2.0 ** (bitwidths_safe - 1))
            quantized_flat = torch.clamp(quantized_flat, min=min_val, max=max_val)
            quantized_flat[bitwidths_flat == 0] = 0

            quantized_states = quantized_flat.reshape(batch_size, seq_len, num_dims)
            # Store per-token per-channel scales so dequant can reconstruct exactly
            scales = scales_flat.reshape(batch_size, seq_len, num_dims)
            return quantized_states, scales

        elif self.scaling_type == "factored":
            # Factored O(T+D) scaling: effective_scale[t,d] = channel_scale[d] * token_scale[t]
            # channel_scale is static (from calibration); token_scale is computed per token.
            if self.channelwise_scalings is None:
                raise ValueError("channelwise_scalings must be set for scaling_type='factored'")
            fc_scale = self.channelwise_scalings.to(latent_flat.device)  # (num_dims,)

            # Normalize by channel scale, then compute per-group token alpha on normalised latents
            latent_normalized = latent_flat / fc_scale.unsqueeze(0)  # (num_tokens, num_dims)
            alpha_parts = []
            total_r = 0
            for group_idx in range(self.num_groups):
                group_rank = self.ranks[group_idx]
                group_norm = latent_normalized[:, total_r: total_r + group_rank]
                alpha_g = group_norm.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
                alpha_parts.append(alpha_g)
                total_r += group_rank
            alpha = torch.cat(alpha_parts, dim=1)  # (num_tokens, num_groups)

            repeats = torch.tensor(self.ranks, device=latent_flat.device, dtype=torch.long)
            # Expand token alpha to per-dim
            alpha_expanded = alpha.repeat_interleave(repeats, dim=1)  # (num_tokens, num_dims)

            # Effective scale per token per dim
            effective_scale = fc_scale.unsqueeze(0) * alpha_expanded  # (num_tokens, num_dims)

            bitwidths_safe = bitwidths_flat.float().clamp(min=1)
            stepsize = 2.0 * effective_scale / (2.0 ** bitwidths_safe)
            quantized_flat = torch.round(latent_flat / stepsize)
            max_val = (2.0 ** (bitwidths_safe - 1)) - 1
            min_val = -(2.0 ** (bitwidths_safe - 1))
            quantized_flat = torch.clamp(quantized_flat, min=min_val, max=max_val)
            quantized_flat[bitwidths_flat == 0] = 0

            quantized_states = quantized_flat.reshape(batch_size, seq_len, num_dims)
            # Store per-group token scales only (channel scale is static on the module)
            scales = alpha.reshape(batch_size, seq_len, self.num_groups)
            return quantized_states, scales

        elif self.scaling_type == "kvtc":
            # KVTC-style two-level scaling:
            #   static per-dim scale (channelwise_scalings, e.g. std dev from calibration)
            #   × adaptive per-kvtc_group scale (one scale per kvtc_group_size dims per token)
            # Finer-grained than "factored" (which has one adaptive scale per entire head group).
            if self.channelwise_scalings is None:
                raise ValueError("channelwise_scalings must be set for scaling_type='kvtc'")
            fc_scale = self.channelwise_scalings.to(latent_flat.device)  # (num_dims,)

            # Normalize by per-dim static scale
            latent_normalized = latent_flat / fc_scale.unsqueeze(0)  # (num_tokens, num_dims)

            # Compute adaptive scale per kvtc_group_size-dim block
            gs = self.kvtc_group_size
            num_kvtc_groups = (num_dims + gs - 1) // gs
            alpha_parts = []
            for g in range(num_kvtc_groups):
                start = g * gs
                end = min(start + gs, num_dims)
                alpha_g = latent_normalized[:, start:end].abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
                alpha_parts.append(alpha_g)
            alpha = torch.cat(alpha_parts, dim=1)  # (num_tokens, num_kvtc_groups)

            # Expand adaptive scale to per-dim
            group_sizes = [min(gs, num_dims - g * gs) for g in range(num_kvtc_groups)]
            repeats = torch.tensor(group_sizes, device=latent_flat.device, dtype=torch.long)
            alpha_expanded = alpha.repeat_interleave(repeats, dim=1)  # (num_tokens, num_dims)

            effective_scale = fc_scale.unsqueeze(0) * alpha_expanded  # (num_tokens, num_dims)
            bitwidths_safe = bitwidths_flat.float().clamp(min=1)
            stepsize = 2.0 * effective_scale / (2.0 ** bitwidths_safe)
            quantized_flat = torch.round(latent_flat / stepsize)
            max_val = (2.0 ** (bitwidths_safe - 1)) - 1
            min_val = -(2.0 ** (bitwidths_safe - 1))
            quantized_flat = torch.clamp(quantized_flat, min=min_val, max=max_val)
            quantized_flat[bitwidths_flat == 0] = 0

            quantized_states = quantized_flat.reshape(batch_size, seq_len, num_dims)
            scales = alpha.reshape(batch_size, seq_len, num_kvtc_groups)
            return quantized_states, scales

        # Tokenwise scaling (original behavior)
        # Check if we should use per-head or per-group scaling
        use_per_head = (self.num_heads is not None and self.num_heads_per_group is not None)
        
        # Debug logging for first call
        if not hasattr(self, '_quantization_debug_logged'):
            self._quantization_debug_logged = True
        
        if use_per_head:
            # Per-head scaling (vectorized): compute alpha per head, expand to num_dims, single batched quantize
            alpha_parts = []
            total_ranks = 0
            for group_idx in range(self.num_groups):
                group_rank = self.ranks[group_idx]
                head_rank = group_rank // self.num_heads_per_group
                if head_rank * self.num_heads_per_group != group_rank:
                    raise ValueError(
                        f"group_rank ({group_rank}) must be divisible by num_heads_per_group "
                        f"({self.num_heads_per_group})"
                    )
                group_latent = latent_flat[:, total_ranks : total_ranks + group_rank]
                group_latent_3d = group_latent.reshape(
                    num_tokens, self.num_heads_per_group, head_rank
                )
                alpha_g = (
                    group_latent_3d.abs().max(dim=2, keepdim=True)[0].clamp(min=1e-8).squeeze(2)
                )
                alpha_parts.append(alpha_g)
                total_ranks += group_rank
            alpha = torch.cat(alpha_parts, dim=1)
            repeats = torch.tensor(
                [
                    self.ranks[g] // self.num_heads_per_group
                    for g in range(self.num_groups)
                    for _ in range(self.num_heads_per_group)
                ],
                device=latent_flat.device,
                dtype=torch.long,
            )
            alpha_expanded = alpha.repeat_interleave(repeats, dim=1)
            bitwidths_safe = bitwidths_flat.float().clamp(min=1)
            stepsize = 2.0 * alpha_expanded / (2.0 ** bitwidths_safe)
            quantized_flat = torch.round(latent_flat / stepsize)
            max_val = (2.0 ** (bitwidths_safe - 1)) - 1
            min_val = -(2.0 ** (bitwidths_safe - 1))
            quantized_flat = torch.clamp(quantized_flat, min=min_val, max=max_val)
            quantized_flat[bitwidths_flat == 0] = 0
            quantized_states = quantized_flat.reshape(batch_size, seq_len, num_dims)
            scales = alpha.reshape(batch_size, seq_len, self.num_heads)

        else:
            # Per-group scaling (vectorized): compute alpha per group, expand to num_dims, single batched quantize
            alpha_parts = []
            total_ranks = 0
            for group_idx in range(self.num_groups):
                group_rank = self.ranks[group_idx]
                group_latent = latent_flat[:, total_ranks : total_ranks + group_rank]
                alpha_g = group_latent.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
                alpha_parts.append(alpha_g)
                total_ranks += group_rank
            alpha = torch.cat(alpha_parts, dim=1)
            repeats = torch.tensor(
                self.ranks, device=latent_flat.device, dtype=torch.long
            )
            alpha_expanded = alpha.repeat_interleave(repeats, dim=1)
            bitwidths_safe = bitwidths_flat.float().clamp(min=1)
            stepsize = 2.0 * alpha_expanded / (2.0 ** bitwidths_safe)
            quantized_flat = torch.round(latent_flat / stepsize)
            max_val = (2.0 ** (bitwidths_safe - 1)) - 1
            min_val = -(2.0 ** (bitwidths_safe - 1))
            quantized_flat = torch.clamp(quantized_flat, min=min_val, max=max_val)
            quantized_flat[bitwidths_flat == 0] = 0
            quantized_states = quantized_flat.reshape(batch_size, seq_len, num_dims)
            scales = alpha.reshape(batch_size, seq_len, self.num_groups)

        return quantized_states, scales
    
    def dequantize_with_per_dim_bitwidths(
        self,
        quantized_states: torch.Tensor,
        scales: torch.Tensor,
        target_dtype: Optional[torch.dtype] = None,
        bitwidths_per_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dequantize latent states using per-dimension bitwidths and scales.
        
        Args:
            quantized_states: Tensor of shape (batch_size, seq_len, sum(ranks)) with quantized values
            scales: Tensor of shape:
                   - (batch_size, seq_len, num_dims) if channelwise scaling
                   - (batch_size, seq_len, num_heads) if per-head tokenwise scaling
                   - (batch_size, seq_len, num_groups) if per-group tokenwise scaling
                   - (batch_size, seq_len) if old format (backward compatibility)
            target_dtype: Optional dtype to convert result to (default: keep float32 from dequantization)
            bitwidths_per_token: Optional (seq_len, num_dims) or (batch*seq_len, num_dims). When set,
                   use per-token per-dimension bitwidths (required for correct reconstruction after
                   adaptive quantization). Otherwise use self.bitwidths (single per-dim).
            
        Returns:
            Dequantized latent states of shape (batch_size, seq_len, sum(ranks))
        """
        batch_size, seq_len, num_dims = quantized_states.shape
        num_tokens = batch_size * seq_len
        
        # Resolve bitwidths: per-token (adaptive) or static per-dim
        bitwidths = None
        if bitwidths_per_token is not None:
            bitwidths_per_token = bitwidths_per_token.to(quantized_states.device)
            # Allow (seq_len, *) or (num_tokens, *); slice to num_dims if needed (cache may have extra dims)
            if bitwidths_per_token.dim() == 2 and bitwidths_per_token.shape[1] >= num_dims:
                bw_seq = bitwidths_per_token.shape[0]
                if bw_seq == seq_len or bw_seq == num_tokens:
                    bitwidths_per_token = bitwidths_per_token[:, :num_dims].contiguous()
                elif bw_seq < seq_len and bw_seq > 0:
                    # Cache has fewer rows (e.g. before new token's bitwidths appended): repeat last row
                    last_row = bitwidths_per_token[-1:].expand(seq_len - bw_seq, num_dims)
                    bitwidths_per_token = torch.cat([bitwidths_per_token[:, :num_dims], last_row], dim=0)
                else:
                    bitwidths_per_token = None
            else:
                bitwidths_per_token = None
            if bitwidths_per_token is not None:
                if bitwidths_per_token.shape == (seq_len, num_dims):
                    bitwidths = bitwidths_per_token.unsqueeze(0).expand(batch_size, seq_len, num_dims).reshape(num_tokens, num_dims)
                elif bitwidths_per_token.shape == (num_tokens, num_dims):
                    bitwidths = bitwidths_per_token
            if bitwidths is None:
                raise ValueError(
                    f"bitwidths_per_token must be (seq_len={seq_len}, num_dims={num_dims}) or "
                    f"({num_tokens}, {num_dims}), got {bitwidths_per_token.shape if bitwidths_per_token is not None else 'None'}"
                )
        else:
            if self.bitwidths is None:
                raise ValueError("Bitwidths not configured. Pass bitwidths_per_token or call configure_per_dim_quantization first.")
            bitwidths = self.bitwidths.to(quantized_states.device)
            if bitwidths.numel() != num_dims:
                raise ValueError(
                    f"Bitwidths size mismatch: expected {num_dims} dimensions, "
                    f"got {bitwidths.numel()}"
                )
            bitwidths = bitwidths.unsqueeze(0).expand(num_tokens, num_dims)
        
        # Handle channelwise scaling
        if self.scaling_type == "channelwise":
                if scales.dim() == 3 and scales.shape[2] == num_dims:
                    scales_flat = scales.reshape(-1, num_dims)
                elif self.channelwise_scalings is not None:
                    scales_flat = self.channelwise_scalings.to(quantized_states.device).unsqueeze(0).expand(batch_size * seq_len, num_dims)
                else:
                    raise ValueError(f"Channelwise scaling requires scales of shape (batch_size, seq_len, {num_dims}) or channelwise_scalings to be set")
                
                quantized_flat = quantized_states.reshape(-1, num_dims)
                bitwidths_expanded = bitwidths
                pow2 = self._pow2_table.to(quantized_states.device)[bitwidths_expanded.long().clamp(0, 16)]
                stepsize = 2.0 * scales_flat / pow2
                dequantized_flat = quantized_flat.float() * stepsize
                dequantized_flat = dequantized_flat * (bitwidths_expanded != 0).to(dequantized_flat.dtype)
        
        # Restore channel-wise scaling SC if available (inverse whitening after dequantization)
        if self.scaling_type == "channelwise":
                if self.channelwise_scaling is not None:
                    sc = self.channelwise_scaling.to(dequantized_flat.device)
                    if sc.numel() != num_dims:
                        raise ValueError(
                            f"Channel-wise scaling size mismatch: expected {num_dims} dimensions, "
                            f"got {sc.numel()}"
                        )
                    dequantized_flat = dequantized_flat * sc.unsqueeze(0)
                dequantized_states = dequantized_flat.reshape(batch_size, seq_len, num_dims)
                if target_dtype is not None:
                    dequantized_states = dequantized_states.to(target_dtype)
                return dequantized_states

        elif self.scaling_type == "channel_group":
            # scales shape: (batch_size, seq_len, num_dims) — per-token per-channel, repeated within bucket
            if not (scales.dim() == 3 and scales.shape[2] == num_dims):
                raise ValueError(
                    f"channel_group dequant requires scales of shape (batch, seq_len, {num_dims}), got {scales.shape}"
                )
            scales_flat = scales.reshape(-1, num_dims)
            quantized_flat = quantized_states.reshape(-1, num_dims)
            pow2 = self._pow2_table.to(quantized_states.device)[bitwidths.long().clamp(0, 16)]
            stepsize = 2.0 * scales_flat / pow2
            dequantized_flat = quantized_flat.float() * stepsize
            dequantized_flat = dequantized_flat * (bitwidths != 0).to(dequantized_flat.dtype)
            # Restore whitening SC if present
            if self.channelwise_scaling is not None:
                sc = self.channelwise_scaling.to(dequantized_flat.device)
                dequantized_flat = dequantized_flat * sc.unsqueeze(0)
            dequantized_states = dequantized_flat.reshape(batch_size, seq_len, num_dims)
            if target_dtype is not None:
                dequantized_states = dequantized_states.to(target_dtype)
            return dequantized_states

        elif self.scaling_type == "factored":
            # scales shape: (batch_size, seq_len, num_groups) — per-group token scales
            if self.channelwise_scalings is None:
                raise ValueError("channelwise_scalings must be set for scaling_type='factored'")
            if not (scales.dim() == 3 and scales.shape[2] == self.num_groups):
                raise ValueError(
                    f"factored dequant requires scales of shape (batch, seq_len, {self.num_groups}), got {scales.shape}"
                )
            fc_scale = self.channelwise_scalings.to(quantized_states.device)  # (num_dims,)
            scales_flat = scales.reshape(-1, self.num_groups)  # (num_tokens, num_groups)

            # Expand per-group token scale to per-dim
            alpha_expanded = scales_flat.repeat_interleave(self._repeats.to(quantized_states.device), dim=1)  # (num_tokens, num_dims)

            effective_scale = fc_scale.unsqueeze(0) * alpha_expanded  # (num_tokens, num_dims)
            quantized_flat = quantized_states.reshape(-1, num_dims)
            pow2 = self._pow2_table.to(quantized_states.device)[bitwidths.long().clamp(0, 16)]
            stepsize = 2.0 * effective_scale / pow2
            dequantized_flat = quantized_flat.float() * stepsize
            dequantized_flat = dequantized_flat * (bitwidths != 0).to(dequantized_flat.dtype)
            # Restore whitening SC if present
            if self.channelwise_scaling is not None:
                sc = self.channelwise_scaling.to(dequantized_flat.device)
                dequantized_flat = dequantized_flat * sc.unsqueeze(0)
            dequantized_states = dequantized_flat.reshape(batch_size, seq_len, num_dims)
            if target_dtype is not None:
                dequantized_states = dequantized_states.to(target_dtype)
            return dequantized_states

        elif self.scaling_type == "kvtc":
            # scales shape: (batch_size, seq_len, num_kvtc_groups)
            if self.channelwise_scalings is None:
                raise ValueError("channelwise_scalings must be set for scaling_type='kvtc'")
            fc_scale = self.channelwise_scalings.to(quantized_states.device)  # (num_dims,)
            num_kvtc_groups = scales.shape[2]
            gs = self.kvtc_group_size
            group_sizes = [min(gs, num_dims - g * gs) for g in range(num_kvtc_groups)]
            repeats = torch.tensor(group_sizes, device=quantized_states.device, dtype=torch.long)
            scales_flat = scales.reshape(-1, num_kvtc_groups)  # (num_tokens, num_kvtc_groups)
            alpha_expanded = scales_flat.repeat_interleave(repeats, dim=1)  # (num_tokens, num_dims)
            effective_scale = fc_scale.unsqueeze(0) * alpha_expanded  # (num_tokens, num_dims)
            quantized_flat = quantized_states.reshape(-1, num_dims)
            pow2 = self._pow2_table.to(quantized_states.device)[bitwidths.long().clamp(0, 16)]
            stepsize = 2.0 * effective_scale / pow2
            dequantized_flat = quantized_flat.float() * stepsize
            dequantized_flat = dequantized_flat * (bitwidths != 0).to(dequantized_flat.dtype)
            if self.channelwise_scaling is not None:
                sc = self.channelwise_scaling.to(dequantized_flat.device)
                dequantized_flat = dequantized_flat * sc.unsqueeze(0)
            dequantized_states = dequantized_flat.reshape(batch_size, seq_len, num_dims)
            if target_dtype is not None:
                dequantized_states = dequantized_states.to(target_dtype)
            return dequantized_states

        # Tokenwise scaling (original behavior)
        if scales.dim() == 2:
            scales = scales.unsqueeze(-1).expand(batch_size, seq_len, self.num_groups)
            use_per_head = False
        elif scales.dim() == 3:
            if scales.shape[2] == self.num_heads and self.num_heads is not None:
                use_per_head = True
            elif scales.shape[2] == self.num_groups:
                use_per_head = False
            else:
                raise ValueError(
                    f"Scales shape mismatch: expected (batch_size, seq_len, {self.num_heads}) for per-head "
                    f"or (batch_size, seq_len, {self.num_groups}) for per-group, got {scales.shape}"
                )
        else:
            raise ValueError(f"Unexpected scales shape: {scales.shape}")
        
        quantized_flat = quantized_states.reshape(-1, num_dims)
        # bitwidths already (num_tokens, num_dims) from above

        if use_per_head:
            scales_flat = scales.reshape(-1, self.num_heads)
            repeats = torch.tensor(
                [
                    self.ranks[g] // self.num_heads_per_group
                    for g in range(self.num_groups)
                    for _ in range(self.num_heads_per_group)
                ],
                device=quantized_flat.device,
                dtype=torch.long,
            )
            scales_expanded = scales_flat.repeat_interleave(repeats, dim=1)
        else:
            scales_flat = scales.reshape(-1, self.num_groups)
            scales_expanded = scales_flat.repeat_interleave(self._repeats.to(quantized_flat.device), dim=1)

        pow2 = self._pow2_table.to(quantized_flat.device)[bitwidths.long().clamp(0, 16)]
        stepsize = 2.0 * scales_expanded / pow2
        dequantized_flat = quantized_flat.float() * stepsize
        dequantized_flat = dequantized_flat * (bitwidths != 0).to(dequantized_flat.dtype)
        
        # Restore channel-wise scaling SC if available (inverse whitening after dequantization)
        if self.channelwise_scaling is not None:
            sc = self.channelwise_scaling.to(dequantized_flat.device)
            if sc.numel() != num_dims:
                raise ValueError(
                    f"Channel-wise scaling size mismatch: expected {num_dims} dimensions, "
                    f"got {sc.numel()}"
                )
            # Apply per-dimension multiplication: v_restored = v_dequant * SC
            dequantized_flat = dequantized_flat * sc.unsqueeze(0)
        
        # Reshape back to original shape
        dequantized_states = dequantized_flat.reshape(batch_size, seq_len, num_dims)
        
        # Return as float32 (dequantization produces float32)
        # The caller will convert to the original dtype if needed
        return dequantized_states
    
    @staticmethod
    def from_linear_whiten(
        old_module: nn.Linear,
        ranks: list,
        uneven_split: bool = False,
    ):   
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i], uneven_split=uneven_split)
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        expected_vt_shape = (sum(ranks), old_module.in_features)
        if new_module.VT.weight.data.shape != VT_weight.shape:
            raise ValueError(
                f"[CRITICAL] VT weight shape mismatch in from_linear_whiten: "
                f"new_module.VT.weight.shape={new_module.VT.weight.data.shape}, "
                f"VT_weight.shape={VT_weight.shape}, "
                f"expected={expected_vt_shape}, "
                f"sum(ranks)={sum(ranks)}, ranks={ranks}"
            )
        if VT_weight.shape != expected_vt_shape:
            raise ValueError(
                f"[CRITICAL] VT_weight shape doesn't match expected: "
                f"VT_weight.shape={VT_weight.shape}, expected={expected_vt_shape}, "
                f"sum(ranks)={sum(ranks)}, ranks={ranks}"
            )
        new_module.VT.weight.data = VT_weight

        return new_module

    @staticmethod
    def from_linear_joint_pca(
        old_module: nn.Linear,
        ranks: list,
        joint_V_per_group: list,
    ):
        """
        Create HeadwiseLowRankModule from joint cross-layer KV PCA basis.

        joint_V_per_group[g] = V_l^g ∈ R^{group_dim × group_dim} — the per-layer
        slice of the joint eigenvector matrix for group g.  It will be truncated to
        ranks[g] columns inside this method.

          L_g = V_l^g[:, :rank_g]          → U[g].weight  (group_dim × rank)
          R_g = V_l^{g,:rank_g,T} @ W_g    → VT weight slice  (rank × in_features)

        Forward:  z_g = x @ R_g^T  =  x @ W_g^T @ V_l^g
                  out_g = z_g @ L_g^T  =  x @ W_g^T @ V_l^g @ V_l^{g,T}
        """
        num_groups = len(ranks)
        new_module = HeadwiseLowRankModule(
            ranks, old_module.in_features, old_module.out_features,
            bias=old_module.bias is not None,
        )
        # w[g] ∈ R^{group_dim × in_features}
        w = old_module.weight.data.reshape(num_groups, -1, old_module.in_features)
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(num_groups, -1)

        R_list = []
        for g in range(num_groups):
            rank_g = ranks[g]
            V_g = joint_V_per_group[g].to(w.device).to(torch.float32)
            # Truncate to rank_g principal components
            V_g = V_g[:, :rank_g]  # (group_dim, rank_g)
            w_g = w[g].to(torch.float32)  # (group_dim, in_features)

            L_g = V_g.to(old_module.weight.dtype)            # (group_dim, rank_g)
            R_g = (V_g.T @ w_g).to(old_module.weight.dtype)  # (rank_g, in_features)

            if new_module.U[g].weight.data.shape != L_g.shape:
                raise ValueError(
                    f"[from_linear_joint_pca] U[{g}] shape mismatch: "
                    f"{new_module.U[g].weight.data.shape} != {L_g.shape}"
                )
            new_module.U[g].weight.data = L_g.contiguous()
            if old_module.bias is not None:
                new_module.U[g].bias.data = b[g]
            R_list.append(R_g)

        VT_weight = torch.cat(R_list, dim=0).contiguous()
        expected_vt_shape = (sum(ranks), old_module.in_features)
        if VT_weight.shape != expected_vt_shape:
            raise ValueError(
                f"[from_linear_joint_pca] VT shape mismatch: "
                f"{VT_weight.shape} != {expected_vt_shape}"
            )
        new_module.VT.weight.data = VT_weight
        return new_module


# Optional: JIT-compile hot adaptive quantization paths for HeadwiseLowRankModule.
# Enable by setting PALU_TORCH_COMPILE_ADAPTIVE=1 in the environment. When disabled
# (the default), the module runs in standard eager mode.
_PALU_TORCH_COMPILE_ADAPTIVE = os.environ.get("PALU_TORCH_COMPILE_ADAPTIVE", "0") == "1"
if _PALU_TORCH_COMPILE_ADAPTIVE and hasattr(torch, "compile"):
    try:
        HeadwiseLowRankModule.quantize_with_per_dim_bitwidths = torch.compile(
            HeadwiseLowRankModule.quantize_with_per_dim_bitwidths, dynamic=True
        )
        HeadwiseLowRankModule.dequantize_with_per_dim_bitwidths = torch.compile(
            HeadwiseLowRankModule.dequantize_with_per_dim_bitwidths, dynamic=True
        )
    except Exception as _e:
        logger.warning(
            f"torch.compile for HeadwiseLowRankModule quant/dequant failed; "
            f"falling back to eager mode: {_e}"
        )
    
    @staticmethod
    def from_linear(
        old_module: nn.Linear,
        ranks: list,
    ):
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_decomposition_from_weight(w[i], ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]
        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        expected_vt_shape = (sum(ranks), old_module.in_features)
        if new_module.VT.weight.data.shape != VT_weight.shape:
            raise ValueError(
                f"[CRITICAL] VT weight shape mismatch in from_linear: "
                f"new_module.VT.weight.shape={new_module.VT.weight.data.shape}, "
                f"VT_weight.shape={VT_weight.shape}, "
                f"expected={expected_vt_shape}, "
                f"sum(ranks)={sum(ranks)}, ranks={ranks}"
            )
        if VT_weight.shape != expected_vt_shape:
            raise ValueError(
                f"[CRITICAL] VT_weight shape doesn't match expected: "
                f"VT_weight.shape={VT_weight.shape}, expected={expected_vt_shape}, "
                f"sum(ranks)={sum(ranks)}, ranks={ranks}"
            )
        new_module.VT.weight.data = VT_weight
        
        return new_module