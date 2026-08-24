#!/usr/bin/env python3
"""
Script to compute channelwise scalings for quantization.

This script:
1. Loads calibration data
2. Runs forward passes through the model to collect latent states (A(x)) from HeadwiseLowRankModule layers
3. For each dimension in each layer, computes max absolute value across all tokens
4. Saves these scalings to a file for use during generation
"""

import argparse
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
import os
from loguru import logger
from palu.data_utils import get_calib_data
from utils import load_model_and_tokenizer


class ChannelwiseScalingCapture:
    """Captures latent states and computes per-dimension max absolute values."""
    
    def __init__(self, max_samples_per_dim=100000, percentile=99.0, use_percentile=True):
        self.max_abs_values = defaultdict(lambda: None)  # layer_name -> Tensor [num_dims]
        self.all_abs_values = defaultdict(lambda: None)  # layer_name -> List[Tensor] for percentile computation
        self.count = defaultdict(int)
        self.max_samples_per_dim = max_samples_per_dim  # Limit samples to avoid OOM
        self.percentile = percentile  # Percentile to use (e.g., 99.0 for 99th percentile)
        self.use_percentile = use_percentile  # If True, use percentile; if False, use max
    
    def register(self, module, name):
        """
        Register hook to capture latent states from HeadwiseLowRankModule.
        
        Args:
            module: HeadwiseLowRankModule instance
            name: Layer identifier (e.g., "model.layers.0.self_attn.k_proj")
        """
        def forward_hook(module_instance, inputs, output):
            """
            Hook into project_to_latent or VT output to capture A(x).
            output shape: (batch_size, seq_len, sum(ranks))
            """
            # Detach to save memory and avoid gradient computation
            latent_states = output.detach()  # (batch_size, seq_len, num_dims)
            
            # Reshape to (batch_size * seq_len, num_dims) to process all tokens
            if latent_states.dim() == 3:
                latent_flat = latent_states.reshape(-1, latent_states.shape[-1])
            elif latent_states.dim() == 2:
                latent_flat = latent_states
            else:
                logger.warning(f"Unexpected latent_states shape for {name}: {latent_states.shape}")
                latent_flat = latent_states.reshape(-1, latent_states.shape[-1]) if latent_states.numel() > 0 else latent_states
            
            # Compute absolute values per dimension across all tokens in this batch
            abs_batch = latent_flat.abs()  # (num_tokens, num_dims)
            
            if self.use_percentile:
                # Store all absolute values for percentile computation
                # Move to CPU immediately to save GPU memory
                abs_batch_cpu = abs_batch.detach().cpu()
                
                if self.all_abs_values[name] is None:
                    # Initialize: store as list of tensors (one per dimension) on CPU
                    num_dims = abs_batch_cpu.shape[-1]
                    self.all_abs_values[name] = [torch.tensor([], device='cpu', dtype=abs_batch_cpu.dtype) 
                                                 for _ in range(num_dims)]
                
                # Append samples for each dimension (limit to max_samples_per_dim to avoid OOM)
                for dim in range(abs_batch_cpu.shape[-1]):
                    dim_samples = abs_batch_cpu[:, dim]  # (num_tokens,)
                    current_count = len(self.all_abs_values[name][dim])
                    
                    if current_count >= self.max_samples_per_dim:
                        # Skip if we've collected enough samples
                        continue
                    
                    # Limit samples to max_samples_per_dim
                    if current_count + len(dim_samples) > self.max_samples_per_dim:
                        remaining = self.max_samples_per_dim - current_count
                        dim_samples = dim_samples[:remaining]
                    
                    self.all_abs_values[name][dim] = torch.cat([
                        self.all_abs_values[name][dim], 
                        dim_samples
                    ])
            else:
                # Original behavior: compute max absolute value per dimension
                max_abs_batch = abs_batch.max(dim=0)[0]  # (num_dims,)
                
                # Initialize or update running max
                if self.max_abs_values[name] is None:
                    self.max_abs_values[name] = max_abs_batch.clone()
                else:
                    # Update: take max of current max and new batch max
                    self.max_abs_values[name] = torch.maximum(
                        self.max_abs_values[name], 
                        max_abs_batch
                    )
        
            self.count[name] += 1
        
        # Hook into the VT layer (which produces A(x))
        if hasattr(module, 'VT'):
            module.VT.register_forward_hook(forward_hook)
        else:
            logger.warning(f"Module {name} does not have VT attribute, cannot register hook")
    
    def get_scalings(self):
        """
        Get computed scalings (max absolute values per dimension).
        
        Returns:
            Dictionary: layer_name -> Tensor [num_dims]
        """
        # Compute scalings (either percentile-based or max-based)
        scalings = {}
        if self.use_percentile:
            # Compute percentile per dimension
            for name, dim_samples_list in self.all_abs_values.items():
                if dim_samples_list is None or len(dim_samples_list) == 0:
                    continue
                
                num_dims = len(dim_samples_list)
                percentile_values = torch.zeros(num_dims, dtype=dim_samples_list[0].dtype)
                
                for dim in range(num_dims):
                    if len(dim_samples_list[dim]) == 0:
                        # Fallback to small value if no samples
                        percentile_values[dim] = 1e-8
                    else:
                        # Compute percentile for this dimension
                        percentile_values[dim] = torch.quantile(
                            dim_samples_list[dim].float(), 
                            self.percentile / 100.0
                        ).to(dim_samples_list[dim].dtype)
                
                scalings[name] = percentile_values.clamp(min=1e-8)
                logger.debug(f"Computed {self.percentile}th percentile scalings for {name}: "
                           f"shape={scalings[name].shape}, "
                           f"min={scalings[name].min().item():.4e}, "
                           f"max={scalings[name].max().item():.4e}, "
                           f"mean={scalings[name].mean().item():.4e}")
        else:
            # Original behavior: use max absolute values
            for name, max_abs in self.max_abs_values.items():
                scalings[name] = max_abs.clamp(min=1e-8)
        return scalings


def run_channelwise_scaling_calibration(
    model, 
    tokenizer, 
    device, 
    nsamples=256, 
    seqlen=2048,
    calib_dataset="wikitext2",
    cache_file=None, 
    use_cache=True,
    percentile=99.0,
    use_percentile=True
):
    """
    Run calibration to compute channelwise scalings.
    
    Args:
        model: The compressed model with HeadwiseLowRankModule layers
        tokenizer: Tokenizer for the model
        device: Device to run calibration on
        nsamples: Number of calibration samples
        seqlen: Sequence length for calibration
        calib_dataset: Calibration dataset name
        cache_file: Optional path to cache file. If None, auto-generates based on model_id
        use_cache: Whether to use cached results if available
        percentile: Percentile to use for scaling (if use_percentile=True)
        use_percentile: Whether to use percentile-based scaling (vs max-based)
        
    Returns:
        Dictionary: layer_name -> Tensor [num_dims] with max absolute values per dimension
    """
    model_id = model.config._name_or_path
    
    # Generate cache file path if not provided
    if cache_file is None:
        os.makedirs("cache", exist_ok=True)
        if use_percentile:
            suffix = f"p{percentile:.1f}"  # e.g., "p99.0"
        else:
            suffix = "max"
        cache_file = f"cache/channelwise_scalings_{suffix}_{model_id.replace('/', '_')}_{calib_dataset}_{nsamples}_{seqlen}.pt"
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        logger.info(f"[Channelwise Scaling] Cache file exists: {cache_file}", fg="yellow")
        logger.info(f"[Channelwise Scaling] Loading scalings from cache...", fg="yellow")
        scalings = torch.load(cache_file, map_location="cpu")
        logger.info(f"[Channelwise Scaling] Loaded scalings for {len(scalings)} layers", fg="green")
        return scalings
    
    logger.info(f"[Channelwise Scaling] No cache file found. Running calibration...", fg="yellow")
    
    # Set model to eval mode
    model.eval()
    model.to(device)
    
    # Get calibration data
    calib_loader = get_calib_data(
        calib_dataset,
        tokenizer,
        model_id,
        nsamples=nsamples,
        seqlen=seqlen
    )
    
    # Create capture object
    capture = ChannelwiseScalingCapture(
        max_samples_per_dim=100000,
        percentile=percentile,
        use_percentile=use_percentile,
    )
    
    # Register hooks for all HeadwiseLowRankModule instances
    num_registered = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == "HeadwiseLowRankModule":
            capture.register(module, name)
            num_registered += 1
            logger.debug(f"Registered hook for {name}: ranks={module.ranks}, total_dims={sum(module.ranks)}")
    
    if num_registered == 0:
        raise RuntimeError("No HeadwiseLowRankModule instances found in model!")
    
    logger.info(f"Registered hooks for {num_registered} HeadwiseLowRankModule instances")
    
    # Run forward passes through calibration data
    logger.info(f"Processing {len(calib_loader)} calibration samples...")
    with torch.no_grad():
        for batch in tqdm(calib_loader, desc="Calibrating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass - this will trigger the hooks
            outputs = model(**batch)
    
    # Get computed scalings
    scalings = capture.get_scalings()
    
    # Verify we collected scalings
    if len(scalings) == 0:
        raise RuntimeError("No scalings collected! Hooks may not have been called.")
    
    # Log some statistics
    logger.info(f"Collected scalings for {len(scalings)} layers")
    for i, (name, scaling) in enumerate(list(scalings.items())[:3]):
        logger.info(f"  {name}: shape={scaling.shape}, min={scaling.min().item():.2e}, "
                   f"max={scaling.max().item():.2e}, mean={scaling.mean().item():.2e}")
    
    # Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    torch.save(scalings, cache_file)
    logger.info(f"[Channelwise Scaling] Scalings saved to {cache_file}", fg="green")
    
    return scalings


def main():
    parser = argparse.ArgumentParser(
        description="Compute channelwise scalings for quantization from calibration data"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the compressed model (local path or HuggingFace model ID)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run calibration on"
    )
    
    parser.add_argument(
        "--nsamples",
        type=int,
        default=256,
        help="Number of calibration samples"
    )
    
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length for calibration"
    )
    
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        help="Calibration dataset name (wikitext2, c4, fineweb, openr1math, fineweb_openr1math, or LongBench dataset)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save scalings. If None, uses default cache location"
    )
    
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cached scalings if available"
    )
    
    parser.add_argument(
        "--flash2",
        action="store_true",
        help="Use flash attention 2"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information"
    )
    
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile to use for channelwise scaling (default: 99.0 for 99th percentile). "
             "Lower values (e.g., 95.0) use smaller alpha, improving quantization accuracy but risking overflow. "
             "Higher values (e.g., 99.9) are more conservative."
    )
    
    parser.add_argument(
        "--use_max",
        action="store_true",
        help="Use max absolute value instead of percentile (original behavior, less accurate)"
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level="INFO" if not args.verbose else "DEBUG"
    )
    
    # Load the compressed model
    logger.info(f"Loading compressed model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, use_flash_attn2=args.flash2)
    
    # Run calibration
    scalings = run_channelwise_scaling_calibration(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        calib_dataset=args.calib_dataset,
        cache_file=args.output_path,
        use_cache=args.use_cache,
        percentile=args.percentile,
        use_percentile=not args.use_max,
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Channelwise scaling calibration completed!")
    logger.info(f"Scalings computed for {len(scalings)} layers")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
