#!/usr/bin/env python3
"""
Example script for generation with per-dimension bitwidth quantization.

This script demonstrates how to:
1. Load a compressed model
2. Configure quantization with bitwidth allocations
3. Perform generation using QuantizedCache
"""

import argparse
import torch
from transformers import AutoTokenizer
from loguru import logger
from tqdm import tqdm

from utils import load_model_and_tokenizer
from palu.quantization_utils import load_bitwidth_allocations
from transformers.cache_utils import DynamicCache


def configure_model_quantization(model, bitwidth_file: str, scaling_type: str = "tokenwise",
                                  channelwise_scaling_file: str = None,
                                  channel_group_size: int = 64, kvtc_group_size: int = 64):
    """
    Configure quantization for all attention layers in the model.

    Args:
        model: The model to configure
        bitwidth_file: Path to the .pt file with bitwidth allocations
        scaling_type: "tokenwise" (default), "channelwise", "channel_group", "factored", or "kvtc"
        channelwise_scaling_file: Path to .pt file with per-channel scalings.
            Required for scaling_type in ("channelwise", "channel_group", "factored", "kvtc").
        channel_group_size: Bucket size G for "channel_group" scaling (default 64).
        kvtc_group_size: Dims per adaptive-scale group for "kvtc" scaling (default 64).
    """
    logger.info(f"Loading bitwidth allocations from {bitwidth_file}...")
    bit_alloc = load_bitwidth_allocations(bitwidth_file)
    
    # Load channelwise scalings if needed
    channelwise_scalings_dict = None
    if scaling_type in ("channelwise", "channel_group", "factored", "kvtc"):
        if channelwise_scaling_file is None:
            raise ValueError(f"channelwise_scaling_file must be provided when scaling_type='{scaling_type}'")
        logger.info(f"Loading channelwise scalings from {channelwise_scaling_file}...")
        channelwise_scalings_dict = torch.load(channelwise_scaling_file, map_location="cpu")
        logger.info(f"Loaded channelwise scalings for {len(channelwise_scalings_dict)} layers")
    
    # Print available layer names for debugging
    logger.info(f"Available layer names in bitwidth file: {list(bit_alloc.keys())[:5]}...")
    
    logger.info("Configuring quantization for attention layers...")
    num_configured = 0
    
    # First, let's see what module names actually exist in the model
    logger.info("Checking model structure...")
    model_module_names = []
    all_kv_proj_modules = []
    
    # Check for HeadwiseLowRankModule (from either kernel/palu_attention.py or palu/model/modules/svd_linear.py)
    for name, module in model.named_modules():
        module_type = type(module).__name__
        # Check if it's a HeadwiseLowRankModule by class name or by having the quantization method
        is_headwise = (module_type == 'HeadwiseLowRankModule' or 
                      (hasattr(module, 'configure_per_dim_quantization') and 
                       hasattr(module, 'use_per_dim_quantization')))
        
        if is_headwise:
            model_module_names.append((name, module_type))
        
        # Also collect all k_proj and v_proj modules to see what we're working with
        if name.endswith('.k_proj') or name.endswith('.v_proj'):
            all_kv_proj_modules.append((name, module_type))
    
    if len(model_module_names) == 0:
        logger.warning("No HeadwiseLowRankModule instances found in model!")
        logger.warning(f"Found {len(all_kv_proj_modules)} k_proj/v_proj modules:")
        for name, mtype in all_kv_proj_modules[:10]:
            logger.warning(f"  {name}: {mtype}")
        logger.warning("The model might not be using Palu attention layers.")
        logger.warning("You may need to convert the model to use Palu attention first.")
        return bit_alloc
    
    logger.info(f"Found {len(model_module_names)} HeadwiseLowRankModule instances")
    
    # Directly configure k_proj and v_proj modules (HeadwiseLowRankModule instances)
    logger.info("Attempting to configure quantization...")
    for name, module in model.named_modules():
        # Check if this is a HeadwiseLowRankModule (from either location)
        module_type = type(module).__name__
        is_headwise = (module_type == 'HeadwiseLowRankModule' or 
                      (hasattr(module, 'configure_per_dim_quantization') and 
                       hasattr(module, 'use_per_dim_quantization')))
        
        if is_headwise:
            logger.debug(f"Processing module: {name}")
            # Try to find matching bitwidth allocation
            # The name should be something like "model.layers.0.self_attn.k_proj"
            if name in bit_alloc:
                logger.debug(f"  Found in bit_alloc: {name}")
                # Get bitwidths for this layer
                from palu.quantization_utils import get_bitwidths_for_layer
                
                # Determine num_groups and group_rank from the module
                if hasattr(module, 'num_groups') and hasattr(module, 'ranks'):
                    num_groups = module.num_groups
                    # Calculate group_rank - it's the rank per group
                    if len(module.ranks) > 0:
                        group_rank = module.ranks[0]  # All groups should have same rank
                    else:
                        total_rank = sum(module.ranks) if module.ranks else 0
                        group_rank = total_rank // num_groups if num_groups > 0 else 0
                    
                    if group_rank > 0:
                        bitwidths = get_bitwidths_for_layer(bit_alloc, name, num_groups, group_rank)
                        if bitwidths is not None:
                            # Get num_heads from the attention module for per-head scaling
                            num_heads = None
                            try:
                                # First try to get from model config (most reliable)
                                if hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
                                    num_heads = model.config.num_attention_heads
                                else:
                                    # Fallback: Extract layer path: "model.layers.0.self_attn"
                                    parts = name.split('.')
                                    if len(parts) >= 3 and parts[-1] in ['k_proj', 'v_proj']:
                                        attn_path = '.'.join(parts[:-1])  # Remove 'k_proj' or 'v_proj'
                                        attn_module = dict(model.named_modules()).get(attn_path)
                                        if attn_module is not None:
                                            # Try different ways to get num_heads
                                            if hasattr(attn_module, 'num_heads'):
                                                num_heads = attn_module.num_heads
                                            elif hasattr(attn_module, 'config') and hasattr(attn_module.config, 'num_attention_heads'):
                                                num_heads = attn_module.config.num_attention_heads
                                            else:
                                                logger.warning(f"  Could not find num_heads in {attn_path}. Available attributes: {[a for a in dir(attn_module) if not a.startswith('_')][:10]}")
                                        else:
                                            logger.warning(f"  Attention module not found at path: {attn_path}")
                            except Exception as e:
                                logger.warning(f"  Could not get num_heads: {e}")
                            
                            # Get channelwise scalings for this layer
                            channelwise_scalings = None
                            actual_scaling_type = scaling_type
                            
                            if scaling_type == "channelwise":
                                if channelwise_scalings_dict is not None:
                                    if name in channelwise_scalings_dict:
                                        channelwise_scalings = channelwise_scalings_dict[name]
                                        logger.debug(f"  Using channelwise scalings for {name}: shape={channelwise_scalings.shape}")
                                    else:
                                        logger.warning(f"  Channelwise scalings not found for {name}, falling back to tokenwise")
                                        actual_scaling_type = "tokenwise"
                                else:
                                    logger.warning(f"  Channelwise scalings dict not loaded, falling back to tokenwise")
                                    actual_scaling_type = "tokenwise"

                            elif scaling_type in ("channel_group", "factored", "kvtc"):
                                # Both new variants use the same channelwise_scalings file format
                                if channelwise_scalings_dict is not None:
                                    if name in channelwise_scalings_dict:
                                        channelwise_scalings = channelwise_scalings_dict[name]
                                        logger.debug(f"  Using channelwise scalings for {name} ({scaling_type}): shape={channelwise_scalings.shape}")
                                    else:
                                        logger.warning(f"  Channelwise scalings not found for {name}, falling back to tokenwise")
                                        actual_scaling_type = "tokenwise"
                                else:
                                    logger.warning(f"  Channelwise scalings dict not loaded, falling back to tokenwise")
                                    actual_scaling_type = "tokenwise"

                            # Configure quantization with scaling type
                            module.configure_per_dim_quantization(
                                bitwidths,
                                num_heads=num_heads,
                                scaling_type=actual_scaling_type,
                                channelwise_scalings=channelwise_scalings,
                                channel_group_size=channel_group_size,
                                kvtc_group_size=kvtc_group_size,
                            )

                            num_configured += 1
                        else:
                            logger.warning(f"✗ Could not get bitwidths for {name} (groups={num_groups}, rank={group_rank})")
                            # Show what's actually in bit_alloc for this layer
                            layer_bits = bit_alloc.get(name, {})
                            if isinstance(layer_bits, dict):
                                logger.warning(f"  Layer has {len(layer_bits)} heads in bit_alloc")
                                if len(layer_bits) > 0:
                                    first_head = list(layer_bits.keys())[0]
                                    first_bits = layer_bits[first_head]
                                    logger.warning(f"  First head ({first_head}) bitwidths shape: {first_bits.shape if hasattr(first_bits, 'shape') else type(first_bits)}")
                                    logger.warning(f"  Total dimensions across all heads: {sum(b.numel() if hasattr(b, 'numel') else len(b) for b in layer_bits.values())}")
                            else:
                                logger.warning(f"  Layer bitwidths type: {type(layer_bits)}")
                    else:
                        logger.warning(f"Skipping {name}: invalid group_rank={group_rank}")
                else:
                    logger.debug(f"Skipping {name}: missing num_groups or ranks attributes")
            else:
                # Name not found in bit_alloc - log for debugging
                if name.endswith('.k_proj') or name.endswith('.v_proj'):
                    logger.warning(f"✗ Module {name} not found in bit_alloc")
                    logger.warning(f"  Looking for entries ending with: {name.split('.')[-2]}.{name.split('.')[-1]}")
                    # Show similar entries
                    similar = [k for k in bit_alloc.keys() if name.split('.')[-1] in k]
                    if similar:
                        logger.warning(f"  Similar entries found: {similar[:3]}")
                    else:
                        logger.warning(f"  No similar entries found in bit_alloc")
    
    logger.info(f"Configured quantization for {num_configured} attention layers")
    if num_configured == 0:
        logger.warning("No layers were configured! Available layer names in bitwidth file:")
        for layer_name in list(bit_alloc.keys())[:10]:
            logger.warning(f"  - {layer_name}")
        logger.warning("Check that layer names match your model structure.")
    
    return bit_alloc


def generate_with_quantized_cache(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cuda",
    disable_quantization: bool = False
):
    """
    Generate text using manual generation loop with standard DynamicCache.
    
    The quantization happens inside HeadwiseLowRankModule.forward() as:
    A(x) → Quantize → Dequantize → BA(x)
    
    This uses standard DynamicCache and stores BA(x), just like the standard path.
    The only difference is that A(x) is optionally quantized before reconstruction.
    
    This method provides more control over generation (temperature, top_p, etc.)
    compared to model.generate(), but both methods support quantization.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        device: Device to run on
        disable_quantization: If True, disable quantization (use full-precision A(x))
        
    Returns:
        Generated text
    """
    logger.info("=" * 80)
    logger.info("GENERATION PATH: generate_with_quantized_cache() - Using manual generation loop")
    logger.info("=" * 80)
    model.eval()
    # Don't call model.to(device) if using device_map="auto" (which load_model_and_tokenizer uses)
    # The model is already on the correct device(s)
    
    # Get the device from the model's first parameter
    device = next(model.parameters()).device
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    prompt_length = input_ids.shape[1]
    
    # Check if quantization is configured (bitwidths are set, regardless of use_per_dim_quantization flag)
    has_quantization_configured = any(
        hasattr(m, 'bitwidths') and m.bitwidths is not None
        for m in model.modules()
    )
    
    # Check if quantization is currently enabled (use_per_dim_quantization flag is True)
    has_quantization_enabled = any(
        hasattr(m, 'use_per_dim_quantization') and m.use_per_dim_quantization
        for m in model.modules()
    )
    
    # Temporarily disable quantization in modules if requested
    if disable_quantization and has_quantization_enabled:
        # Store original settings
        quantization_settings = {}
        for name, module in model.named_modules():
            if hasattr(module, 'use_per_dim_quantization') and module.use_per_dim_quantization:
                quantization_settings[name] = True
                module.use_per_dim_quantization = False
        
        logger.info("Quantization DISABLED - using full-precision A(x) (no quantization/dequantization)")
    else:
        quantization_settings = {}
        if has_quantization_enabled:
            logger.info("Quantization ENABLED - A(x) will be quantized/dequantized before reconstruction")
        elif has_quantization_configured:
            logger.info("Quantization CONFIGURED but DISABLED - using full-precision A(x)")
        else:
            logger.info("No quantization configured - using standard path")
    
    # Use standard DynamicCache (no custom cache needed)
    from transformers.cache_utils import DynamicCache
    past_key_values = DynamicCache()
    
    logger.info(f"Generating {max_new_tokens} tokens with standard DynamicCache...")
    
    # Process prompt (first forward pass - this processes the entire prompt)
    with torch.no_grad():
        
        # Forward through the model with the prompt
        # Quantization happens inside HeadwiseLowRankModule.forward() if enabled
        # The cache stores BA(x) (after reconstruction), just like standard path
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=past_key_values,
        )
        
        # Get the cache from the outputs (standard DynamicCache)
        if isinstance(outputs.past_key_values, tuple):
            # Convert tuple to DynamicCache
            past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
        elif isinstance(outputs.past_key_values, DynamicCache):
            past_key_values = outputs.past_key_values
        else:
            # Unknown format - try to use it
            past_key_values = outputs.past_key_values
        
        cache_length = past_key_values.get_seq_length(0) if hasattr(past_key_values, 'get_seq_length') else 'unknown'
        logger.info(f"Prompt processed - cache length: {cache_length}")
        
        # Calculate and log KV-cache size
        from utils import calculate_kv_cache_size
        cache_stats = calculate_kv_cache_size(past_key_values)
        if 'error' not in cache_stats:
            logger.info(f"KV-Cache size: {cache_stats['total_mb']:.2f} MB ({cache_stats['total_gb']:.3f} GB)")
            logger.info(f"  - Cached tokens: {cache_stats['total_tokens']}")
            logger.info(f"  - Layers: {cache_stats['num_layers']}")
            logger.info(f"  - Dtype: {cache_stats['dtype']}")
        
        # Get logits for the last token
        logits = outputs.logits[:, -1, :]
        
        # Generate tokens one by one
        generated_ids = input_ids.clone()
        
        for step in tqdm(range(max_new_tokens), desc="Generating"):
            # Sample next token
            if temperature > 0:
                logits_scaled = logits / temperature
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_scaled, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits_scaled[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Debug: Log first few tokens to detect repetition
            if step < 5:
                token_str = tokenizer.decode(next_token[0])
                top_3_logits, top_3_indices = logits[0].topk(3)
                top_3_tokens = [tokenizer.decode([idx.item()]) for idx in top_3_indices]
                logger.debug(f"Step {step}: token='{token_str}' (id={next_token.item()}), top3={top_3_tokens}, top3_logits={[f'{l:.3f}' for l in top_3_logits.tolist()]}")
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS token
            # Note: We continue generating even if EOS is hit, to reach max_new_tokens
            # If you want to stop on EOS, uncomment the break statement below
            if next_token.item() == tokenizer.eos_token_id:
                logger.debug(f"EOS token generated at step {step+1}/{max_new_tokens}")
                # Uncomment to stop generation on EOS:
                # break
            
            # Forward pass with the new token (q_len == 1)
            # Quantization happens inside HeadwiseLowRankModule.forward() if enabled
            # The cache stores BA(x) (after reconstruction), just like standard path
            outputs = model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past_key_values,
            )
            
            # Update past_key_values for next iteration (standard DynamicCache handling)
            if isinstance(outputs.past_key_values, tuple):
                # Convert tuple to DynamicCache
                past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
            elif isinstance(outputs.past_key_values, DynamicCache):
                past_key_values = outputs.past_key_values
            else:
                past_key_values = outputs.past_key_values
            
            logits = outputs.logits[:, -1, :]
            
            # Log cache size every 10 steps
            if (step + 1) % 10 == 0:
                from utils import calculate_kv_cache_size
                cache_stats = calculate_kv_cache_size(past_key_values)
                if 'error' not in cache_stats:
                    logger.debug(f"Step {step+1}: KV-Cache size: {cache_stats['total_mb']:.2f} MB ({cache_stats['total_tokens']} tokens)")
    
    # Restore quantization settings if they were disabled
    if quantization_settings:
        for name, module in model.named_modules():
            if name in quantization_settings:
                module.use_per_dim_quantization = True
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
    return generated_text


def generate_simple(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cuda"
):
    """
    Simple generation using model.generate().
    
    With the simplified approach, quantization happens automatically in HeadwiseLowRankModule.forward()
    if use_per_dim_quantization=True, so this method supports quantization without any special handling.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run on (ignored if model uses device_map)
        
    Returns:
        Generated text
    """
    logger.info("=" * 80)
    logger.info("GENERATION PATH: generate_simple() - Using model.generate()")
    logger.info("=" * 80)
    
    # Check if quantization is enabled
    has_quantization_enabled = any(
        hasattr(m, 'use_per_dim_quantization') and m.use_per_dim_quantization
        for m in model.modules()
    )
    if has_quantization_enabled:
        logger.info("✓ Quantization is ENABLED - HeadwiseLowRankModule.forward() will quantize A(x)")
        logger.info("  Quantization happens automatically during model.generate() forward passes")
    else:
        logger.info("⚠ Quantization is DISABLED - using full-precision A(x)")
    
    model.eval()
    # Don't call model.to(device) if using device_map="auto"
    # Get the device from the model's first parameter
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]
    
    # Validate top_p (should be between 0 and 1)
    if top_p < 0 or top_p > 1:
        logger.warning(f"top_p={top_p} is invalid (should be 0-1). Clamping to [0, 1]")
        top_p = max(0.0, min(1.0, top_p))
    
    # Always enable sampling when temperature > 0 or top_p < 1.0
    # Default temperature is 1.0 and top_p is 0.9, so sampling is enabled by default
    do_sample = temperature > 0 or top_p < 1.0
    
    # Override model's generation_config BEFORE building kwargs to ensure it's set
    if do_sample and hasattr(model, 'generation_config'):
        model.generation_config.do_sample = True
        # Also set temperature and top_p in generation_config if they're valid
        if temperature > 0:
            model.generation_config.temperature = float(temperature)
        if top_p < 1.0:
            model.generation_config.top_p = float(top_p)
    
    with torch.no_grad():
        # Build generation kwargs - always set do_sample explicitly
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,  # Use the boolean directly
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        # Add sampling parameters ONLY if sampling is enabled
        if do_sample:
            if temperature > 0:
                generate_kwargs["temperature"] = float(temperature)
            if top_p < 1.0:
                generate_kwargs["top_p"] = float(top_p)
        
        logger.info(f"Generation kwargs: do_sample={generate_kwargs.get('do_sample')}, temperature={generate_kwargs.get('temperature', 'N/A')}, top_p={generate_kwargs.get('top_p', 'N/A')}")
        if hasattr(model, 'generation_config'):
            logger.info(f"Model generation_config: do_sample={getattr(model.generation_config, 'do_sample', 'N/A')}, temperature={getattr(model.generation_config, 'temperature', 'N/A')}, top_p={getattr(model.generation_config, 'top_p', 'N/A')}")
        
        outputs = model.generate(**generate_kwargs)
    
    # Calculate KV-cache size estimate based on model config and sequence length
    # Since model.generate() doesn't expose the cache directly, we estimate it
    generated_length = outputs.shape[1] - prompt_length
    total_tokens = outputs.shape[1]  # Total sequence length
    
    if hasattr(model, 'config'):
        from utils import calculate_kv_cache_size
        
        num_layers = getattr(model.config, 'num_hidden_layers', None)
        num_heads = getattr(model.config, 'num_attention_heads', None)
        head_dim = getattr(model.config, 'head_dim', None)
        
        if num_layers and num_heads:
            if head_dim is None:
                hidden_size = getattr(model.config, 'hidden_size', 4096)
                head_dim = hidden_size // num_heads
            
            # Cache stores keys and values: 2 * (batch * num_heads * seq_len * head_dim) per layer
            # Assuming float16 (2 bytes per element)
            bytes_per_token_per_layer = 2 * num_heads * head_dim * 2  # 2 for K+V, 2 bytes for float16
            estimated_bytes = bytes_per_token_per_layer * total_tokens * num_layers
            estimated_mb = estimated_bytes / (1024 * 1024)
            estimated_gb = estimated_bytes / (1024 * 1024 * 1024)
            
            logger.info(f"KV-Cache size (estimated): {estimated_mb:.2f} MB ({estimated_gb:.3f} GB)")
            logger.info(f"  - Total tokens: {total_tokens} (prompt: {prompt_length}, generated: {generated_length})")
            logger.info(f"  - Layers: {num_layers}, Heads: {num_heads}, Head_dim: {head_dim}")
            logger.info(f"  - Note: This is an estimate. Actual cache size may vary with quantization.")
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with per-dimension bitwidth quantization"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the compressed model"
    )
    
    parser.add_argument(
        "--bitwidth_file",
        type=str,
        required=True,
        help="Path to the .pt file with bitwidth allocations (from run_bit_allocation)"
    )
    
    parser.add_argument(
        "--scaling_type",
        type=str,
        default="tokenwise",
        choices=["tokenwise", "channelwise", "channel_group", "factored", "kvtc"],
        help="Scaling type: 'tokenwise' (one scale per token), 'channelwise' (one static scale per dim), "
             "'channel_group' (one scale per dim per G-token bucket), "
             "'factored' (scale = channel_scale * token_scale, O(T+D) storage), "
             "'kvtc' (like factored but adaptive scale per kvtc_group_size dims, matching KVTC paper)"
    )

    parser.add_argument(
        "--channel_group_size",
        type=int,
        default=64,
        help="Bucket size G for --scaling_type=channel_group (default 64)"
    )

    parser.add_argument(
        "--kvtc_group_size",
        type=int,
        default=64,
        help="Dims per adaptive-scale group for --scaling_type=kvtc (default 64)"
    )

    parser.add_argument(
        "--channelwise_scaling_file",
        type=str,
        default=None,
        help="Path to .pt file with per-channel scalings. Required for "
             "--scaling_type in {channelwise, channel_group, factored}."
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Input prompt for generation"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    
    parser.add_argument(
        "--use_manual_loop",
        action="store_true",
        help="Use manual generation loop instead of model.generate() (more control, same quantization behavior)"
    )

    parser.add_argument(
        "--disable_quantization",
        action="store_true",
        help="Disable quantization (use full-precision A(x) instead of quantized A(x))"
    )
    
    parser.add_argument(
        "--flash2",
        action="store_true",
        help="Use flash attention 2"
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, use_flash_attn2=args.flash2)
    
    # Validate arguments
    if args.scaling_type == "channelwise" and args.channelwise_scaling_file is None:
        raise ValueError("--channelwise_scaling_file must be provided when --scaling_type=channelwise")
    
    # Configure quantization
    # Quantization happens directly in HeadwiseLowRankModule.forward() via use_per_dim_quantization flag.
    logger.info(f"Configuring quantization with scaling_type={args.scaling_type}...")
    configure_model_quantization(
        model,
        args.bitwidth_file,
        scaling_type=args.scaling_type,
        channelwise_scaling_file=args.channelwise_scaling_file,
        channel_group_size=args.channel_group_size,
        kvtc_group_size=args.kvtc_group_size,
    )
    
    # With simplified approach, quantization is controlled via use_per_dim_quantization flag
    # If disable_quantization is set, temporarily disable quantization in modules
    if args.disable_quantization:
        logger.info("Disabling quantization in HeadwiseLowRankModule instances...")
        num_disabled = 0
        for name, module in model.named_modules():
            if hasattr(module, 'use_per_dim_quantization') and module.use_per_dim_quantization:
                module.use_per_dim_quantization = False
                num_disabled += 1
        logger.info(f"Disabled quantization in {num_disabled} module(s)")
    
    # Verify quantization is configured
    has_quantization = False
    quantized_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'use_per_dim_quantization') and module.use_per_dim_quantization:
            has_quantization = True
            quantized_modules.append(name)
    
    if has_quantization:
        logger.info(f"Quantization enabled for {len(quantized_modules)} modules")
        logger.info(f"Example modules: {quantized_modules[:3]}")
    else:
        logger.warning("No quantization configured! Make sure bitwidth_file contains matching layer names.")
        logger.warning("Layer names should match the model structure (e.g., 'model.layers.0.self_attn.k_proj')")
        logger.warning("The script will continue but quantization will not be applied.")
    
    # Generate
    logger.info(f"Generating with prompt: '{args.prompt}'")
    
    use_manual_loop = args.use_manual_loop 
    # With simplified approach, quantization happens automatically in HeadwiseLowRankModule.forward()
    # if use_per_dim_quantization=True. Both generation methods support quantization.
    if use_manual_loop:
        if args.disable_quantization:
            logger.info("Using manual generation loop (quantization DISABLED - using full precision A(x))")
        else:
            logger.info("Using manual generation loop (quantization ENABLED - A(x) will be quantized/dequantized)")
        generated_text = generate_with_quantized_cache(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            disable_quantization=args.disable_quantization
        )
    else:
        # Use model.generate() - quantization still works automatically if configured
        if args.disable_quantization:
            logger.info("Using model.generate() (quantization DISABLED - using full precision A(x))")
        else:
            logger.info("Using model.generate() (quantization ENABLED if configured - A(x) will be quantized/dequantized)")
        generated_text = generate_simple(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("Generated text:")
    logger.info("=" * 80)
    logger.info(generated_text)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
