import argparse
import importlib
import numpy as np
import random, torch
from functools import reduce
from palu.model import HeadwiseLowRankModule
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

# Set seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_model_numel(model):
    param_cnt = 0
    for name, module in model.named_modules():
        if hasattr(module, '_nelement'):
            param_cnt += module._nelement()
    return param_cnt

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**3
    return size_all_mb


def get_module_by_name(module, module_name):
    names = module_name.split(sep='.')
    return reduce(getattr, names, module)



def dump_to_huggingface_repos(model, tokenizer, save_path, args):
    tokenizer.save_pretrained(save_path)
    #model.generation_config = Gene
    #if "vicuna" in model.config._name_or_path.lower() or "longchat" in model.config._name_or_path.lower():
        #NOTE(brian1009): Ad-hoc fixing the bug in Vicuna
    #    model.config.generation_config = GenerationConfig(temperature=1.0, top_p=1.0)
    model.save_pretrained(save_path)
    config = model.config.to_dict()
    config["head_wise_ranks"] = {}
    for name, module in model.named_modules():
        if isinstance(module, HeadwiseLowRankModule):
            config["head_wise_ranks"][name] = module.ranks
    
    if "llama" in model.config._name_or_path.lower() or model.config.model_type == "llama":
        config["model_type"] = "palullama"
        config['architectures'] = ['PaluLlamaForCausalLM']
    elif "mistral" in model.config._name_or_path.lower():
        config["model_type"] = "palumistral"
        config['architectures'] = ['PaluMistralForCausalLM']
    elif "qwen2" in model.config._name_or_path.lower():
        config["model_type"] = "paluqwen2"
        config['architectures'] = ['PaluQwenForCausalLM']
    else:
        raise NotImplementedError
            
    config["original_model_name_or_path"] = model.config._name_or_path
    import json

    json.dump(config, open(save_path + "/config.json", "w"), indent=2)
    
    
def load_model_and_tokenizer(model_name_or_path, use_flash_attn2=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2" if use_flash_attn2 else "sdpa",
    )
    model.eval()
    # Fix the bug in generation configs
    #TODO: Add reference to the issue that also faced this bug
    if "vicuna" in model.config._name_or_path.lower() or "longchat" in model.config._name_or_path.lower():
        model.generation_config.do_sample = True
        
    return model, tokenizer



def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name_or_path', type=str, help='model to load')
    parser.add_argument('--lt_bits', type=int, help='Bits for low_rank latents. \
                        When lt_bits is less than 16, we quantize the low_rank latents in low_bits', default=16)
    parser.add_argument('--lt_group_size', type=int, help='Group size for low_rank latents', default=0)
    parser.add_argument('--lt_sym', action='store_true', help='Symmetric quantization for low_rank latents')
    parser.add_argument('--lt_clip_ratio', type=float, help='Clip ratio for low_rank latents', default=1.0)
    parser.add_argument('--lt_hadamard', action='store_true', help='Apply Hadamard transform to low_rank latents')
    parser.add_argument('--flash2', action='store_true', help='whether to use flash-attention2')
    return parser


def calculate_kv_cache_size(cache, model_config=None):
    """
    Calculate the size of the KV-cache in bytes and tokens.
    
    Args:
        cache: A cache object (DynamicCache, tuple, or dict)
        model_config: Optional model config to get num_heads and head_dim if not in cache
        
    Returns:
        dict with:
            - total_bytes: Total size in bytes
            - total_tokens: Total number of cached tokens (seq_len)
            - num_layers: Number of layers with cache
            - per_layer_bytes: Dict mapping layer_idx to size in bytes
            - dtype: Data type of the cache
            - total_mb: Total size in MB
            - total_gb: Total size in GB
    """
    from transformers.cache_utils import DynamicCache
    
    total_bytes = 0
    total_tokens = 0
    num_layers = 0
    per_layer_bytes = {}
    dtype = None
    
    # Convert to DynamicCache if needed
    if isinstance(cache, tuple):
        cache = DynamicCache.from_legacy_cache(cache)
    elif not isinstance(cache, DynamicCache):
        # Try to access as dict
        if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            pass  # Already has the structure
        else:
            return {
                "total_bytes": 0,
                "total_tokens": 0,
                "num_layers": 0,
                "per_layer_bytes": {},
                "dtype": None,
                "total_mb": 0,
                "total_gb": 0,
                "error": "Unknown cache format"
            }
    
    # Get cache structure
    if isinstance(cache, DynamicCache):
        key_cache = cache.key_cache
        value_cache = cache.value_cache
    else:
        key_cache = getattr(cache, 'key_cache', None)
        value_cache = getattr(cache, 'value_cache', None)
    
    if key_cache is None or value_cache is None:
        return {
            "total_bytes": 0,
            "total_tokens": 0,
            "num_layers": 0,
            "per_layer_bytes": {},
            "dtype": None,
            "total_mb": 0,
            "total_gb": 0,
            "error": "Cache has no key_cache or value_cache"
        }
    
    # Handle different cache formats
    if isinstance(key_cache, dict):
        layer_indices = key_cache.keys()
    elif isinstance(key_cache, list):
        layer_indices = range(len(key_cache))
    else:
        return {
            "total_bytes": 0,
            "total_tokens": 0,
            "num_layers": 0,
            "per_layer_bytes": {},
            "dtype": None,
            "total_mb": 0,
            "total_gb": 0,
            "error": f"Unknown key_cache type: {type(key_cache)}"
        }
    
    for layer_idx in layer_indices:
        try:
            # Get key and value states for this layer
            if isinstance(key_cache, dict):
                key_states = key_cache.get(layer_idx)
                value_states = value_cache.get(layer_idx) if isinstance(value_cache, dict) else None
            else:
                key_states = key_cache[layer_idx] if layer_idx < len(key_cache) else None
                value_states = value_cache[layer_idx] if isinstance(value_cache, list) and layer_idx < len(value_cache) else None
            
            if key_states is None:
                continue
                
            # Get dtype
            if dtype is None:
                dtype = key_states.dtype
            
            # Calculate size
            # Key states shape: (batch_size, num_heads, seq_len, head_dim)
            # Value states shape: (batch_size, num_heads, seq_len, head_dim)
            key_bytes = key_states.numel() * key_states.element_size()
            value_bytes = value_states.numel() * value_states.element_size() if value_states is not None else key_bytes
            
            layer_bytes = key_bytes + value_bytes
            total_bytes += layer_bytes
            per_layer_bytes[layer_idx] = layer_bytes
            
            # Get sequence length (number of cached tokens)
            if len(key_states.shape) >= 3:
                seq_len = key_states.shape[2]  # (batch, heads, seq_len, head_dim)
                if total_tokens == 0:  # Only set once (should be same for all layers)
                    total_tokens = seq_len
            
            num_layers += 1
            
        except (IndexError, KeyError, AttributeError) as e:
            continue
    
    return {
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "num_layers": num_layers,
        "per_layer_bytes": per_layer_bytes,
        "dtype": str(dtype) if dtype is not None else None,
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024)
    }