"""
Load bitwidth allocations from .pt files produced by run_bit_allocation.py.

Format expected from run_bit_allocation:
  torch.save({
    "bit_alloc": {layer_name: {head_idx: LongTensor (rank,)}, ...},
    "bit_alloc_k": {...},
    "bit_alloc_v": {...},
    "total_bits_k": ...,
    "total_bits_v": ...,
    ...
  }, path)
"""

import torch
from typing import Dict, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def load_bitwidth_allocations(bitwidth_file: str) -> Dict:
    """
    Load bitwidth allocations from a .pt file (from run_bit_allocation or compatible).

    Returns:
        bit_alloc: dict layer_name -> {head_idx: LongTensor (rank,)}, or layer_name -> LongTensor.
                   Merged from bit_alloc_k and bit_alloc_v if "bit_alloc" is not present.
    """
    data = torch.load(bitwidth_file, map_location="cpu")
    if "bit_alloc" in data:
        bit_alloc = data["bit_alloc"]
    elif "bit_alloc_k" in data and "bit_alloc_v" in data:
        bit_alloc = {**data["bit_alloc_k"], **data["bit_alloc_v"]}
        logger.info("Loaded bit_alloc from bit_alloc_k + bit_alloc_v")
    else:
        raise KeyError(
            f"Bitwidth file {bitwidth_file} must contain 'bit_alloc' or both 'bit_alloc_k' and 'bit_alloc_v'. "
            f"Keys found: {list(data.keys())}"
        )
    return bit_alloc


def get_bitwidths_for_layer(
    bit_alloc: Dict,
    layer_name: str,
    num_groups: int,
    group_rank: int,
) -> Optional[torch.Tensor]:
    """
    Get the per-dimension bitwidth tensor for a single layer.

    Args:
        bit_alloc: From load_bitwidth_allocations()
        layer_name: e.g. "model.layers.0.self_attn.k_proj"
        num_groups: module.num_groups
        group_rank: module.ranks[0] or similar (used only for validation)

    Returns:
        1D LongTensor of shape (sum(ranks),) or None if layer not found / invalid.
    """
    layer_bits = bit_alloc.get(layer_name)
    if layer_bits is None:
        return None
    if isinstance(layer_bits, torch.Tensor):
        return layer_bits.flatten()
    if isinstance(layer_bits, dict):
        tensors = [layer_bits[h] for h in sorted(layer_bits.keys()) if layer_bits[h] is not None]
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0].flatten()
        return torch.cat([t.flatten() for t in tensors])
    return None
