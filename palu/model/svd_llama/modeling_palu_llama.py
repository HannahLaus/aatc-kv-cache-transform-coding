from transformers import LlamaForCausalLM
import torch.nn as nn
import re
from types import SimpleNamespace
from .configuration_palu_llama import PaluLlamaConfig
from ..modules.svd_linear import HeadwiseLowRankModule
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class PaluLlamaForCausalLM(LlamaForCausalLM):
    config_class = PaluLlamaConfig
    def __init__(self, config:PaluLlamaConfig):
        super().__init__(config)
        self.head_wise_ranks=config.head_wise_ranks
        self._reinit_callback = None  # Callback to re-patch modules after re-initialization

        full_name_dict = {module: name for name, module in self.named_modules()}
        linear_info = {}
        modules = [self]
        while len(modules) > 0:
            submodule = modules.pop()
            for name, raw_linear in submodule.named_children():
                if isinstance(raw_linear, nn.Linear):
                    full_name = full_name_dict[raw_linear]
                    linear_info[raw_linear] = {
                        "father": submodule,
                        "name": name,
                        "full_name": full_name,
                    }
                else:
                    modules.append(raw_linear)


        # Create replacement modules for all layers that need to be replaced
        # IMPORTANT: Iterate over linear_info (which contains the ORIGINAL modules) 
        # to ensure we replace each module exactly once
        created_modules = {}  # Track created modules to verify they're unique
        for original_module, info in linear_info.items():
            full_name = info["full_name"]
            if full_name in self.head_wise_ranks:
                # Create a NEW HeadwiseLowRankModule instance for each layer
                expected_ranks = self.head_wise_ranks[full_name]
                new_layer = HeadwiseLowRankModule(
                    expected_ranks,
                    original_module.in_features,
                    original_module.out_features,
                    bias=original_module.bias is not None
                )
                # Store expected ranks and layer name for validation during forward
                # Store as tuple (immutable) to make it harder to accidentally modify
                # This helps detect if the module was replaced during checkpoint loading
                new_layer._expected_ranks = tuple(expected_ranks)  # Store as tuple (immutable)
                new_layer._layer_name = full_name
                new_layer._init_module_id = id(new_layer)  # Store module ID at init time
                # Store weak reference to model to access head_wise_ranks (can't be overwritten by checkpoint)
                import weakref
                new_layer._model_weakref = weakref.ref(self)
                module_id = id(new_layer)
                vt_id = id(new_layer.VT) if hasattr(new_layer, 'VT') else 'N/A'
                
                # Debug: Check if we've seen this module/VT ID before
                if module_id in created_modules:
                    logger.error(
                        f"[CRITICAL] Module ID {module_id} already exists! "
                        f"Previous layer: {created_modules[module_id]}, current: {full_name}"
                    )
                created_modules[module_id] = full_name
                
                # Module creation logging removed for cleaner output
                
                # Replace the original module with the new one
                setattr(info["father"], info["name"], new_layer)
        
        # POST INIT verification logging removed for cleaner output
    
    def validate_module_ids_after_loading(self):
        """Validate module IDs after checkpoint loading to detect if modules were replaced.
        This should be called right after from_pretrained completes.
        """
        logger.info("[POST LOAD] Verifying module IDs after checkpoint loading for layers 0-2:")
        for name, module in self.named_modules():
            if isinstance(module, HeadwiseLowRankModule):
                if name in self.head_wise_ranks:
                    match = re.search(r'layers\.(\d+)', name)
                    if match:
                        layer_num = int(match.group(1))
                        if layer_num in [0, 1, 2]:
                            module_id = id(module)
                            vt_id = id(module.VT) if hasattr(module, 'VT') else 'N/A'
                            expected_output = sum(self.head_wise_ranks[name])
                            actual_output = module.VT.weight.shape[0] if hasattr(module, 'VT') and hasattr(module.VT, 'weight') else 'N/A'
                            logger.info(
                                f"[POST LOAD] {name}: module_id={module_id}, VT_id={vt_id}, "
                                f"VT.shape[0]={actual_output}, expected={expected_output}"
                            )
    
    def _reinitialize_corrupted_modules(self):
        """Re-initialize modules that were corrupted by checkpoint loading.
        This method should be called after checkpoint loading to restore correct ranks and VT dimensions.
        """
        logger.info("[REINIT] Re-initializing modules corrupted by checkpoint loading...")
        reinitialized_count = 0
        
        # First, check if all modules share the same ID (clear sign of corruption)
        module_ids = {}
        vt_ids = {}
        for name, module in self.named_modules():
            if isinstance(module, HeadwiseLowRankModule):
                if name in self.head_wise_ranks:
                    module_id = id(module)
                    vt_id = id(module.VT) if hasattr(module, 'VT') else None
                    if module_id not in module_ids:
                        module_ids[module_id] = []
                    module_ids[module_id].append(name)
                    if vt_id:
                        if vt_id not in vt_ids:
                            vt_ids[vt_id] = []
                        vt_ids[vt_id].append(name)
        
        # Check for shared module IDs
        shared_module_ids = {mid: names for mid, names in module_ids.items() if len(names) > 1}
        shared_vt_ids = {vid: names for vid, names in vt_ids.items() if len(names) > 1}
        
        if shared_module_ids:
            logger.error(
                f"[REINIT] CRITICAL: Found {len(shared_module_ids)} shared module IDs! "
                f"This indicates all modules were replaced with the same instance during checkpoint loading. "
                f"Example: module_id {list(shared_module_ids.keys())[0]} is shared by {len(shared_module_ids[list(shared_module_ids.keys())[0]])} modules."
            )
            # Force re-initialization of all modules with shared IDs
            for shared_id, names in shared_module_ids.items():
                if len(names) > 1:
                    logger.error(
                        f"[REINIT] CRITICAL: Module ID {shared_id} is shared by {len(names)} modules: {names[:3]}..."
                    )
        if shared_vt_ids:
            logger.error(
                f"[REINIT] CRITICAL: Found {len(shared_vt_ids)} shared VT IDs! "
                f"This indicates all VT layers were replaced with the same instance. "
                f"Example: VT_id {list(shared_vt_ids.keys())[0]} is shared by {len(shared_vt_ids[list(shared_vt_ids.keys())[0]])} modules."
            )
            # Force re-initialization of all modules with shared VT IDs
            for shared_vt_id, names in shared_vt_ids.items():
                if len(names) > 1:
                    logger.error(
                        f"[REINIT] CRITICAL: VT ID {shared_vt_id} is shared by {len(names)} modules: {names[:3]}..."
                    )
        
        # Find all modules that need to be re-initialized
        modules_to_fix = []
        for name, module in self.named_modules():
            if isinstance(module, HeadwiseLowRankModule):
                if name in self.head_wise_ranks:
                    expected_ranks = self.head_wise_ranks[name]
                    expected_output = sum(expected_ranks)
                    
                    # Check if module has wrong ranks or VT dimensions
                    needs_fix = False
                    
                    # Check ranks - convert both to lists for comparison
                    module_ranks = list(module.ranks) if hasattr(module, 'ranks') and module.ranks is not None else None
                    expected_ranks_list = list(expected_ranks) if expected_ranks is not None else None
                    
                    if module_ranks is not None and expected_ranks_list is not None:
                        if module_ranks != expected_ranks_list:
                            needs_fix = True
                            logger.warning(
                                f"[REINIT] {name}: ranks mismatch - "
                                f"module.ranks={module_ranks} (sum={sum(module_ranks)}) != expected={expected_ranks_list} (sum={sum(expected_ranks_list)})"
                            )
                    else:
                        # Log why ranks check failed
                        logger.debug(
                            f"[REINIT] {name}: Could not compare ranks - "
                            f"module_ranks={module_ranks}, expected_ranks_list={expected_ranks_list}"
                        )
                    if module_ranks is None:
                        needs_fix = True
                        logger.warning(
                            f"[REINIT] {name}: module.ranks is None or missing"
                        )
                    
                    # Check VT dimensions
                    if hasattr(module, 'VT') and hasattr(module.VT, 'weight'):
                        actual_output = module.VT.weight.shape[0]
                        if actual_output != expected_output:
                            needs_fix = True
                            logger.warning(
                                f"[REINIT] {name}: VT dimension mismatch - "
                                f"VT.shape[0]={actual_output} != expected={expected_output} "
                                f"(module.ranks={module_ranks}, expected_ranks={expected_ranks_list})"
                            )
                    else:
                        needs_fix = True
                        logger.warning(
                            f"[REINIT] {name}: VT.weight is missing"
                        )
                    
                    # Also consider a module corrupted if its ID is shared
                    module_id = id(module)
                    vt_id = id(module.VT) if hasattr(module, 'VT') else None
                    if module_id in shared_module_ids and len(shared_module_ids[module_id]) > 1:
                        if not needs_fix: # Avoid duplicate logging if already marked for fix
                            logger.warning(f"[REINIT] {name}: Module ID is shared, marking for re-initialization.")
                        needs_fix = True
                    if vt_id and vt_id in shared_vt_ids and len(shared_vt_ids[vt_id]) > 1:
                        if not needs_fix: # Avoid duplicate logging
                            logger.warning(f"[REINIT] {name}: VT ID is shared, marking for re-initialization.")
                        needs_fix = True
                    
                    if needs_fix:
                        # Get parent module and attribute name
                        parts = name.split('.')
                        parent = self
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        attr_name = parts[-1]
                        
                        modules_to_fix.append({
                            'name': name,
                            'parent': parent,
                            'attr_name': attr_name,
                            'expected_ranks': expected_ranks,
                            'in_features': module.in_features,
                            'out_features': module.out_features,
                            'bias': module.bias is not None if hasattr(module, 'bias') else False,
                        })
        
        # Re-initialize each corrupted module
        for info in modules_to_fix:
            logger.info(
                f"[REINIT] Re-initializing {info['name']} with ranks={info['expected_ranks']} "
                f"(sum={sum(info['expected_ranks'])})"
            )
            
            # Create new module with correct ranks
            new_module = HeadwiseLowRankModule(
                info['expected_ranks'],
                info['in_features'],
                info['out_features'],
                bias=info['bias']
            )
            
            # Restore metadata
            new_module._expected_ranks = tuple(info['expected_ranks'])
            new_module._layer_name = info['name']
            new_module._init_module_id = id(new_module)
            import weakref
            new_module._model_weakref = weakref.ref(self)
            
            # Replace the corrupted module
            setattr(info['parent'], info['attr_name'], new_module)
            reinitialized_count += 1
            
            # Note: The new module needs to be re-patched by the quantization manager
            # This will be done automatically if _patch_projection_modules is called again
            # or we can add a callback mechanism here
        
        if reinitialized_count > 0:
            logger.warning(
                f"[REINIT] Re-initialized {reinitialized_count} modules that were corrupted by checkpoint loading. "
                f"These modules now have correct ranks but may need their weights loaded from the checkpoint."
            )
            # Call callback to re-patch modules if one is registered
            if self._reinit_callback is not None:
                logger.info("[REINIT] Calling re-patching callback to re-patch re-initialized modules...")
                try:
                    self._reinit_callback()
                    logger.info("[REINIT] Successfully re-patched re-initialized modules.")
                except Exception as e:
                    logger.error(f"[REINIT] Error during re-patching callback: {e}")
            else:
                logger.warning(
                    "[REINIT] No re-patching callback registered. "
                    "Re-initialized modules may not work correctly until re-patched. "
                    "Call model.register_reinit_callback(quantization_manager._patch_projection_modules) to enable auto re-patching."
                )
        else:
            logger.info("[REINIT] No modules needed re-initialization.")
        
        return reinitialized_count
    
    def register_reinit_callback(self, callback):
        """Register a callback to be called after module re-initialization.
        
        This is used to re-patch modules after they've been re-initialized,
        since the patched forward closures reference the old module objects.
        
        Args:
            callback: A callable that will re-patch modules (e.g., quantization_manager._patch_projection_modules)
        """
        self._reinit_callback = callback
        logger.info("[REINIT] Registered re-patching callback.")
    
    def _validate_and_fix_vt_weights(self):
        """Validate that all VT weights have the correct dimensions after checkpoint loading.
        This is called after state_dict loading to ensure the checkpoint didn't overwrite
        the correctly initialized VT layers with incorrect dimensions.
        """
        for name, module in self.named_modules():
            if isinstance(module, HeadwiseLowRankModule):
                if name in self.head_wise_ranks:
                    expected_output = sum(self.head_wise_ranks[name])
                    if hasattr(module, 'VT') and hasattr(module.VT, 'weight'):
                        actual_output = module.VT.weight.shape[0]
                        if actual_output != expected_output:
                            logger.error(
                                f"[CRITICAL VT WEIGHT CHECK] {name}: "
                                f"VT.weight.shape[0]={actual_output} != expected={expected_output} "
                                f"(ranks={self.head_wise_ranks[name]}, sum={expected_output}). "
                                f"This indicates the checkpoint overwrote the correctly initialized VT layer. "
                                f"Module will need to be reinitialized or checkpoint needs to be fixed."
                            )
    
    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to validate VT weights after loading."""
        # Log module IDs before loading (only for first few layers to reduce verbosity)
        logger.info("[CHECKPOINT LOAD] Before loading checkpoint, verifying module IDs for layers 0-2:")
        for name, module in self.named_modules():
            if isinstance(module, HeadwiseLowRankModule):
                if name in self.head_wise_ranks:
                    # Only log for layers 0, 1, 2
                    match = re.search(r'layers\.(\d+)', name)
                    if match:
                        layer_num = int(match.group(1))
                        if layer_num in [0, 1, 2]:
                            module_id = id(module)
                            vt_id = id(module.VT) if hasattr(module, 'VT') else 'N/A'
                            expected_output = sum(self.head_wise_ranks[name])
                            actual_output = module.VT.weight.shape[0] if hasattr(module, 'VT') and hasattr(module.VT, 'weight') else 'N/A'
                            logger.info(
                                f"[CHECKPOINT LOAD BEFORE] {name}: module_id={module_id}, VT_id={vt_id}, "
                                f"VT.shape[0]={actual_output}, expected={expected_output}"
                            )
        
        result = super().load_state_dict(state_dict, strict=strict)
        
        # Log module IDs after loading (only for first few layers to reduce verbosity)
        logger.info("[CHECKPOINT LOAD] After loading checkpoint, verifying module IDs for layers 0-2:")
        for name, module in self.named_modules():
            if isinstance(module, HeadwiseLowRankModule):
                if name in self.head_wise_ranks:
                    # Only log for layers 0, 1, 2
                    match = re.search(r'layers\.(\d+)', name)
                    if match:
                        layer_num = int(match.group(1))
                        if layer_num in [0, 1, 2]:
                            module_id = id(module)
                            vt_id = id(module.VT) if hasattr(module, 'VT') else 'N/A'
                            expected_output = sum(self.head_wise_ranks[name])
                            actual_output = module.VT.weight.shape[0] if hasattr(module, 'VT') and hasattr(module.VT, 'weight') else 'N/A'
                            status = "✓" if actual_output == expected_output else "✗ MISMATCH"
                            logger.info(
                                f"[CHECKPOINT LOAD AFTER] {name}: module_id={module_id}, VT_id={vt_id}, "
                                f"VT.shape[0]={actual_output}, expected={expected_output} {status}"
                            )
        
        # Validate VT weights after loading
        self._validate_and_fix_vt_weights()
        
        # Re-initialize any modules that were corrupted by checkpoint loading
        # This fixes the issue where the checkpoint replaces unique modules with a single shared module
        self._reinitialize_corrupted_modules()
        
        return result
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to skip loading VT weights with incorrect dimensions.
        This is called for each parameter during state_dict loading.
        """
        # Check if this is a VT weight parameter
        if prefix.endswith('.VT.weight'):
            # Extract the module name (e.g., "model.layers.0.self_attn.k_proj" from "model.layers.0.self_attn.k_proj.VT.weight")
            module_name = prefix[:-len('.VT.weight')]
            
            # Log for first few layers to verify this is being called
            match = re.search(r'layers\.(\d+)', module_name)
            if match:
                layer_num = int(match.group(1))
                if layer_num in [0, 1, 2]:
                    logger.info(f"[LOAD STATE DICT] Loading VT weight for {module_name}")
            
            if module_name in self.head_wise_ranks:
                expected_output = sum(self.head_wise_ranks[module_name])
                
                # Get the actual module to check its current VT shape
                module = None
                for name, m in self.named_modules():
                    if name == module_name and isinstance(m, HeadwiseLowRankModule):
                        module = m
                        break
                
                if module is not None and hasattr(module, 'VT') and hasattr(module.VT, 'weight'):
                    current_vt_shape = module.VT.weight.shape
                    expected_shape = (expected_output, current_vt_shape[1])
                    
                    # Check if the state_dict has this key and if it has wrong dimensions
                    full_key = prefix + 'weight'
                    if full_key in state_dict:
                        loaded_weight = state_dict[full_key]
                        if loaded_weight.shape[0] != expected_output:
                            logger.warning(
                                f"[SKIP VT LOAD] {module_name}: Skipping VT weight load - "
                                f"checkpoint has shape {loaded_weight.shape[0]} but expected {expected_output}. "
                                f"Keeping correctly initialized VT with shape {current_vt_shape[0]}."
                            )
                            # Remove this key from state_dict so it won't be loaded
                            del state_dict[full_key]
                            # Add to unexpected_keys to avoid strict mode errors
                            if full_key not in unexpected_keys:
                                unexpected_keys.append(full_key)
        
        # Call parent implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    @staticmethod
    def get_kv_info(llama: LlamaForCausalLM, num_heads_in_lr_groups: int):
        num_lr_groups = llama.config.num_attention_heads // num_heads_in_lr_groups
        num_lr_kv_groups = llama.config.num_key_value_heads // num_heads_in_lr_groups
        head_dim = llama.config.hidden_size // llama.config.num_attention_heads
        lr_group_dims = head_dim * num_heads_in_lr_groups
        
        if num_lr_groups * num_heads_in_lr_groups != llama.config.num_attention_heads:
            raise ValueError(
                f"num_heads must be divisible by num_heads_in_lr_groups (got `num_heads`: {llama.config.num_attention_heads}"
                f" and `num_heads_in_lr_groups`: {num_heads_in_lr_groups})."
            )
    
        if num_lr_kv_groups * num_heads_in_lr_groups != llama.config.num_key_value_heads:
            raise ValueError(
                f"num_key_value_heads must be divisible by num_heads_in_lr_groups (got `num_key_value_heads`: {llama.config.num_key_value_heads}"
                f" and `num_heads_in_lr_groups`: {num_heads_in_lr_groups})."
            )

        return SimpleNamespace(
            num_lr_groups=num_lr_kv_groups,
            lr_group_dims=lr_group_dims,
        )
