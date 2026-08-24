from loguru import logger
import torch.nn as nn
import torch
import os
import click
from tqdm import tqdm
from .data_utils import get_calib_data
from .model import HeadwiseLowRankModule

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def get_whiten_scale_matrix(model, tokenizer, args, dev):
    model_id = model.config._name_or_path
    #NOTE (brian1009): Might need to check the random seed, currently we have < 0.1 perplexity difference at Llama2-7B
    calib_loader = get_calib_data(
        "wikitext2", 
        tokenizer, 
        model_id, 
        nsamples=256, 
        seqlen=2048
    )
    cross_layer_pca = getattr(args, 'cross_layer_pca', False)
    suffix = "_cross_layer" if cross_layer_pca else ""
    cache_file = f"cache/whiten/{model_id.replace('/','_')}_w2_scaling_matrices_fp16{suffix}.pt"
    os.makedirs("cache/whiten", exist_ok=True)
    """
    cache format:
    [
        {
            "attn.q_proj": torch.Tensor,
            "attn.k_proj": torch.Tensor,
            "attn.v_proj": torch.Tensor,
            "attn.o_proj": torch.Tensor,
            "mlp.gate_proj": torch.Tensor,
            "mlp.up_proj": torch.Tensor,
            "mlp.down_proj": torch.Tensor
        },
        ... (stacked n times, in the order of model layers)
    ]
    """
    logger.info(f"[whiten] Calibration dataset: {args.calib_dataset}", fg="yellow")
    logger.info(f"[whiten] Search cache_file={cache_file}", fg="yellow")
    if os.path.exists(cache_file) and args.use_cache:
        logger.info(f"[whiten] File {cache_file} exist.", fg="green")
        logger.info(f"[whiten] Load scaling diag matrix from cache: {cache_file}", fg="yellow")
        scaling_matrics = torch.load(cache_file, map_location="cpu")


        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            subset = find_layers(layer) # Collect all linear layers
            for name in subset:
                if name in scaling_matrics[i]:
                    scaling_diag_matrix = scaling_matrics[i][name]
                    subset[name].scaling_diag_matrix = scaling_diag_matrix

        return 
    
    logger.info(f"No cache_file={cache_file}", fg="red")
    logger.info(f"Create whiten scale matrix dict...", fg="yellow")

    # Create Scaling Matrix with low-resource inference
    # Adapted from https://github.com/AIoT-MLSys-Lab/SVD-LLM/blob/main/SVDLLM.py
    # Here, inference are performed in an layer-wise manner.
    use_cache = model.config.use_cache
    model.config.use_cache = False
    #FIXME: This is not a good implementation...
    if "llama" in model_id or "mistral" in model_id or "vicuna" in model_id or "longchat":
        layers = model.model.layers
    elif "opt" in model_id:
        layers = model.model.decoder.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    scaling_matrices = []
    raw_matrices = []  # raw X^T X per layer, needed for cross-layer PCA
    logger.info("[Decomposition] Start to calculate the scaling matrix in layer-wise manner...")
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks, position_ids=position_ids[0].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        layer_scaling_matrices = {}
        layer_raw_matrices = {}
        for name in subset:
            if not ("k_proj" in name or "v_proj" in name):
                continue
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().cuda()
            if cross_layer_pca:
                layer_raw_matrices[name] = raw_scaling_diag_matrix.cpu()
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                subset[name].scaling_diag_matrix = scaling_diag_matrix
            except Exception as e:
                logger.warning("eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    logger.warning("raw scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    logger.warning("raw scaling_diag_matrix contains Inf!")
                if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                    logger.warning("raw scaling_diag_matrix is not a symmetric matrix!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-3) * torch.eye(raw_scaling_diag_matrix.shape[0]).cuda()
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                if torch.isnan(scaling_diag_matrix).any():
                    logger.warning("scaling_diag_matrix contains NaN!")
                elif torch.isinf(scaling_diag_matrix).any():
                    logger.warning("scaling_diag_matrix contains Inf!")
                del eigenvalues
                subset[name].scaling_diag_matrix = scaling_diag_matrix
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                logger.warning("scaling_diag_matrix is not full rank!")
                reg_inv =  1e-3 * torch.eye(scaling_diag_matrix.shape[0]).cuda() 
                scaling_diag_matrix += reg_inv
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                del reg_inv
            
            del scaling_matrix_inv
            layer_scaling_matrices[name] = scaling_diag_matrix.cpu()
            torch.cuda.empty_cache()
        scaling_matrices.append(layer_scaling_matrices)
        if cross_layer_pca:
            raw_matrices.append(layer_raw_matrices)
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()
        
    if cross_layer_pca:
        logger.info("[Decomposition] Computing joint whitening matrix (cross-layer PCA)...")
        raw_joint = None
        for layer_raw in raw_matrices:
            for name, raw in layer_raw.items():
                raw_gpu = raw.double().cuda()
                raw_joint = raw_gpu if raw_joint is None else raw_joint + raw_gpu
        try:
            joint_chol = torch.linalg.cholesky(raw_joint).float()
        except Exception:
            logger.warning("[Decomposition] Joint Cholesky failed, regularizing...")
            eigenvalues = torch.linalg.eigvalsh(raw_joint)
            raw_joint += (-eigenvalues[0] + 1e-3) * torch.eye(raw_joint.shape[0], device=raw_joint.device)
            joint_chol = torch.linalg.cholesky(raw_joint).float()
        joint_chol_cpu = joint_chol.cpu()
        for layer_mats in scaling_matrices:
            for name in layer_mats:
                layer_mats[name] = joint_chol_cpu
        logger.info("[Decomposition] Cross-layer PCA done — all layers share the joint whitening matrix.")

    model.config.use_cache = use_cache
    if args.use_cache:
        torch.save(scaling_matrices, cache_file)
        logger.info(f"Save the whiten scale matrix dict to:  {cache_file}")

def compress_model_whiten(model, tokenizer, args, dev, selection_result):
    logger.info("Compressing model with whiten decomposition...")
    # NOTE(brian1009): Prepare whiten scaling matrix
    get_whiten_scale_matrix(model, tokenizer, args, dev)
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
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

    logger.info(f"Start decompose the layer with selected ranks... #target layers: {len(selection_result.keys())}")
    for layername, selected_head_rank in tqdm(selection_result.items()):
        logger.debug(f"Decompose {layername} with ranks: {selected_head_rank}")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
    
        head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten(
            raw_linear,
            selected_head_rank,
            uneven_split=getattr(args, 'uneven_split', False)
        )
        setattr(info["father"], info["name"],  head_wise_svd_linear)

def compress_model_svd(model, selection_result):
    logger.info("Compressing model with svd decomposition...")
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
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

    logger.info(f"Start decompose the layer with selected ranks... #target layers: {len(selection_result.keys())}")
    for layername, selected_head_rank in tqdm(selection_result.items()):
        logger.debug(f"Decompose {layername} with ranks: {selected_head_rank}")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        print("head-wise svd", layername, raw_linear)
        head_wise_svd_linear = HeadwiseLowRankModule.from_linear(
            raw_linear,
            selected_head_rank
        )
        setattr(info["father"], info["name"],  head_wise_svd_linear)

@torch.no_grad()
def get_joint_kv_pca_matrices(model, tokenizer, args, dev, selection_result):
    """
    Compute joint cross-layer KV PCA bases.

    For each group g, we accumulate the shared D×D covariance:
        M_g = sum_{l=0}^{L-1} K_l^{g,T} K_l^g  ∈ R^{group_dim × group_dim}
    where K_l^g ∈ R^{n_tokens × group_dim} is the k_proj output for layer l, group g.

    The top-r eigenvectors of M_g form an orthonormal basis V_g ∈ R^{group_dim × group_dim}
    (descending eigenvalue order). The same V_g is used for ALL layers — no per-layer
    slicing needed. This ensures V_g[:, :r] @ V_g[:, :r]^T = I_{D×D} at full rank,
    making reconstruction lossless at ratio=1.0.

    Returns:
        {
          "k": {g: V_g ∈ R^{group_dim × group_dim}},  # shared eigvecs, descending order
          "num_layers": int,
          "group_dim_k": int,
          "num_groups_k": int,
        }
    """
    model_id = model.config._name_or_path
    n_samples = getattr(args, 'n_joint_pca_samples',
                        min(getattr(args, 'n_whiten_calib_samples', 16), 16))

    cache_file = f"cache/joint_kv_pca/{model_id.replace('/','_')}_joint_kv_pca.pt"
    os.makedirs("cache/joint_kv_pca", exist_ok=True)
    if os.path.exists(cache_file) and getattr(args, 'use_cache', False):
        logger.info(f"[JointKVPCA] Loading cached PCA matrices from {cache_file}")
        return torch.load(cache_file, map_location="cpu")

    calib_loader = get_calib_data(
        "wikitext2", tokenizer, model_id, nsamples=n_samples, seqlen=2048
    )

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "llama" in model_id.lower() or "mistral" in model_id.lower() or "vicuna" in model_id.lower() or "longchat" in model_id.lower():
        layers = model.model.layers
    elif "opt" in model_id.lower():
        layers = model.model.decoder.layers
    else:
        raise ValueError(f"[JointKVPCA] Unsupported model architecture: {model_id}")

    num_layers = len(layers)

    # Determine group structure from selection_result (same as what rank_search produced)
    first_k, first_v = None, None
    for layername, ranks in selection_result.items():
        if 'k_proj' in layername and first_k is None:
            first_k = (layername, ranks)
        if 'v_proj' in layername and first_v is None:
            first_v = (layername, ranks)
        if first_k and first_v:
            break

    if first_k is None:
        raise ValueError("[JointKVPCA] selection_result must contain k_proj entries")

    ref_k = dict(model.named_modules())[first_k[0]]
    num_groups_k, group_dim_k = len(first_k[1]), ref_k.out_features // len(first_k[1])

    logger.info(
        f"[JointKVPCA] K only — groups_k={num_groups_k} group_dim_k={group_dim_k}  "
        f"Shared D×D covariance per group: {group_dim_k}×{group_dim_k} ({group_dim_k*group_dim_k*8/1e6:.2f} MB fp64)"
    )

    # D×D shared covariance: sum of K_l^T K_l across all layers
    M_k = {g: torch.zeros(group_dim_k, group_dim_k, dtype=torch.float64) for g in range(num_groups_k)}

    # Collect initial hidden states (same Catcher pattern as get_whiten_scale_matrix)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((n_samples, 2048, model.config.hidden_size), dtype=dtype, device=dev)
    cache_dict = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            if cache_dict['i'] < n_samples:
                inps[cache_dict['i']] = inp
            cache_dict['i'] += 1
            if cache_dict['attention_mask'] is None:
                cache_dict['attention_mask'] = kwargs.get('attention_mask')
                cache_dict['position_ids'] = kwargs.get('position_ids')
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    attention_masks = cache_dict['attention_mask']
    position_ids = cache_dict['position_ids']
    n_actual = min(n_samples, cache_dict['i'])

    logger.info(f"[JointKVPCA] Processing {n_actual} calibration samples (layer-by-layer per sample)...")

    # For each sample: run all layers sequentially, collect K outputs per group,
    # then accumulate shared D×D covariance sum_l K_l^T K_l on CPU.
    for s in tqdm(range(n_actual), desc="[JointKVPCA] samples"):
        inp_s = inps[s:s+1]  # (1, seq, hidden)
        am_s = attention_masks[:1] if attention_masks is not None else None
        pid_s = position_ids[0:1] if position_ids is not None else None

        K_layers = []  # K_layers[l][g] = cpu tensor (seq, group_dim_k)

        for l in range(num_layers):
            layer = layers[l].to(dev)

            k_cap = {}

            def make_k_hook(cap):
                def hook(module, inp_h, out_h):
                    o = out_h.detach().float().squeeze(0)  # (seq, out_features_k)
                    for g in range(num_groups_k):
                        cap[g] = o[:, g * group_dim_k:(g + 1) * group_dim_k].cpu()
                return hook

            handles = []
            for name, module in layer.named_modules():
                if 'k_proj' in name and isinstance(module, nn.Linear):
                    handles.append(module.register_forward_hook(make_k_hook(k_cap)))

            out_s = layer(inp_s, attention_mask=am_s, position_ids=pid_s)[0]

            for h in handles:
                h.remove()

            K_layers.append(dict(k_cap))

            layer = layer.cpu()
            inp_s = out_s
            torch.cuda.empty_cache()

        # Accumulate shared D×D covariance for K: sum_l K_l^T K_l
        for g in range(num_groups_k):
            for l in range(num_layers):
                K_lg = K_layers[l][g].double()  # (seq, group_dim_k)
                M_k[g].add_(K_lg.T @ K_lg)

    model.config.use_cache = use_cache

    # Eigen-decompose K covariance; keep all eigenvectors (descending order)
    result = {
        "k": {},
        "num_layers": num_layers,
        "group_dim_k": group_dim_k,
        "num_groups_k": num_groups_k,
    }

    logger.info("[JointKVPCA] Computing K eigenvectors (may take a moment for large matrices)...")

    for g in tqdm(range(num_groups_k), desc="[JointKVPCA] K eigh"):
        M = M_k[g]
        M = (M + M.T) / 2.0  # symmetrize for numerical stability
        _, eigvecs = torch.linalg.eigh(M)
        result["k"][g] = eigvecs.flip(-1).float()  # (group_dim_k, group_dim_k), descending order

    if getattr(args, 'use_cache', False):
        torch.save(result, cache_file)
        logger.info(f"[JointKVPCA] Saved PCA matrices to {cache_file}")

    return result


def compress_model_joint_kv_pca(model, tokenizer, args, dev, selection_result):
    """
    Compress K projections with joint cross-layer PCA; V projections with per-layer whitening.

    Rationale: K determines which tokens get attended to — attention patterns are correlated
    across layers (shared sinks, recency) so a joint basis is beneficial.  V carries
    layer-specific content; a shared basis causes early-layer starvation because later layers
    dominate the joint covariance, leaving early-layer V slices near-zero.
    """
    logger.info("Compressing model: joint cross-layer PCA for K, per-layer whitening for V...")

    # --- Joint PCA for K ---
    joint_pca = get_joint_kv_pca_matrices(model, tokenizer, args, dev, selection_result)

    # --- Per-layer whitening for V ---
    k_only_result = {name: ranks for name, ranks in selection_result.items() if "k_proj" in name}
    v_only_result = {name: ranks for name, ranks in selection_result.items() if "v_proj" in name}

    # Whitening calibration sets scaling_diag_matrix on each v_proj nn.Linear
    get_whiten_scale_matrix(model, tokenizer, args, dev)

    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while modules:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {"father": submodule, "name": name, "full_name": full_name}
            else:
                modules.append(raw_linear)

    # Apply joint PCA to k_proj
    logger.info(f"[JointKVPCA] Applying joint PCA to {len(k_only_result)} k_proj layers...")
    for layername, selected_head_rank in tqdm(k_only_result.items()):
        num_groups = len(selected_head_rank)
        # V_g is (group_dim_k, group_dim_k) — shared across all layers
        joint_V_per_group = [joint_pca["k"][g] for g in range(num_groups)]

        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        new_module = HeadwiseLowRankModule.from_linear_joint_pca(
            raw_linear, selected_head_rank, joint_V_per_group
        )
        setattr(info["father"], info["name"], new_module)

    # Apply per-layer whitening to v_proj
    logger.info(f"[JointKVPCA] Applying per-layer whitening to {len(v_only_result)} v_proj layers...")
    for layername, selected_head_rank in tqdm(v_only_result.items()):
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        new_module = HeadwiseLowRankModule.from_linear_whiten(
            raw_linear,
            selected_head_rank,
            uneven_split=getattr(args, 'uneven_split', False),
        )
        setattr(info["father"], info["name"], new_module)


# Wrapper for different decompose methods
def compress_model(model, tokenizer, args, dev, selection_result):
    if args.decompose_method == "whiten":
        compress_model_whiten(model, tokenizer, args, dev, selection_result)
    elif args.decompose_method == "svd":
        compress_model_svd(model, selection_result)
    elif args.decompose_method == "joint_kv_pca":
        compress_model_joint_kv_pca(model, tokenizer, args, dev, selection_result)
    else:
        raise ValueError(f"Decomposition method {args.decompose_method} is not supported.")