Where# KV Cache Compression Through the Lens of Transform Coding

Reference implementation of **AATC (Attention-Aware Transform Coding)**.

> **KV Cache Compression Through the Lens of Transform Coding**
> Hannah Laus, Claudio Mayrink Verdun, Hao Wang, Flavio du Pin Calmon, Felix Krahmer
> arXiv:2608.14191 — the PDF is included in this repository as `2608.14191v1.pdf`.

The KV cache is the dominant memory cost of long-context LLM inference. Existing
quantization methods lower precision *uniformly* and minimize reconstruction
error on the cache itself, without accounting for how that error propagates
through attention. We prove that under a white-noise quantization model the
expected attention-aware distortion decomposes into additive key and value
contributions that factor across tokens and channels (Theorem 1), and use the
classical transform-coding recipe — decorrelate, then allocate by reverse
water-filling — to spend a bit budget against *that* objective.

On `Llama-3.1-8B-Instruct` and `Qwen-2.5-7B-Instruct`, evaluated across
LongBench, RULER, GSM8K, MMLU-Pro and MATH-500, AATC is near-lossless at
approximately 5.8x compression, whereas every baseline degrades on at least one
benchmark.

---

## AATC vs. the PALU baseline — the distinction that matters

Both AATC and the PALU baseline start from the same `compress.py` script, and
the *only* thing separating them is the rank ratio. Getting this right is the
difference between running our method and running a baseline:

| | rank ratio | what the checkpoint does | how bits are spent |
|---|---|---|---|
| **AATC** (ours) | `--param_ratio_target 1.0` | Whitening + SVD with **no truncation** — the transform is exactly invertible and contributes **zero** error | **Non-uniform** per-channel allocation from `run_bit_allocation.py` |
| **PALU** (baseline) | `--param_ratio_target 0.7` | Low-energy subspaces are **deleted** — the checkpoint is lossy before a single bit is spent | **Flat** budget over the surviving latent dimensions |

In other words, PALU compresses by throwing away rank and then quantizes what is
left uniformly; AATC keeps the full rank and instead distributes a fixed budget
*unevenly* across channels, guided by the attention-aware distortion. All of
AATC's loss is confined to the allocation step.

The evaluation scripts expose these as separate methods:

| Method key | Model | Description |
|---|---|---|
| `baseline` | base | FP16, no compression |
| `kivi` | base | KIVI uniform quantization (keys per-channel, values per-token) |
| `kvquant` | base | KVQuant non-uniform quantization with dense-and-sparse outliers |
| `palu` | ratio 0.7 | PALU decomposition, full-precision latents (no quantization) |
| `palu_uniform` | ratio 0.7 | PALU baseline: uniform latent quantization with Hadamard rotation |
| `aatc` | **ratio 1.0** | **This paper.** Per-channel allocated bitwidths; requires `--bitwidth_file` |

---

## Installation

```bash
pip install -r requirements.txt
```

**Fast Hadamard transform** — needed for the Hadamard rotation used by
`palu_uniform`:

```bash
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform && pip install -v .
```

**lm-evaluation-harness** — the backend for `run_gsm8k_mmlu.py`:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e .
```

**RULER** — needed only for the RULER benchmark. The scripts drive
[NVIDIA/RULER](https://github.com/NVIDIA/RULER) directly rather than vendoring it:

```bash
git clone https://github.com/NVIDIA/RULER
export RULER_REPO_PATH=/path/to/RULER          # or pass --ruler_repo_path

# one-time data download
cd RULER/scripts/data/synthetic/json
python download_paulgraham_essay.py            # haystack for the niah tasks
bash download_qa_dataset.sh                    # qa_1 / qa_2
```

**LongBench and the reasoning benchmarks need no manual setup** — they are
pulled from the Hugging Face hub at run time.

---

## The AATC pipeline

Four stages. Stages 1–3 are one-time offline costs; stage 4 is evaluation.

### 1. Build the transform — `compress.py`

Whitening + SVD of the K/V projections. **Use rank ratio 1.0 for AATC.**

```bash
python compress.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --calib_dataset wikitext2 \
    --param_ratio_target 1.0 \
    --search_method fisher_uniform \
    --decompose_method whiten \
    --head_group_size 4 \
    --dump_huggingface_model --use_cache
```

The paper derives the transform from wikitext2. Output goes to
`{MODEL_NAME}_ratio-{RATIO}_gs-{GROUP}-{SEARCH}-{DECOMPOSE}/`.

| Option | Default | Description |
|---|---|---|
| `--model_id` | `meta-llama/Llama-2-7b-hf` | Pretrained model path or HF ID |
| `--param_ratio_target` | `-1` | **1.0 for AATC**, 0.7 for the PALU baseline |
| `--search_method` | `fisher_uniform` | Rank search: `fisher`, `fisher_uniform`, `uniform` |
| `--decompose_method` | `whiten` | `whiten` (whitening-based SVD, used in the paper) or `svd` |
| `--head_group_size` | `4` | Group size for group-wise decomposition (G-LRD) |
| `--calib_dataset` | `wikitext2` | `wikitext2`, `c4`, or `ptb` |
| `--calib_seqlen` | `1024` | Calibration sequence length |
| `--n_fisher_calib_samples` | `32` | Samples for the Fisher-information search |
| `--n_whiten_calib_samples` | `256` | Samples for whitening |
| `--dump_huggingface_model` | off | Save the checkpoint (you want this) |
| `--use_cache` | off | Reuse cached calibration results |
| `--uneven_split` | off | `L = U@S, R = V` instead of splitting `sqrt(S)` |

### 2. Rate-distortion calibration — `run_rd_calibration.py`

Collects the per-channel variances `sigma2` and the attention-aware importance
weights `w` (the `q_c^2` term for keys, `W_Oc` for values) that Theorem 1 needs.

```bash
python run_rd_calibration.py \
    --model_path /Path/To/Ratio-1.0/Model \
    --dataset fineweb_openr1math \
    --nsamples 256 --seqlen 2048 \
    --output_path rd_stats.pt
```

| Option | Default | Description |
|---|---|---|
| `--model_path` | **required** | The rank-ratio-1.0 checkpoint from stage 1 |
| `--dataset` | `wikitext2` | **Paper uses `fineweb_openr1math`** (50/50 FineWeb + OpenR1-Math-220k) |
| `--nsamples` | `256` | Calibration sequences |
| `--seqlen` | `2048` | Sequence length |
| `--max_query_capture_batches` | `64` | Batches kept for key calibration — lower this if you OOM |
| `--gradient_checkpointing` | off | Trade compute for GPU memory |
| `--low_memory` / `--high_accuracy` | off | Memory / accuracy presets |
| `--output_path` | auto | Where to write the `.pt` statistics |
| `--use_cache` | off | Skip recalibration if a cached file exists |
| `--flash2` | off | Flash Attention 2 |

If the job is killed for memory, retry with
`--seqlen 1024 --nsamples 128 --max_query_capture_batches 32 --gradient_checkpointing`.

### 3. Reverse water-filling — `run_bit_allocation.py`

The step that makes this AATC rather than PALU: a fixed budget distributed
*non-uniformly* over the full-rank channels (Eq. 2).

```bash
python run_bit_allocation.py \
    --stats_path rd_stats.pt \
    --compression_ratio 0.125 \
    --min_bits 0 --max_bits 16 \
    --sigma2_normalization per_layer \
    --w_normalization per_layer \
    --protect_first_n_layers 3 \
    --protect_bits_per_dim_v 2.0 --protect_bits_per_dim_k 3.0 \
    --output_path bit_allocations.pt
```

The flags above reproduce the paper's configuration: bits in `[0, 16]`,
allocated **globally** across layers (not `--per_layer`) and separately for keys
and values; `sigma2` and `w` normalized by each layer's own 95th percentile to
remove the activation-scale bias that otherwise starves early layers; and the
first three layers protected, receiving the average budget for values and one
bit more than average for keys. Set the protect values to your target average.

| Option | Default | Description |
|---|---|---|
| `--stats_path` | **required** | RD statistics from stage 2 |
| `--compression_ratio` | `0.5` | Fraction of the original bit budget to keep |
| `--compression_ratio_k` / `_v` | None | Separate ratios for keys / values |
| `--total_bits` / `--total_bits_k` / `--total_bits_v` | None | Absolute budgets instead of a ratio |
| `--min_bits` / `--max_bits` | `0` / `8` | Per-channel bitwidth range (**paper: 0 / 16**) |
| `--sigma2_normalization` | `global` | `per_layer` normalizes by each layer's 95th percentile (**paper**) |
| `--w_normalization` | `global` | Same, for the key importance weights (**paper**) |
| `--w_o_normalization` | `global` | Same, for the value output-projection weights |
| `--protect_first_n_layers` | `0` | Exempt the first N layers from global water-filling (**paper: 3**) |
| `--protect_bits_per_dim_k` / `_v` | None | Fixed bits/dim for the protected layers |
| `--protect_budget_split` | None | `importance` / `fisher` / `log_fisher` make the protect value a floor |
| `--per_layer` | off | Allocate within each layer instead of globally |
| `--use_variance_only` | off | **AATC var-only** ablation: drop `q_c^2` and `W_Oc`, keep only variance |
| `--use_variance_only_k` / `_v` | None | Per-projection override of the above |
| `--min_avg_bits_per_layer` | `0.0` | Floor on average bits/dim per layer |
| `--bits_per_dimension` | `16` | Original precision |
| `--show_suggestions` | off | Print budget suggestions from current usage |
| `--output_path` | auto | Where to write the bitwidth `.pt` |

`--use_variance_only` produces the **AATC var-only** row from Table II — the
ablation that isolates how much the attention-awareness itself is worth.

### 4. Optional — channel scales and the KVQuant baseline

```bash
# Per-channel scales for the factored / channelwise AATC quantizer
python run_channelwise_scaling_calibration.py \
    --model_path /Path/To/Ratio-1.0/Model \
    --calib_dataset fineweb_openr1math \
    --nsamples 256 --seqlen 2048 --percentile 99.0 \
    --output_path fc_scale.pt

# Calibrated quantizer for the KVQuant baseline (nuq LUT + Fisher k-means)
python run_kvquant_calibration.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --nbits 4 --nsamples 8 --seqlen 512 \
    --sparsity_threshold 0.99 \
    --output kvquant_q.pt
```

---

## Evaluation

All three benchmark scripts share the same method keys and flag names, so a
configuration transfers between them unchanged. `--transform_model` points at
the checkpoint from stage 1 — the **ratio-1.0** one for `aatc`, the **ratio-0.7**
one for `palu` / `palu_uniform`. (`--palu_model` still works as an alias.)

The paper's inference-time defaults for AATC: keep the first 4 tokens in full
precision, a 128-token full-precision recent window with a sliding step of 16,
and `G = 2` head groups for Llama (`G = 1` for Qwen).

### LongBench — `run_longbench_comparison.py`

Seven English subsets (`triviaqa`, `qasper`, `trec`, `samsum`, `lcc`,
`repobench-p`, `qmsum`), scored with F1, ROUGE-L, classification accuracy and
edit similarity. Datasets download automatically from the Hugging Face hub.

```bash
CUDA_VISIBLE_DEVICES=0 python run_longbench_comparison.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --transform_model /Path/To/Ratio-1.0/Model \
    --methods baseline kivi kvquant aatc \
    --bitwidth_file bit_allocations.pt \
    --channelwise_scaling_file fc_scale.pt \
    --attention_sink_tokens 4 --recent_tokens 128 --sliding_step_size 16 \
    --output_dir results/longbench
```

| Option | Default | Description |
|---|---|---|
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Base model |
| `--transform_model` | None | Stage-1 checkpoint (required for `palu*` / `aatc`) |
| `--methods` | `baseline` | Any of the six method keys |
| `--v1_tasks` | all 7 | Subset of tasks to run |
| `--bitwidth_file` | None | **Required for `aatc`** — output of stage 3 |
| `--scaling_type` | `tokenwise` | `tokenwise`, `channelwise`, `channel_group`, `factored`, `kvtc` |
| `--channelwise_scaling_file` | None | Scales from `run_channelwise_scaling_calibration.py` |
| `--channel_group_size` | `64` | Channel group size for grouped scaling |
| `--attention_sink_tokens` | `0` | Leading tokens kept in FP16 (**paper: 4**) |
| `--recent_tokens` | `0` | Recent-window tokens kept in FP16 (**paper: 128**) |
| `--sliding_step_size` | `recent_tokens // 2` | Tokens requantized per step (**paper: 16**) |
| `--lt_bits` | `2` | Latent bitwidth for `palu_uniform` |
| `--lt_hadamard` / `--no_lt_hadamard` | on | Hadamard rotation for `palu_uniform` |
| `--kivi_bits` / `--kivi_group_size` / `--kivi_residual_length` | `2` / `32` / `128` | KIVI settings |
| `--kvquant_bits` | `4` | KVQuant bitwidth (2–5) |
| `--kvquant_quantizer_file` | None | Calibrated quantizer from `run_kvquant_calibration.py` |
| `--max_length` | auto | Max prompt tokens; inferred from the model config minus the task's generation budget |
| `--limit` | None | First N samples per task (smoke test) |
| `--output_dir` | `results/longbench` | Predictions and summary |

### Reasoning — `run_gsm8k_mmlu.py`

GSM8K (8-shot CoT), MMLU-Pro (5-shot), and MATH-500 (0-shot CoT, boxed exact
match), through lm-evaluation-harness.

```bash
CUDA_VISIBLE_DEVICES=0 python run_gsm8k_mmlu.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --transform_model /Path/To/Ratio-1.0/Model \
    --methods baseline kivi kvquant aatc \
    --benchmarks gsm8k mmlu_pro math_500 \
    --bitwidth_file bit_allocations.pt \
    --output_dir results/reasoning
```

Beyond the shared flags: `--benchmarks` (`gsm8k`, `mmlu_pro`, `math_500`,
`mmlu_pro_gen`), `--mmlu_pro_subjects`, `--math500_max_new_tokens` (2048),
`--batch_size` (8), `--max_length` (4096). `mmlu_pro_gen` runs MMLU-Pro through
`model.generate` so that AATC's recent-token window behaves exactly as it does
on GSM8K.

### RULER — `run_ruler_method_comparison.py`

Long-context retrieval and tracing across 4k–32k. This is where the gap between
a uniform budget and an allocated one is widest (Fig. 1).

```bash
CUDA_VISIBLE_DEVICES=0 python run_ruler_method_comparison.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --transform_model /Path/To/Ratio-1.0/Model \
    --methods baseline kivi kvquant palu_uniform aatc \
    --bitwidth_file bit_allocations.pt \
    --context_lengths 4096 8192 16384 32768 \
    --ruler_repo_path /path/to/RULER \
    --num_samples 500 --max_new_tokens 50 \
    --output_dir results/ruler
```

Beyond the shared flags: `--context_lengths`, `--ruler_repo_path`, `--tasks`,
`--task_category` (`retrieval`, `lookup`, `tracing`, `aggregation`, `qa`),
`--num_samples` (500), `--limit`, `--max_new_tokens` (50), `--tokenizer_path`.

`run_ruler.py` is the underlying single-model RULER driver; the comparison
script above calls into it and is what you want for reproducing the paper.

### Generation demo — `run_generation_with_quantization.py`

Single-prompt sanity check that the allocated bitwidths load and generate.

```bash
python run_generation_with_quantization.py \
    --model_path /Path/To/Ratio-1.0/Model \
    --bitwidth_file bit_allocations.pt \
    --prompt "The future of AI is" \
    --max_new_tokens 200 --temperature 0.7 --top_p 0.9
```

### Memory accounting — `kv_effective_bits.py`

Nominal bitwidth understates real memory: every method stores fp16 metadata and
keeps some values in full precision. This reports *effective* bits per element
so methods can be compared at matched memory rather than matched nominal bits.

```bash
python kv_effective_bits.py --method aatc --bitwidth_file bit_allocations.pt \
    --seqlen 8192 --num_groups 2 --residual_length 128 --num_sink_tokens 4
python kv_effective_bits.py --method kivi    --nbits 2 --group_size 32 --seqlen 8192
python kv_effective_bits.py --method kvquant --nbits 2 --seqlen 8192
```

---

## Repository layout

```
compress.py                             stage 1 — whitening + SVD transform
run_rd_calibration.py                   stage 2 — RD statistics (driver)
rd_calibration.py                         └─ calibration internals
run_bit_allocation.py                   stage 3 — reverse water-filling (driver)
bit_allocation.py                         └─ allocation algorithms
run_channelwise_scaling_calibration.py  per-channel quantizer scales
run_kvquant_calibration.py              KVQuant baseline quantizer

run_longbench_comparison.py             LongBench v1 evaluation
run_gsm8k_mmlu.py                       GSM8K / MMLU-Pro / MATH-500
run_ruler_method_comparison.py          RULER across context lengths
run_ruler.py                              └─ single-model RULER driver
run_generation_with_quantization.py     single-prompt generation demo

kivi_cache.py                           KIVI baseline cache
kvquant_cache.py                        KVQuant baseline cache
kv_effective_bits.py                    effective bits/element accounting
utils.py                                model loading and checkpoint dumping

palu/                                   decomposed model implementation
                                        (Llama and Qwen variants, quantized
                                        cache modules, low-rank linear layers)
longbench_utils/                        LongBench prompt formats and metrics,
                                        vendored from THUDM/LongBench (MIT)
```

---

## License

MIT — see [LICENSE](LICENSE). The `palu/` directory derives from
[PALU](https://github.com/shadowpa0327/Palu) and `longbench_utils/` from
[THUDM/LongBench](https://github.com/THUDM/LongBench), both MIT licensed.

---

## Citation

```bibtex
@misc{laus2026kvcache,
      title={KV Cache Compression Through the Lens of Transform Coding},
      author={Hannah Laus and Claudio Mayrink Verdun and Hao Wang and Flavio du Pin Calmon and Felix Krahmer},
      year={2026},
      eprint={2608.14191},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.14191},
}
```

The decomposed-model implementation in `palu/` builds on PALU:

```bibtex
@misc{chang2024palucompressingkvcachelowrank,
      title={Palu: Compressing KV-Cache with Low-Rank Projection},
      author={Chi-Chih Chang and Wei-Cheng Lin and Chien-Yu Lin and Chong-Yan Chen and Yu-Fang Hu and Pei-Shuo Wang and Ning-Chi Huang and Luis Ceze and Kai-Chiang Wu},
      year={2024},
      eprint={2407.21118},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.21118},
}
```
