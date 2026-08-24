import os
import numpy as np
import torch
from datasets import load_dataset
import random
from loguru import logger


def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    cache_file = (
        f"cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    )
    random.seed(seed)
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        logger.info(f"[Calib data] Load from {cache_file}", fg="yellow")
        return traindataset

    # Check if it's a Longbench dataset
    longbench_datasets = ["lcc", "triviaqa", "qasper", "trec", "samsum", "repobench-p", "qmsum", "multi_news", "lsht"]
    if name in longbench_datasets:
        # Load Longbench dataset
        try:
            from longbench_utils import DATASET2PROMPT
            longbench_data = load_dataset('THUDM/LongBench', name, split='test')
            prompt_format = DATASET2PROMPT.get(name, "{context}")

            traindataset = []
            # Use first nsamples from the dataset
            for i, json_obj in enumerate(longbench_data):
                if i >= nsamples:
                    break

                # Extract context from the data
                context = json_obj.get("context", json_obj.get("input", ""))

                # Format prompt if needed
                # Create a copy of json_obj without 'context' to avoid duplicate key error
                format_kwargs = {k: v for k, v in json_obj.items() if k != "context"}
                format_kwargs["context"] = context  # Use the extracted context

                try:
                    prompt = prompt_format.format(**format_kwargs)
                except KeyError as e:
                    # If format fails, just use context directly
                    logger.warning(f"Failed to format prompt for {name}, using context directly: {e}")
                    prompt = context

                # Tokenize
                trainenc = tokenizer(prompt, truncation=True, max_length=seqlen, return_tensors="pt")
                inp = trainenc.input_ids[:, :seqlen]
                attention_mask = trainenc.attention_mask[:, :seqlen] if "attention_mask" in trainenc else torch.ones_like(inp)

                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

            torch.save(traindataset, cache_file)
            logger.info(f"[Calib data] Created {len(traindataset)} samples from Longbench dataset {name}")
            return traindataset
        except Exception as e:
            logger.error(f"Failed to load Longbench dataset {name}: {e}")
            raise

    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
        traindataset = _sample_text(tot_text, tokenizer, nsamples, seqlen)

    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
        traindataset = _sample_text(tot_text, tokenizer, nsamples, seqlen)

    elif name == "fineweb":
        traindata = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        traindataset = _sample_streaming(traindata, "text", tokenizer, nsamples, seqlen, seed)

    elif name == "openr1math":
        traindata = load_dataset(
            "open-r1/OpenR1-Math-220k",
            split="train",
            streaming=True,
        )
        traindataset = _sample_streaming_openr1math(traindata, tokenizer, nsamples, seqlen, seed)

    elif name == "fineweb_openr1math":
        # 50/50 mixture matching KVtc calibration setup
        n_each = nsamples // 2
        logger.info(f"[Calib data] Loading {n_each} FineWeb + {n_each} OpenR1Math samples (seqlen={seqlen})")
        fw_data = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        math_data = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
        fw_samples = _sample_streaming(fw_data, "text", tokenizer, n_each, seqlen, seed)
        math_samples = _sample_streaming_openr1math(math_data, tokenizer, nsamples - n_each, seqlen, seed + 1)
        traindataset = fw_samples + math_samples
        random.shuffle(traindataset)

    else:
        raise NotImplementedError(
            f"Dataset {name} not supported. Supported: wikitext2, c4, fineweb, openr1math, "
            f"fineweb_openr1math, and LongBench datasets: {', '.join(longbench_datasets)}"
        )

    print(f"[Calib data] {name}: {len(traindataset)} samples at seqlen={seqlen}")
    torch.save(traindataset, cache_file)
    return traindataset


def _sample_text(tot_text, tokenizer, nsamples, seqlen):
    """Sample random fixed-length chunks from a concatenated text string."""
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset


def _sample_streaming(dataset, text_field, tokenizer, nsamples, seqlen, seed):
    """Sample from a streaming HuggingFace dataset by concatenating and chunking text."""
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    traindataset = []
    buffer = ""
    for item in dataset:
        text = item.get(text_field, "")
        if not text:
            continue
        buffer += text + "\n\n"
        # Once we have enough text, tokenize and extract chunks
        if len(buffer) > seqlen * nsamples * 6:
            break
    enc = tokenizer(buffer, return_tensors="pt").input_ids[0]
    for i in range(nsamples):
        start = random.randint(0, max(0, enc.shape[0] - seqlen - 1))
        chunk = enc[start: start + seqlen]
        if chunk.shape[0] < seqlen:
            chunk = torch.nn.functional.pad(chunk, (0, seqlen - chunk.shape[0]))
        traindataset.append({"input_ids": chunk.unsqueeze(0), "attention_mask": torch.ones(1, seqlen)})
    return traindataset


def _sample_streaming_openr1math(dataset, tokenizer, nsamples, seqlen, seed):
    """Sample from OpenR1-Math by concatenating problem + solution text."""
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    traindataset = []
    buffer = ""
    for item in dataset:
        # OpenR1-Math has 'problem' and 'solution' fields (or 'messages')
        messages = item.get("messages", None)
        if messages:
            text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        else:
            problem = item.get("problem", "")
            solution = item.get("solution", "")
            text = f"{problem}\n{solution}"
        if not text.strip():
            continue
        buffer += text + "\n\n"
        if len(buffer) > seqlen * nsamples * 6:
            break
    enc = tokenizer(buffer, return_tensors="pt").input_ids[0]
    for i in range(nsamples):
        start = random.randint(0, max(0, enc.shape[0] - seqlen - 1))
        chunk = enc[start: start + seqlen]
        if chunk.shape[0] < seqlen:
            chunk = torch.nn.functional.pad(chunk, (0, seqlen - chunk.shape[0]))
        traindataset.append({"input_ids": chunk.unsqueeze(0), "attention_mask": torch.ones(1, seqlen)})
    return traindataset
