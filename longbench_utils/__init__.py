"""LongBench v1 prompt templates and scoring metrics.

Vendored from the official LongBench repository (MIT licensed):
https://github.com/THUDM/LongBench

Only the evaluation harness is vendored here (prompt formats, per-task
max generation lengths, and the metric implementations) — it is a few
hundred lines that `run_longbench_comparison.py` imports directly. The
*datasets* are not stored in this repo; they are pulled from the Hugging
Face hub at run time via `load_dataset("THUDM/LongBench", <task>)`.
"""

from .scorer import scorer

import json
import os


MODEL2MAXLEN = json.load(open(os.path.join(os.path.dirname(__file__), "config/model2maxlen.json"), "r"))
DATASET2PROMPT = json.load(open(os.path.join(os.path.dirname(__file__), "config/dataset2prompt.json"),"r"))
DATASET2MAXLEN = json.load(open(os.path.join(os.path.dirname(__file__), "config/dataset2maxlen.json"), "r"))
