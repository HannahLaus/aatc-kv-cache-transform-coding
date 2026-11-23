#!/usr/bin/env python3
"""Rescore saved MATH-500 generations without rerunning the model.

run_gsm8k_mmlu.py writes <method>_math500_records.json (one record per problem
with the gold answer and the model's boxed answer). This script recomputes the
accuracy with the same comparison function the eval uses (_answers_equal), so
installing math-verify after a run and rescoring gives exactly the number the
run would have produced with it installed. It also prints the plain
normalised-string-match accuracy so the two scorers can be compared.

    pip install math-verify
    python scripts/rescore_math500.py results/nemo/gsm8k_mmlu/*_math500_records.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_gsm8k_mmlu import _answers_equal, _normalize_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="<method>_math500_records.json files")
    ap.add_argument("--write", action="store_true",
                    help="Update the sibling <method>_math500.json summary in place")
    args = ap.parse_args()

    try:
        import math_verify  # noqa: F401
        scorer = "math_verify"
    except ImportError:
        scorer = "string match (math_verify NOT installed)"
    print(f"scorer: {scorer}")

    for path in args.records:
        recs = json.load(open(path))
        n = len(recs)
        if n == 0:
            print(f"{path}: no records")
            continue
        sm = sum(_normalize_answer(r["gold"]) == _normalize_answer(r["pred_boxed"]) for r in recs)
        mv = sum(bool(_answers_equal(r["gold"], r["pred_boxed"])) for r in recs)
        print(f"{path}: n={n}  string_match={sm / n:.4f}  {scorer}={mv / n:.4f}  "
              f"(stored in run: {sum(bool(r['correct']) for r in recs) / n:.4f})")
        if args.write:
            summ_path = path.replace("_math500_records.json", "_math500.json")
            if os.path.exists(summ_path):
                summ = json.load(open(summ_path))
                summ["accuracy"] = mv / n
                summ["correct"] = mv
                summ["scorer"] = scorer
                by_subject = {}
                for r in recs:
                    d = by_subject.setdefault(r.get("subject", "unknown"), {"correct": 0, "total": 0})
                    d["total"] += 1
                    d["correct"] += int(bool(_answers_equal(r["gold"], r["pred_boxed"])))
                for d in by_subject.values():
                    d["accuracy"] = d["correct"] / d["total"]
                summ["by_subject"] = by_subject
                json.dump(summ, open(summ_path, "w"), indent=2)
                print(f"  updated {summ_path}")


if __name__ == "__main__":
    main()
