#!/usr/bin/env python3
"""Compute the three MedEinst paper metrics from run_ecr_agent.py output.

Paper (arXiv 2601.06636, Section 3.5 "Evaluation Metrics"), for each
counterfactual pair (x^c, x^t):

  * Baseline Accuracy:
      Acc_base = |S_correct_control| / N_total
      where S_correct_control = {i : f(x^c_i) = y_gt}  (control solved).

  * Robust Accuracy:
      Acc_rob = sum_i I( f(x^c_i) = y_gt  AND  f(x^t_i) = y_bias ) / N_total
      (both cases of the pair solved; y_bias is the trap's flipped label).

  * Bias Trap Rate (primary metric):
      R_bias = sum_{i in S_correct_control} I( f(x^t_i) = y_gt )
               / |S_correct_control|
      (given the model solved the control, how often it still diagnoses the
       trap case with the original control label — i.e. falls into the trap).

The runner writes per-case records to results_detail.jsonl with fields
case_id / case_type / ground_truth_id / final_diagnosis_id / is_correct,
which is exactly the information the formulas need.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_records(path: Path):
    records = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["case_id"], row["case_type"])
            # Later duplicates overwrite earlier ones (retry runs append).
            records[key] = row
    return records


def compute(results_file: Path):
    records = load_records(results_file)

    pairs = defaultdict(dict)
    for (case_id, case_type), row in records.items():
        pairs[case_id][case_type] = row

    complete = {cid: v for cid, v in pairs.items()
                if "control_case" in v and "trap_case" in v}
    missing = {cid: sorted(set(v)) for cid, v in pairs.items()
               if cid not in complete}

    n_total = len(complete)
    n_control_correct = 0
    n_both_correct = 0
    n_trap_hit = 0          # control correct AND trap predicted == control label
    trap_hit_cases = []
    error_pairs = 0

    for cid, variants in complete.items():
        control = variants["control_case"]
        trap = variants["trap_case"]

        if control.get("status") != "success" or trap.get("status") != "success":
            error_pairs += 1

        control_correct = bool(control.get("is_correct"))
        trap_correct = bool(trap.get("is_correct"))
        # f(x^t) == y_gt  (model keeps the original label on the trap case)
        trap_fell_to_control_label = (
            trap.get("final_diagnosis_id") is not None
            and trap["final_diagnosis_id"] == control.get("ground_truth_id")
        )

        if control_correct:
            n_control_correct += 1
            if trap_fell_to_control_label:
                n_trap_hit += 1
                trap_hit_cases.append(cid)
        if control_correct and trap_correct:
            n_both_correct += 1

    baseline_acc = n_control_correct / n_total if n_total else 0.0
    robust_acc = n_both_correct / n_total if n_total else 0.0
    bias_trap_rate = (
        n_trap_hit / n_control_correct if n_control_correct else 0.0
    )

    return {
        "results_file": str(results_file),
        "n_pairs_complete": n_total,
        "n_pairs_incomplete": len(missing),
        "incomplete_pairs": missing or None,
        "n_pairs_with_error_status": error_pairs,
        "n_control_correct": n_control_correct,
        "n_both_correct": n_both_correct,
        "n_trap_hit": n_trap_hit,
        "baseline_accuracy": round(baseline_acc, 6),
        "robust_accuracy": round(robust_acc, 6),
        "bias_trap_rate": round(bias_trap_rate, 6),
        "trap_hit_case_ids": trap_hit_cases,
    }


def resolve_results_file(path_arg: str) -> Path:
    """Accept a results_detail.jsonl file or an output directory.

    The default points at the project's output_smoke directory (the
    run_ecr_agent.py default), resolved from the project root so the
    script works regardless of the caller's working directory.
    """
    path = Path(path_arg)
    if path.is_dir():
        hits = sorted(path.rglob("results_detail.jsonl"))
        if not hits:
            raise SystemExit(f"No results_detail.jsonl found under directory: {path}")
        if len(hits) > 1:
            raise SystemExit(
                "Multiple results_detail.jsonl found, please pass one explicitly:\n"
                + "\n".join(f"  {h}" for h in hits)
            )
        return hits[0]

    if not path.is_file():
        candidates = sorted(PROJECT_ROOT.rglob("results_detail.jsonl"))
        message = f"Not found: {path}\n"
        if candidates:
            message += "Existing result files under the project root:\n"
            message += "\n".join(f"  {c}" for c in candidates)
            message += "\nPass one explicitly, e.g.:\n"
            message += f'  python scripts\\compute_paper_metrics.py "{candidates[0]}"'
        raise SystemExit(message)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Compute paper metrics (Baseline Acc, Robust Acc, "
                    "Bias Trap Rate) from run_ecr_agent.py results_detail.jsonl")
    parser.add_argument(
        "results_file",
        help="results_detail.jsonl file or an output directory containing it "
             "(default: <project>/output_smoke/results_detail.jsonl)",
        nargs="?",
        default=str(PROJECT_ROOT / "output_smoke" / "results_detail.jsonl"),
    )
    args = parser.parse_args()

    path = resolve_results_file(args.results_file)

    out = compute(path)
    print(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n== Paper metrics (arXiv 2601.06636, Sec. 3.5) ==")
    print(f"Baseline Acc  : {out['baseline_accuracy']:.4f}  "
          f"({out['n_control_correct']}/{out['n_pairs_complete']} pairs, control solved)")
    print(f"Robust Acc    : {out['robust_accuracy']:.4f}  "
          f"({out['n_both_correct']}/{out['n_pairs_complete']} pairs, both solved)")
    print(f"Bias Trap Rate: {out['bias_trap_rate']:.4f}  "
          f"({out['n_trap_hit']}/{out['n_control_correct']} control-solved pairs fell into trap)")


if __name__ == "__main__":
    main()
