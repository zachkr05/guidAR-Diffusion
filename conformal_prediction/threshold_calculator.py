# threshold_calculator.py
"""
Compute a conformal threshold (quantile) using the score:

Score = (sum of probabilities) + lambda * (size of set)

Then sort scores ascending and extract the finite-sample corrected
(1 - alpha) quantile via k = ceil((N + 1) * (1 - alpha)).

Assumes your .pkl is a list of dict records like:
{
  "probs": {"chair": 0.1, "table": 0.8, "bomb": 0.1},
  "labels": ["chair", "table"],   # used only if you want to define set size from it
  ...
}

You can choose how to define "Size of Set":
- from labels length (default): len(record["labels"])
- from predicted set defined by probs >= prob_threshold
"""

import argparse
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_pkl(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def sum_probs(probs: Dict[str, Any]) -> float:
    if not isinstance(probs, dict):
        return 0.0
    s = 0.0
    for _, v in probs.items():
        try:
            s += float(v)
        except Exception:
            pass
    return float(s)


def set_size_from_labels(rec: Dict[str, Any]) -> int:
    labels = rec.get("labels", [])
    if labels is None:
        return 0
    try:
        return int(len(labels))
    except Exception:
        return 0


def set_size_from_prob_threshold(rec: Dict[str, Any], prob_threshold: float) -> int:
    probs = rec.get("probs", {})
    if not isinstance(probs, dict):
        return 0
    cnt = 0
    for _, v in probs.items():
        try:
            if float(v) >= prob_threshold:
                cnt += 1
        except Exception:
            continue
    return cnt


def compute_scores(
    records: List[Dict[str, Any]],
    lam: float,
    size_mode: str,
    prob_threshold: float,
) -> List[float]:
    scores = []
    for rec in records:
        probs = rec.get("probs", {})
        aps_part = sum_probs(probs)

        if size_mode == "labels":
            size = set_size_from_labels(rec)
        elif size_mode == "prob_threshold":
            size = set_size_from_prob_threshold(rec, prob_threshold)
        else:
            raise ValueError("size_mode must be 'labels' or 'prob_threshold'")

        score = aps_part + lam * float(size)
        scores.append(float(score))
    return scores


def finite_sample_quantile(sorted_scores: List[float], alpha: float) -> Tuple[int, float]:
    """
    Return (k, threshold) where
      k = ceil((N + 1) * (1 - alpha))
    and threshold is the k-th smallest score with 1-indexing,
    clipped to [1, N].
    """
    N = len(sorted_scores)
    if N == 0:
        raise ValueError("No scores to compute threshold from (N=0).")

    k = math.ceil((N + 1) * (1.0 - alpha))  # 1-indexed rank
    k = max(1, min(N, k))                   # clip
    threshold = sorted_scores[k - 1]        # convert to 0-index
    return k, threshold


def main():
    parser = argparse.ArgumentParser(description="Compute finite-sample conformal threshold from score.")
    parser.add_argument("--pkl", type=str, required=True, help="Input .pkl file (list of records).")
    parser.add_argument("--alpha", type=float, default=0.2, help="Miscoverage level alpha (0.2 -> 80% quantile).")
    parser.add_argument("--lambda_", type=float, default=0.0, help="Penalty weight lambda.")
    parser.add_argument(
        "--size_mode",
        type=str,
        default="labels",
        choices=["labels", "prob_threshold"],
        help="How to compute 'Size of Set'.",
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Used only if size_mode=prob_threshold: count probs >= this threshold.",
    )
    parser.add_argument("--drop_unlabeled", action="store_true", help="Drop records with empty/missing labels (if present).")
    parser.add_argument("--print_top", type=int, default=0, help="Print the largest K scores (debug).")
    args = parser.parse_args()

    data = load_pkl(Path(args.pkl))
    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {args.pkl}, got {type(data)}")

    records: List[Dict[str, Any]] = [r for r in data if isinstance(r, dict)]

    if args.drop_unlabeled:
        filtered = []
        for r in records:
            labels = r.get("labels", [])
            if labels is None or len(labels) == 0:
                continue
            filtered.append(r)
        records = filtered

    scores = compute_scores(
        records=records,
        lam=args.lambda_,
        size_mode=args.size_mode,
        prob_threshold=args.prob_threshold,
    )

    scores_sorted = sorted(scores)  # ascending
    k, thr = finite_sample_quantile(scores_sorted, alpha=args.alpha)

    N = len(scores_sorted)
    print(f"N = {N}")
    print(f"alpha = {args.alpha}")
    print(f"lambda = {args.lambda_}")
    print(f"size_mode = {args.size_mode}" + (f" (prob_threshold={args.prob_threshold})" if args.size_mode == "prob_threshold" else ""))
    print(f"k = ceil((N+1)*(1-alpha)) = {k}")
    print(f"threshold (k-th smallest) = {thr}")

    if args.print_top and args.print_top > 0:
        top = scores_sorted[-args.print_top:]
        print(f"\nTop {args.print_top} largest scores:")
        for v in top:
            print(v)


if __name__ == "__main__":
    main()

