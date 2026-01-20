# aps_threshold_first_hit.py
import argparse
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def load_pkl(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)

def finite_sample_quantile(sorted_scores: List[float], alpha: float) -> Tuple[int, float]:
    """
    k = ceil((N+1)*(1-alpha)) with 1-indexing; threshold is k-th smallest score.
    """
    N = len(sorted_scores)
    if N == 0:
        raise ValueError("No usable calibration examples (N=0).")
    k = math.ceil((N + 1) * (1.0 - alpha))
    k = max(1, min(N, k))
    return k, sorted_scores[k - 1]

def sorted_probs_desc(probs: Dict[str, Any]) -> List[Tuple[str, float]]:
    items = []
    for cls, p in probs.items():
        try:
            items.append((str(cls), float(p)))
        except Exception:
            continue
    items.sort(key=lambda x: x[1], reverse=True)
    return items

def aps_score_single_label(sorted_items: List[Tuple[str, float]], y_true: str) -> Optional[float]:
    """
    Given probs sorted descending, return cumulative sum up to y_true.
    """
    cum = 0.0
    for cls, p in sorted_items:
        cum += p
        if cls == y_true:
            return float(cum)
    return None

def aps_score_first_hit(probs: Dict[str, Any], y_set: List[str]) -> Optional[float]:
    """
    First-hit APS for multi-label:
      s = min_{y in Y} cumulative_sum_up_to_y
    (i.e., stop when you reach the highest-ranked true label.)
    """
    if not isinstance(probs, dict) or len(probs) == 0:
        return None
    if not isinstance(y_set, list) or len(y_set) == 0:
        return None

    sorted_items = sorted_probs_desc(probs)

    scores = []
    for y in y_set:
        y = str(y)
        s = aps_score_single_label(sorted_items, y)
        if s is not None:
            scores.append(s)

    if not scores:
        return None  # none of the true labels existed in probs
    return float(min(scores))

def normalize_labels(labels) -> List[str]:
    if labels is None:
        return []
    if not isinstance(labels, list):
        return []
    out = []
    for x in labels:
        s = str(x).strip()
        if s:
            out.append(s)
    return out

def main():
    parser = argparse.ArgumentParser(description="Compute APS qhat using FIRST-HIT multi-label score.")
    parser.add_argument("--pkl", type=str, required=True, help="Input .pkl file (list of dict records).")
    parser.add_argument("--alpha", type=float, default=0.2, help="Miscoverage level alpha (0.2 => 80% coverage).")
    parser.add_argument("--drop_unlabeled", action="store_true", help="Drop records with missing/empty labels.")
    parser.add_argument("--print_stats", action="store_true", help="Print stats and a few sample scores.")
    # Add near the argparse section
    parser.add_argument(
        "--exclude_all_three",
        action="store_true",
        help="Exclude records whose labels contain all three classes (chair, table, bomb)."
    )


    args = parser.parse_args()

    data = load_pkl(Path(args.pkl))
    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {args.pkl}, got {type(data)}")

    scores: List[float] = []
    skipped = 0
    skipped_reasons = {
        "not_dict": 0,
        "no_labels": 0,
        "bad_probs": 0,
        "excluded_all_three": 0,
        "no_true_label_in_probs": 0,
    }

    for rec in data:
        if not isinstance(rec, dict):
            skipped += 1
            skipped_reasons["not_dict"] += 1
            continue

        labels = normalize_labels(rec.get("labels", None))
        if args.drop_unlabeled and len(labels) == 0:
            skipped += 1
            skipped_reasons["no_labels"] += 1
            continue
        
        if args.exclude_all_three:
            if set(labels) == {"chair", "table", "bomb"}:
                skipped += 1
                skipped_reasons["excluded_all_three"] = skipped_reasons.get("excluded_all_three", 0) + 1
                continue

        if len(labels) == 0:
            # still skip; can't compute APS without truth set
            skipped += 1
            skipped_reasons["no_labels"] += 1
            continue

        probs = rec.get("probs", None)
        if not isinstance(probs, dict) or len(probs) == 0:
            skipped += 1
            skipped_reasons["bad_probs"] += 1
            continue

        s = aps_score_first_hit(probs, labels)
        if s is None:
            skipped += 1
            skipped_reasons["no_true_label_in_probs"] += 1
            continue

        scores.append(s)

    scores_sorted = sorted(scores)
    k, qhat = finite_sample_quantile(scores_sorted, alpha=args.alpha)

    print(f"usable N = {len(scores_sorted)}")
    print(f"skipped  = {skipped}  (breakdown: {skipped_reasons})")
    print(f"alpha = {args.alpha}  => target coverage = {1.0-args.alpha:.3f}")
    print(f"k = ceil((N+1)*(1-alpha)) = {k}")
    print(f"qhat (k-th smallest FIRST-HIT APS score) = {qhat}")

    if args.print_stats and scores_sorted:
        import statistics as stats
        print("\nscore stats:")
        print(f"  min/mean/median/max = {min(scores_sorted):.6f} / {stats.mean(scores_sorted):.6f} / {stats.median(scores_sorted):.6f} / {max(scores_sorted):.6f}")
        print("  first 10 sorted scores:", [round(x, 6) for x in scores_sorted[:10]])
        print("  last  10 sorted scores:", [round(x, 6) for x in scores_sorted[-10:]])

if __name__ == "__main__":
    main()

