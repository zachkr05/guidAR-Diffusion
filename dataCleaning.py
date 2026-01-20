# dataCleaning.py
"""
Balance a labeled costmap .pkl log so you have the SAME number of:
  - 1-label records
  - 2-label records
  - 3-label records

Also:
  - drops unlabeled records (labels missing/empty)
  - removes duplicates (same x, y, labels, probs) with float rounding for probs
  - downsamples each label-count group to the smallest group size (no oversampling)

Expected input: .pkl contains a list of dicts, each with at least:
  - "labels": list[str]
  - "probs": dict[str, float]
  - "x", "y" (optional but used for dedupe key)
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import random
import math


def normalize_labels(labels) -> Tuple[str, ...]:
    """Canonical label tuple: sorted, stripped, non-empty strings."""
    if not isinstance(labels, list) or len(labels) == 0:
        return tuple()
    out = []
    for x in labels:
        s = str(x).strip()
        if s:
            out.append(s)
    return tuple(sorted(out))


def normalize_probs(probs: Dict[str, Any], decimals: int) -> Tuple[Tuple[str, float], ...]:
    """Canonical probs: sorted items with rounded floats."""
    if not isinstance(probs, dict) or len(probs) == 0:
        return tuple()
    items = []
    for k, v in probs.items():
        try:
            items.append((str(k), round(float(v), decimals)))
        except Exception:
            # Skip non-numeric entries
            continue
    items.sort(key=lambda x: x[0])
    return tuple(items)


def make_dedupe_key(rec: Dict[str, Any], decimals: int) -> Tuple[Any, ...]:
    """Duplicate = same (x, y, labels, probs) after normalization."""
    x = int(rec.get("x", -1))
    y = int(rec.get("y", -1))
    labels = normalize_labels(rec.get("labels", []))
    probs = normalize_probs(rec.get("probs", {}), decimals=decimals)
    return (x, y, labels, probs)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def drop_unlabeled(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove records with missing/empty labels; write back normalized labels."""
    out = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        lab = normalize_labels(rec.get("labels", None))
        if len(lab) == 0:
            continue
        rec["labels"] = list(lab)  # normalize in-place
        out.append(rec)
    return out


def dedupe_records(records: List[Dict[str, Any]], decimals: int, keep: str) -> List[Dict[str, Any]]:
    """
    Remove duplicates while preserving order.
    keep='first' keeps first occurrence.
    keep='last' keeps last occurrence.
    """
    if keep not in ("first", "last"):
        raise ValueError("keep must be 'first' or 'last'")

    if keep == "first":
        seen = set()
        out = []
        for rec in records:
            k = make_dedupe_key(rec, decimals)
            if k in seen:
                continue
            seen.add(k)
            out.append(rec)
        return out

    seen = set()
    out_rev = []
    for rec in reversed(records):
        k = make_dedupe_key(rec, decimals)
        if k in seen:
            continue
        seen.add(k)
        out_rev.append(rec)
    return list(reversed(out_rev))


def balance_by_label_count(records: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], Dict[int, int], int]:
    """
    Ensure equal number of records for label-counts 1, 2, 3 by downsampling.
    Only keeps records with 1<=len(labels)<=3.
    """
    groups = defaultdict(list)  # key: label_count -> records
    for rec in records:
        n = len(rec.get("labels", []))
        if n in (1, 2, 3):
            groups[n].append(rec)

    sizes = {k: len(v) for k, v in groups.items()}

    # Need all three groups present to truly balance
    if not all(k in groups for k in (1, 2, 3)):
        missing = [k for k in (1, 2, 3) if k not in groups or len(groups[k]) == 0]
        raise ValueError(f"Cannot balance: missing label-count groups {missing}. Current sizes: {sizes}")

    min_size = min(len(groups[1]), len(groups[2]), len(groups[3]))

    rng = random.Random(seed)
    balanced = []
    for k in (1, 2, 3):
        recs = groups[k]
        idx = list(range(len(recs)))
        rng.shuffle(idx)
        chosen = [recs[i] for i in idx[:min_size]]
        balanced.extend(chosen)

    # Optional: stable ordering for downstream reproducibility
    def sort_key(r):
        return (
            float(r.get("timestamp", 0.0)),
            int(r.get("y", -1)),
            int(r.get("x", -1)),
            tuple(r.get("labels", [])),
        )
    balanced.sort(key=sort_key)

    return balanced, sizes, min_size


def main():
    parser = argparse.ArgumentParser(description="Deduplicate + drop unlabeled + balance 1/2/3-label records equally.")
    parser.add_argument("--in_pkl", type=str, default="combined.pkl", help="Input pickle file.")
    parser.add_argument("--out_pkl", type=str, default="combined_balanced_123.pkl", help="Output pickle file.")
    parser.add_argument("--decimals", type=int, default=8, help="Rounding for probs for dedupe stability.")
    parser.add_argument("--keep", type=str, default="last", choices=["first", "last"], help="Which duplicate to keep.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for balancing downsample.")
    args = parser.parse_args()

    in_path = Path(args.in_pkl)
    out_path = Path(args.out_pkl)

    data = load_pickle(in_path)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {in_path}, got {type(data)}")

    before = len(data)

    labeled = drop_unlabeled(data)
    after_drop = len(labeled)

    deduped = dedupe_records(labeled, decimals=args.decimals, keep=args.keep)
    after_dedupe = len(deduped)

    balanced, sizes, min_size = balance_by_label_count(deduped, seed=args.seed)
    after_balance = len(balanced)

    save_pickle(balanced, out_path)

    print(f"Loaded:                      {in_path} ({before} records)")
    print(f"After dropping unlabeled:               ({after_drop} records) removed {before - after_drop}")
    print(f"After dedupe (x,y,labels,probs):        ({after_dedupe} records) removed {after_drop - after_dedupe}")
    print("Label-count sizes BEFORE balance:", sizes)
    print(f"Balanced per group to: {min_size} each -> total {after_balance}")
    print(f"Saved:                       {out_path}")


if __name__ == "__main__":
    main()

