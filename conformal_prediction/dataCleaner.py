# dataCleaner.py
import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import random

def normalize_labels(labels) -> Tuple[str, ...]:
    """Sort labels so ['table','chair'] and ['chair','table'] match."""
    if labels is None:
        return tuple()
    # remove empty/None strings if any
    cleaned = [str(x).strip() for x in labels if str(x).strip()]
    return tuple(sorted(cleaned))

def normalize_probs(probs: Dict[str, Any], decimals: int) -> Tuple[Tuple[str, float], ...]:
    """
    Convert probs dict to stable, hashable representation:
    - sort by class name
    - round floats to avoid tiny FP diffs
    """
    if not isinstance(probs, dict):
        return tuple()
    items = []
    for k, v in probs.items():
        try:
            fv = float(v)
        except Exception:
            # If something weird is in probs, use NaN so it hashes consistently
            fv = float("nan")
        items.append((str(k), round(fv, decimals)))
    items.sort(key=lambda x: x[0])
    return tuple(items)

def make_dedupe_key(rec: Dict[str, Any], decimals: int) -> Tuple[Any, ...]:
    """
    Duplicate definition:
      same x, y, labels, probs
    """
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

def dedupe_records(records: List[Dict[str, Any]], decimals: int, keep: str = "last") -> List[Dict[str, Any]]:
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

    # keep == "last": scan reversed, then reverse back
    seen = set()
    out_rev = []
    for rec in reversed(records):
        k = make_dedupe_key(rec, decimals)
        if k in seen:
            continue
        seen.add(k)
        out_rev.append(rec)
    return list(reversed(out_rev))

def drop_unlabeled(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove entries with missing/empty labels."""
    out = []
    for rec in records:
        labels = normalize_labels(rec.get("labels", None))
        if len(labels) == 0:
            continue
        # write back normalized labels so downstream is consistent
        rec["labels"] = list(labels)
        out.append(rec)
    return out

def balance_by_label_combo(
    records: List[Dict[str, Any]],
    seed: int = 0,
    mode: str = "downsample"
) -> List[Dict[str, Any]]:
    """
    Make all label-combination groups have equal size.

    mode:
      - "downsample" (default): each group -> min_group_size
        (No oversampling; safest to avoid duplicates.)
    """
    if mode != "downsample":
        raise ValueError("Only 'downsample' is supported (avoids creating duplicates).")

    groups = defaultdict(list)
    for rec in records:
        key = tuple(sorted(rec.get("labels", [])))
        groups[key].append(rec)

    if not groups:
        return []

    sizes = {k: len(v) for k, v in groups.items()}
    min_size = min(sizes.values())

    rng = random.Random(seed)
    balanced = []
    for k, recs in groups.items():
        if len(recs) <= min_size:
            # already small enough
            balanced.extend(recs)
        else:
            # downsample deterministically with seed
            # shuffle copy so we don't mutate original list ordering elsewhere
            idx = list(range(len(recs)))
            rng.shuffle(idx)
            chosen = [recs[i] for i in idx[:min_size]]
            balanced.extend(chosen)

    # Optional: keep a stable order overall (by timestamp then x,y) if present.
    # If you want to preserve original order exactly, remove the sort below.
    def sort_key(r):
        return (
            float(r.get("timestamp", 0.0)),
            int(r.get("y", -1)),
            int(r.get("x", -1)),
            tuple(sorted(r.get("labels", []))),
        )
    balanced.sort(key=sort_key)

    return balanced, sizes, min_size

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean and balance labeled costmap .pkl log:\n"
            " - remove unlabeled entries\n"
            " - remove duplicates (x,y,labels,probs)\n"
            " - balance counts across label combinations (downsample to smallest group)\n"
        )
    )
    parser.add_argument("--in_pkl", type=str, default="rect_probs.pkl",
                        help="Input pickle file (list of dict records).")
    parser.add_argument("--out_pkl", type=str, default="rect_probs_clean_balanced.pkl",
                        help="Output pickle file.")
    parser.add_argument("--decimals", type=int, default=8,
                        help="Rounding for probs when deduping (float stability).")
    parser.add_argument("--keep", type=str, default="last", choices=["first", "last"],
                        help="Which duplicate to keep during dedupe.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for balancing downsample shuffles.")
    args = parser.parse_args()

    in_path = Path(args.in_pkl)
    out_path = Path(args.out_pkl)

    data = load_pickle(in_path)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {in_path}, got {type(data)}")

    before = len(data)

    # 1) remove unlabeled
    labeled = drop_unlabeled(data)
    after_drop = len(labeled)

    # 2) dedupe
    deduped = dedupe_records(labeled, decimals=args.decimals, keep=args.keep)
    after_dedupe = len(deduped)

    # 3) balance label combos
    balanced, sizes, min_size = balance_by_label_combo(deduped, seed=args.seed, mode="downsample")
    after_balance = len(balanced)

    save_pickle(balanced, out_path)

    print(f"Loaded:                 {in_path} ({before} records)")
    print(f"After dropping unlabeled:           ({after_drop} records)  removed {before - after_drop}")
    print(f"After dedupe (x,y,labels,probs):    ({after_dedupe} records)  removed {after_drop - after_dedupe}")
    print(f"Label combo counts BEFORE balance:")
    for k in sorted(sizes.keys(), key=lambda x: (len(x), x)):
        print(f"  {k}: {sizes[k]}")
    print(f"Balancing to min group size: {min_size}")
    print(f"Saved balanced:            {out_path} ({after_balance} records)")

if __name__ == "__main__":
    main()

