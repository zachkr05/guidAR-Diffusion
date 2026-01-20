# combine_pkls.py
import argparse
import pickle
from pathlib import Path
from typing import Any, List

def load_pkl(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)

def save_pkl(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)

def ensure_list(obj: Any, name: str) -> List[Any]:
    if not isinstance(obj, list):
        raise TypeError(f"{name} is {type(obj)}, expected a list (like your rect_probs logs).")
    return obj

def main():
    parser = argparse.ArgumentParser(description="Combine multiple .pkl files (lists) into one .pkl.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["probs_balanced_2.pkl", "rect_probs_clean_balanced.pkl", "rect_probs_dedup.pkl"],
        help="Input pkl files (each must be a list).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="combined.pkl",
        help="Output pkl filename.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Optional: remove exact duplicate dicts (by stable repr).",
    )
    args = parser.parse_args()

    combined: List[Any] = []
    for p in args.inputs:
        path = Path(p)
        data = load_pkl(path)
        data_list = ensure_list(data, str(path))
        combined.extend(data_list)
        print(f"Loaded {path} : {len(data_list)} records")

    print(f"Combined total: {len(combined)} records")

    if args.dedupe:
        # Exact-dedupe by stable, hashable representation.
        # Works well if records are dicts/lists of basic types.
        seen = set()
        deduped = []
        for rec in combined:
            key = repr(rec)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)
        print(f"After exact dedupe: {len(deduped)} records (removed {len(combined) - len(deduped)})")
        combined = deduped

    save_pkl(combined, Path(args.out))
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()

