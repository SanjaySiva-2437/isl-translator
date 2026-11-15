"""
Merge clips you recorded in the app (Collect mode) into the training data.

Input : the original dataset.npz, label_map.json, and one or more clips_*.json files downloaded from the app
Output: a new dataset npz with the same keys, so scripts/train.py works on it unchanged:
  - your clips are added to the TRAIN set (each repeated --repeat times; augmentation makes every copy different)
  - a share of them (--holdout, per word) is kept out and added to the TEST set, so the score on your own clips is honest
  - "n_orig_test" marks where your holdout clips start in X_test, so train.py reports the two scores separately

  - a smaller share (--val-share, per word) goes to the VALIDATION set, so the best training epoch is also chosen on your own clips
  - with --add-new, words that are not in label_map.json become NEW classes; the extended map is written next to --out as label_map_plus.json
    (use that file with train.py and export_onnx.py instead of label_map.json)

Usage (repo root, venv active):
  python scripts\\merge_clips.py --dataset data\\processed\\dataset.npz --label-map label_map.json --clips clips_*.json --out data\\processed\\dataset_plus.npz --add-new --repeat 6
"""
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

T, NF = 32, 142


def norm(name):
    """'48. Hello' / 'Hello' / 'hello ' -> 'hello'."""
    return re.sub(r"^\s*\d+\.\s*", "", name).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--label-map", required=True, type=Path)
    ap.add_argument("--clips", required=True, nargs="+", help="clips_*.json files (wildcards allowed)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--holdout", type=float, default=0.25, help="share of your clips per word kept for testing")
    ap.add_argument("--repeat", type=int, default=3, help="copies of each of your training clips")
    ap.add_argument("--val-share", type=float, default=0.15, help="share of your clips per word used for validation")
    ap.add_argument("--add-new", action="store_true", help="turn words that are not in label_map.json into new classes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    by_name = {norm(k): v for k, v in label_map.items()}
    new_words = {}                                   # display name -> new class id (only with --add-new)
    files = sorted({f for pat in args.clips for f in glob.glob(pat)})
    if not files:
        raise SystemExit("No clip files found. Check the --clips path.")

    clips, unknown = {}, Counter()
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for c in d["clips"]:
            idx = by_name.get(norm(c["label"]))
            if idx is None and args.add_new and norm(c["label"]):
                idx = max(list(label_map.values()) + list(new_words.values())) + 1
                new_words[c["label"].strip()] = idx
                label_map[c["label"].strip()] = idx
                by_name[norm(c["label"])] = idx
            arr = np.asarray(c["data"], np.float32)
            if idx is None:
                unknown[c["label"]] += 1
            elif arr.size != T * NF:
                unknown[c["label"] + " (wrong size)"] += 1
            else:
                clips.setdefault(idx, []).append(arr.reshape(T, NF))
    if unknown:
        print("Skipped clips with unknown label:", dict(unknown))
    if not clips:
        raise SystemExit("No usable clips.")

    rng = np.random.default_rng(args.seed)
    d = np.load(args.dataset)
    Xtr, ytr = [d["X_train"]], [d["y_train"]]
    Xva, yva = [d["X_val"]], [d["y_val"]]
    Xte, yte = [d["X_test"]], [d["y_test"]]
    n_orig_test = len(d["X_test"])
    names = {v: k for k, v in label_map.items()}   # includes the new words, if any
    print(f"{'word':<22}clips  train    val  holdout")
    n_train_own = n_hold = n_val_own = 0
    for idx in sorted(clips):
        arr = np.stack(clips[idx])
        arr = arr[rng.permutation(len(arr))]
        n_hold_w = int(round(args.holdout * len(arr))) if len(arr) >= 4 else 0
        n_val_w = max(1, int(round(args.val_share * len(arr)))) if len(arr) >= 8 else 0
        hold, val, train = arr[:n_hold_w], arr[n_hold_w:n_hold_w + n_val_w], arr[n_hold_w + n_val_w:]
        if len(val):
            Xva.append(val)
            yva.append(np.full(len(val), idx, dtype=d["y_val"].dtype))
        if len(train):
            Xtr.append(np.repeat(train, args.repeat, axis=0))
            ytr.append(np.full(len(train) * args.repeat, idx, dtype=d["y_train"].dtype))
        if len(hold):
            Xte.append(hold)
            yte.append(np.full(len(hold), idx, dtype=d["y_test"].dtype))
        n_train_own += len(train); n_hold += len(hold); n_val_own += len(val)
        print(f"{names[idx]:<22}{len(arr):>5}  {len(train):>5}  {len(val):>5}  {len(hold):>7}")

    np.savez_compressed(
        args.out,
        X_train=np.concatenate(Xtr).astype(np.float32), y_train=np.concatenate(ytr),
        X_val=np.concatenate(Xva).astype(np.float32), y_val=np.concatenate(yva),
        X_test=np.concatenate(Xte).astype(np.float32), y_test=np.concatenate(yte),
        n_orig_test=np.int64(n_orig_test),
    )
    print(f"\nYour clips: {n_train_own} for training (x{args.repeat}), {n_val_own} for validation, {n_hold} held out for testing")
    print(f"Wrote {args.out}")
    if new_words:
        plus = args.out.parent / "label_map_plus.json"
        plus.write_text(json.dumps(label_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"New words added as classes: {sorted(new_words, key=new_words.get)}")
        print(f"Wrote {plus}   <- use this with train.py and export_onnx.py (it has {len(label_map)} classes)")
    if n_hold == 0:
        print("Note: no holdout (need at least 4 clips of a word). Your own-clip score will not be reported.")


if __name__ == "__main__":
    main()
