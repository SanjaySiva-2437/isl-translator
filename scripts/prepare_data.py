"""
Step 5: build label_map.json, a train/val/test split and one packed dataset.

Reads the manifest written by extract_landmarks.py and produces:
  label_map.json        word folder name -> integer (commit this to the repo)
  splits.csv            one row per video with its split (train / val / test)
  dataset.npz           X_train, y_train, X_val, y_val, X_test, y_test
  report.txt            counts per split and videos with weak hand detection

Usage (from the repo root, venv active):
  python scripts\\prepare_data.py --landmarks F:\\include_data\\landmarks --out F:\\include_data\\processed --label-map label_map.json
Add --skip-arrays to only build the label map and split (fast check, no .npy loading).
"""
import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def sort_key(name: str):
    m = re.match(r"\s*(\d+)", name)
    return (int(m.group(1)) if m else 10**9, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label-map", type=Path, default=Path("label_map.json"))
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-hand-rate", type=float, default=0.0,
                    help="drop videos whose hand_frame_rate is below this (default: keep all)")
    ap.add_argument("--skip-arrays", action="store_true")
    args = ap.parse_args()

    with open(args.landmarks / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] in ("ok", "cached")]
    weak = [r for r in rows if float(r["hand_frame_rate"]) < 0.5]
    rows = [r for r in rows if float(r["hand_frame_rate"]) >= args.min_hand_rate]

    labels = sorted({r["label"] for r in rows}, key=sort_key)
    label_map = {name: i for i, name in enumerate(labels)}
    args.label_map.write_text(json.dumps(label_map, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")

    # stratified split per word, by video
    rng = random.Random(args.seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    split_of = {}
    for lab, items in by_label.items():
        items.sort(key=lambda r: r["video_path"])
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * args.test))
        n_val = max(1, round(n * args.val))
        for i, r in enumerate(items):
            split_of[r["video_path"]] = ("test" if i < n_test
                                         else "val" if i < n_test + n_val else "train")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "label", "label_id", "split", "npy_path"])
        for r in rows:
            w.writerow([r["video_path"], r["label"], label_map[r["label"]],
                        split_of[r["video_path"]], r["npy_path"]])

    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["label"]][split_of[r["video_path"]]] += 1
    tot = defaultdict(int)
    lines = [f"{len(labels)} words, {len(rows)} videos (seed {args.seed})", ""]
    for lab in labels:
        c = counts[lab]
        lines.append(f"{label_map[lab]:>3}  {lab:<28} train {c['train']:>2}  val {c['val']:>2}  test {c['test']:>2}")
        for k in ("train", "val", "test"):
            tot[k] += c[k]
    lines += ["", f"TOTAL  train {tot['train']}  val {tot['val']}  test {tot['test']}", ""]
    lines.append(f"Videos with hands in <50% of frames: {len(weak)}")
    for r in weak:
        lines.append(f"  {r['video_path']}  ({r['hand_frame_rate']})")
    (args.out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if args.skip_arrays:
        return
    data = {s: ([], []) for s in ("train", "val", "test")}
    for r in rows:
        arr = np.load(args.landmarks / r["npy_path"])
        X, y = data[split_of[r["video_path"]]]
        X.append(arr)
        y.append(label_map[r["label"]])
    out = {}
    for s, (X, y) in data.items():
        out[f"X_{s}"] = np.stack(X).astype(np.float32)
        out[f"y_{s}"] = np.array(y, dtype=np.int64)
    np.savez_compressed(args.out / "dataset.npz", **out)
    print({k: v.shape for k, v in out.items()})
    print(f"Saved {args.out / 'dataset.npz'}")


if __name__ == "__main__":
    main()
