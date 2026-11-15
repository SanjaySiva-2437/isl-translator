"""
Train several SignNet models with different random seeds, average them, and (optionally) export the
average as ONE ONNX file that the app can use unchanged (same input "clip", same output "logits").

Why: each model makes slightly different mistakes; averaging their probabilities cancels many of them
and makes the confidence scores more trustworthy. It costs N times the training time (about 3 minutes per model).

Usage (repo root, venv active), 5 models, export straight into the app:
  python scripts\\ensemble.py --data data\\processed\\dataset_plus.npz --label-map data\\processed\\label_map_plus.json --feats dir,palm,pos,ext --n 5 --out data\\runs\\ens1 --export app\\model

Add --skip-train to re-export models that are already trained in --out (folders m0, m1, ...).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_model  # noqa: E402


class AverageNet(nn.Module):
    """Average of the models' probabilities. The output is log(average probability), so the app's own
    softmax gives back exactly the averaged probabilities."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        p = torch.stack([F.softmax(m(x), dim=-1) for m in self.models]).mean(0)
        return torch.log(p + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--label-map", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--feats", default="")
    ap.add_argument("--aug", default="strong")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--export", type=Path, default=None, help="folder to write sign_model.onnx and labels.json into (e.g. app\\model)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_train:
        for k in range(args.n):
            print(f"\n===== training model {k + 1} of {args.n} =====", flush=True)
            cmd = [sys.executable, str(here / "train.py"), "--data", str(args.data), "--label-map", str(args.label_map),
                   "--out", str(args.out / f"m{k}"), "--seed", str(args.seed + k), "--aug", args.aug,
                   "--epochs", str(args.epochs)]
            if args.feats:
                cmd += ["--feats", args.feats]
            subprocess.run(cmd, check=True)

    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    ordered = [n for n, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    models = [load_model(args.out / f"m{k}" / "best.pt", len(ordered)) for k in range(args.n)]
    ens = AverageNet(models).eval()

    d = np.load(args.data)
    X, y = torch.tensor(d["X_test"]), torch.tensor(d["y_test"])
    n0 = int(d["n_orig_test"]) if "n_orig_test" in d.files else len(y)
    with torch.no_grad():
        singles = [m(X).argmax(1) for m in models]
        pred = ens(X).argmax(1)
    print("\nTEST RESULTS")
    for k, p in enumerate(singles):
        print(f"  model {k + 1}: {(p == y).float().mean().item():.4f}")
    print(f"  ENSEMBLE of {args.n}: {(pred == y).float().mean().item():.4f}  ({len(y)} clips)")
    print(f"    original test clips: {(pred[:n0] == y[:n0]).float().mean().item():.4f} ({n0} clips)")
    if len(y) > n0:
        print(f"    your own held-out clips: {(pred[n0:] == y[n0:]).float().mean().item():.4f} ({len(y) - n0} clips)")

    if args.export is not None:
        args.export.mkdir(parents=True, exist_ok=True)
        onnx_path = args.export / "sign_model.onnx"
        torch.onnx.export(ens, torch.zeros(1, 32, 142), str(onnx_path), input_names=["clip"], output_names=["logits"],
                          dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
        display = [re.sub(r"^\s*\d+\.\s*", "", n) for n in ordered]
        (args.export / "labels.json").write_text(json.dumps(display, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        got = sess.run(["logits"], {"clip": X.numpy().astype(np.float32)})[0]
        with torch.no_grad():
            ref = ens(X).numpy()
        diff = float(np.abs(ref - got).max())
        print(f"\nWrote {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB) and labels.json")
        print(f"max |pytorch - onnx| over {len(X)} test clips: {diff:.2e}")
        print("PARITY OK" if diff < 1e-3 else "PARITY FAILED - do not use this model file")


if __name__ == "__main__":
    main()
