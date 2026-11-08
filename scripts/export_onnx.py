"""
Step 7a: export the trained model to ONNX and check it matches the PyTorch model.

Writes:
  <out>/sign_model.onnx   the model (input "clip": 1 x 32 x 142, output "logits": 1 x N)
  <out>/labels.json       class names in model order, display names without the "48. " prefix

Usage (from the repo root, venv active):
  pip install onnx onnxruntime
  python scripts\\export_onnx.py --weights F:\\isl-translator\\data\\runs\\run1\\best.pt --data F:\\isl-translator\\data\\processed\\dataset.npz --label-map label_map.json --out app\\model
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SignNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--label-map", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    ordered = [n for n, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    display = [re.sub(r"^\s*\d+\.\s*", "", n) for n in ordered]

    model = SignNet(len(ordered))
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "sign_model.onnx"
    dummy = torch.zeros(1, 32, 142)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["clip"], output_names=["logits"],
        dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    (args.out / "labels.json").write_text(
        json.dumps(display, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB) and labels.json")

    # ---- parity check against PyTorch on the real test clips ----
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    d = np.load(args.data)
    X, y = d["X_test"], d["y_test"]
    with torch.no_grad():
        ref = model(torch.tensor(X)).numpy()
    got = sess.run(["logits"], {"clip": X.astype(np.float32)})[0]
    diff = float(np.abs(ref - got).max())
    acc = float((got.argmax(1) == y).mean())
    print(f"max |pytorch - onnx| over {len(X)} test clips: {diff:.2e}")
    print(f"ONNX test top-1 accuracy: {acc:.4f}")
    print("PARITY OK" if diff < 1e-3 else "PARITY FAILED - do not use this model file")


if __name__ == "__main__":
    main()
