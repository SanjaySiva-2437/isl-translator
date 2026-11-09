"""
Robustness check: score a trained model on the TEST clips under camera-like changes.

Compare two runs (e.g. run1 with --aug basic vs run2 with --aug strong) to see which copes better with
tilt, a turned body, a camera above/below, lost hands, a mirrored (left-handed) signer, and so on.

Usage (repo root, venv active):
  python scripts\\stress_test.py --data data\\processed\\dataset.npz --label-map label_map.json --weights data\\runs\\run1\\best.pt --out data\\runs\\run1\\stress.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from augmentation import NONE, augment
from train import SignNet

CONDITIONS = {
    "clean (no change)": {},
    "tilt up to 10 deg": dict(rot=10),
    "tilt up to 20 deg": dict(rot=20),
    "body turned up to 20 deg": dict(yaw=20),
    "body turned up to 35 deg": dict(yaw=35),
    "camera above/below 15 deg": dict(pitch=15),
    "stretched / sheared": dict(sx=(0.8, 1.2), sy=(0.85, 1.15), shear=0.15),
    "landmark jitter (heavy)": dict(noise_hand=0.02, noise_pose=0.04),
    "10% hand detections missed": dict(drop_rate=0.1),
    "30% hand detections missed": dict(drop_rate=0.3),
    "a hand lost 3-8 frames": dict(chunk_p=1.0),
    "hands leave top of frame": dict(top_p=1.0),
    "different speed / trimming": dict(speed=(0.7, 1.4), pad=0.2),
    "mirrored (left-handed signer)": dict(mirror_p=1.0),
    "everything together": dict(rot=12, yaw=20, pitch=12, shear=0.1, drop_rate=0.15, chunk_p=0.3, top_p=0.25,
                                noise_hand=0.01, noise_pose=0.02, speed=(0.8, 1.25), pad=0.15),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--label-map", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    n_classes = len(json.loads(args.label_map.read_text(encoding="utf-8")))
    d = np.load(args.data)
    X, y = d["X_test"].astype(np.float32), torch.tensor(d["y_test"])
    model = SignNet(n_classes)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    lines = [f"Stress test on {len(X)} test clips ({args.repeats} random repeats per condition)",
             f"weights: {args.weights}", "", f"{'condition':<34}top-1"]
    for name, cfg in CONDITIONS.items():
        reps = 1 if not cfg else args.repeats
        accs = []
        for r in range(reps):
            rng = np.random.default_rng(100 + r)
            xa = augment(X, rng, {**NONE, **cfg})
            with torch.no_grad():
                pred = model(torch.from_numpy(xa)).argmax(1)
            accs.append((pred == y).float().mean().item())
        lines.append(f"{name:<34}{100 * np.mean(accs):5.1f}%")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
