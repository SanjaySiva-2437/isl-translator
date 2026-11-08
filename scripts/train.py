"""
Step 6: train a small sequence model on the landmark clips.

Input : dataset.npz from prepare_data.py (X: N x 32 x 142), label_map.json
Output: <out>/best.pt (weights), <out>/history.csv, <out>/results.txt

Usage (from the repo root, venv active):
  python scripts\\train.py --data F:\\isl-translator\\data\\processed\\dataset.npz --label-map label_map.json --out F:\\isl-translator\\data\\runs\\run1
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignNet(nn.Module):
    def __init__(self, n_classes, n_feat=142, hidden=128, dropout=0.4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2))
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, n_classes)

    def forward(self, x):                      # x: (B, T, 142)
        h, _ = self.gru(self.inp(x))           # (B, T, 2*hidden)
        z = torch.cat([h.mean(1), h.max(1).values], dim=1)
        return self.out(self.drop(z))


def augment(x, gen):
    """x: (B, T, 142) tensor. Small noise, scale, shift and frame-dropout. Presence flags are kept."""
    B, T, _ = x.shape
    x = x.clone()
    coords = torch.ones(142, dtype=torch.bool)
    coords[140:] = False                       # do not touch hand-present flags
    present = x[:, :, 140:142] > 0
    # random global scale and shift (applied to coordinates only)
    scale = 1 + 0.10 * (torch.rand(B, 1, 1, generator=gen) * 2 - 1)
    shift = 0.05 * (torch.rand(B, 1, 1, generator=gen) * 2 - 1)
    x[:, :, :140] = x[:, :, :140] * scale + shift
    # per-value gaussian noise
    x[:, :, :140] += 0.01 * torch.randn(x[:, :, :140].shape, generator=gen)
    # zero out values of absent hands again (they must stay 0)
    for slot in (0, 1):
        m = ~present[:, :, slot]
        x[:, :, slot * 63:(slot + 1) * 63][m] = 0
    # random temporal resample (speed change +-15%)
    out = torch.empty_like(x)
    for b in range(B):
        f = 1 + 0.15 * (torch.rand(1, generator=gen).item() * 2 - 1)
        centre = (T - 1) / 2
        idx = (torch.arange(T) - centre) * f + centre
        idx = idx.round().clamp(0, T - 1).long()
        out[b] = x[b, idx]
    return out


def evaluate(model, X, y, k=5):
    model.eval()
    with torch.no_grad():
        logits = model(X)
    pred = logits.argmax(1)
    top1 = (pred == y).float().mean().item()
    topk = (logits.topk(min(k, logits.shape[1]), dim=1).indices == y[:, None]).any(1).float().mean().item()
    loss = F.cross_entropy(logits, y).item()
    return loss, top1, topk, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--label-map", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    names = [n for n, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    d = np.load(args.data)
    Xtr, ytr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"])
    Xva, yva = torch.tensor(d["X_val"]), torch.tensor(d["y_val"])
    Xte, yte = torch.tensor(d["X_test"]), torch.tensor(d["y_test"])
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  test {tuple(Xte.shape)}  classes {len(names)}")

    model = SignNet(len(names))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * ((len(Xtr) + args.batch - 1) // args.batch))

    best_acc, best_loss, best_epoch, bad = -1.0, 1e9, 0, 0
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        tl, tc = 0.0, 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = augment(Xtr[idx], gen), ytr[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
            tc += (logits.argmax(1) == yb).sum().item()
        vl, va, v5, _ = evaluate(model, Xva, yva)
        hist.append([ep, tl / len(Xtr), tc / len(Xtr), vl, va, v5])
        if va > best_acc or (va == best_acc and vl < best_loss):
            best_acc, best_loss, best_epoch, bad = va, vl, ep, 0
            torch.save(model.state_dict(), args.out / "best.pt")
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:>3}  train loss {tl/len(Xtr):.3f} acc {tc/len(Xtr):.3f} | val loss {vl:.3f} acc {va:.3f} top5 {v5:.3f}")
        if bad >= args.patience:
            print(f"early stop at epoch {ep}")
            break

    with open(args.out / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_top5"])
        w.writerows(hist)

    model.load_state_dict(torch.load(args.out / "best.pt"))
    tl_, ta, t5, pred = evaluate(model, Xte, yte)
    lines = [f"best epoch {best_epoch}  val acc {best_acc:.4f}",
             f"TEST ({len(Xte)} videos): top-1 {ta:.4f}  top-5 {t5:.4f}  loss {tl_:.3f}",
             f"parameters: {n_params:,}", ""]
    wrong = [(names[int(t)], names[int(p)]) for t, p in zip(yte, pred) if t != p]
    lines.append(f"Wrong test videos ({len(wrong)}):")
    lines += [f"  {t}  ->  predicted {p}" for t, p in wrong]
    text = "\n".join(lines)
    (args.out / "results.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
