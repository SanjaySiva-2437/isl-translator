"""
Cross-validated accuracy of the ISL SignNet model.

Why: the normal train.py scores the model on one fixed test set of 97 clips. With so few clips one
mistake moves the score by about 1 point, so we cannot tell a real improvement from luck.
Here every clip of the INCLUDE dataset (train + val + test together) is used as a test clip exactly once:
the clips are split into folds, the model is trained on the other folds and tested on the held-out fold.
The result is an accuracy from ~680 clips instead of 97, with an honest +/- range.

Usage (repo root, venv active):
  python scripts\\cv_eval.py --data F:\\isl-translator\\data\\processed\\dataset.npz --label-map label_map.json --out F:\\isl-translator\\data\\runs\\cv_baseline

Try extra pointing-direction features, compared clip by clip with an earlier run (same folds):
  python scripts\\cv_eval.py --data ... --label-map label_map.json --out ...\\cv_feats --feats dir,palm,pos,ext --compare ...\\cv_baseline\\cv_oof.npz

Quick check that everything runs (about a minute):
  python scripts\\cv_eval.py --data ... --label-map label_map.json --out ...\\cv_smoke --smoke
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augmentation import augment as augment_strong
from cv_utils import confusion_pairs, stratified_folds, stratified_split, wilson
from isl_feats import GROUPS
from train import make_model, augment as augment_basic


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu()


def train_one_fold(Xtr, ytr, Xva, yva, n_classes, args, seed, device):
    """Same recipe as train.py: AdamW + one-cycle schedule, label smoothing, strong augmentation,
    best epoch chosen on the validation clips (never on the held-out fold)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(n_classes, args.groups).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * ((len(Xtr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    yva_t = torch.tensor(yva)
    best = (-1.0, 1e9)
    best_state, bad = None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            if args.aug == "strong":
                xb = torch.from_numpy(augment_strong(Xtr_t[idx].numpy(), rng)).float()
            else:
                xb = augment_basic(Xtr_t[idx], gen)
            loss = F.cross_entropy(model(xb.to(device)), ytr_t[idx].to(device), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        logits = predict(model, Xva, device)
        va = (logits.argmax(1) == yva_t).float().mean().item()
        vl = F.cross_entropy(logits, yva_t).item()
        if va > best[0] or (va == best[0] and vl < best[1]):
            best, bad = (va, vl), 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, ep, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="dataset.npz made by prepare_data.py")
    ap.add_argument("--label-map", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1, help="repeat the whole cross-validation with new random folds")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", choices=["basic", "strong"], default="strong")
    ap.add_argument("--feats", default="", help="extra features, comma separated from: " + ",".join(GROUPS) + " (default: none)")
    ap.add_argument("--ensemble", type=int, default=1, help="train this many models per fold (different seeds) and average them")
    ap.add_argument("--compare", type=Path, default=None, help="cv_oof.npz of an earlier run on the same folds, to count clips fixed/broken")
    ap.add_argument("--smoke", action="store_true", help="3 epochs and 2 folds, only to check that the script runs")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.patience = 3, 3
    args.groups = tuple(g for g in args.feats.split(",") if g)
    for g in args.groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown feature group {g!r}; choose from {GROUPS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    lm = json.loads(args.label_map.read_text(encoding="utf-8"))
    names = [n for n, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    d = np.load(args.data)
    X = np.concatenate([d["X_train"], d["X_val"], d["X_test"]]).astype(np.float32)
    y = np.concatenate([d["y_train"], d["y_val"], d["y_test"]]).astype(np.int64)
    print(f"{len(X)} clips, {len(names)} words, device {device}")

    n_runs = args.repeats
    E = args.ensemble
    fold_acc, single_acc, oof_pred, oof_prob, oof_top5, lines = [], [], [], [], [], []
    t0 = time.time()
    for rep in range(n_runs):
        folds = stratified_folds(y, args.folds, args.seed + rep)
        pred = np.full(len(y), -1)
        prob = np.zeros((len(y), len(names)), dtype=np.float32)
        top5 = np.zeros(len(y), dtype=bool)
        for f, test_idx in enumerate(folds):
            if args.smoke and f >= 2:
                break
            rest = np.setdiff1d(np.arange(len(y)), test_idx)
            tr, va = stratified_split(y, rest, 0.12, args.seed + rep * 100 + f)
            probs = []
            for e in range(E):
                seed = args.seed + rep * 100 + f + 1000 * e
                model, ep, bva = train_one_fold(X[tr], y[tr], X[va], y[va], len(names), args, seed, device)
                probs.append(F.softmax(predict(model, X[test_idx], device), dim=1).numpy())
                single_acc.append(float((probs[-1].argmax(1) == y[test_idx]).mean()))
                print(f"  model {e + 1}/{E}: held-out acc {single_acc[-1]:.4f} (val acc {bva:.3f}, epoch {ep}, {time.time() - t0:.0f}s)", flush=True)
            p = np.mean(probs, axis=0)
            prob[test_idx] = p
            pred[test_idx] = p.argmax(1)
            top5[test_idx] = (np.argsort(-p, axis=1)[:, :5] == y[test_idx][:, None]).any(1)
            acc = float((pred[test_idx] == y[test_idx]).mean())
            fold_acc.append(acc)
            print(f"repeat {rep + 1} fold {f + 1}/{args.folds}: held-out acc {acc:.4f} "
                  f"({len(test_idx)} clips, {time.time() - t0:.0f}s)", flush=True)
        oof_pred.append(pred)
        oof_prob.append(prob)
        oof_top5.append(top5)

    done = oof_pred[0] >= 0
    correct = sum(int((p[done] == y[done]).sum()) for p in oof_pred)
    total = int(done.sum()) * n_runs
    lo, hi = wilson(correct, total)
    top5_acc = float(np.mean([t[done].mean() for t in oof_top5]))
    per_class = np.array([np.mean([(p[done & (y == c)] == c).mean() for p in oof_pred]) for c in range(len(names))])
    worst = np.argsort(per_class)[:8]
    allp = np.concatenate([p[done] for p in oof_pred])
    ally = np.tile(y[done], n_runs)

    lines += [f"CROSS-VALIDATION  ({args.folds} folds x {n_runs} repeat(s), aug={args.aug}, epochs<={args.epochs}, "
              f"features={','.join(args.groups) or 'none'}, models per fold={E})",
              f"clips tested: {total}",
              f"top-1 accuracy: {correct / total:.4f}   (95% range {lo:.4f} to {hi:.4f})",
              f"top-5 accuracy: {top5_acc:.4f}",
              *([f"single models (average of {len(single_acc)}): {np.mean(single_acc):.4f}"] if E > 1 else []),
              f"accuracy per fold: " + ", ".join(f"{a:.3f}" for a in fold_acc) + f"   (mean {np.mean(fold_acc):.4f}, sd {np.std(fold_acc):.4f})",
              "", "Hardest words:"]
    lines += [f"  {names[c]:<24} {per_class[c]:.3f}" for c in worst]
    lines += ["", "Most common mistakes (true -> predicted):"]
    lines += [f"  {t} -> {p}   x{n}" for t, p, n in confusion_pairs(ally, allp, names, 10)]
    if args.compare is not None and args.compare.exists():
        base = np.load(args.compare)
        if len(base["y"]) == len(y) and (base["y"] == y).all() and base["pred"].shape[0] >= 1:
            bp, npred = base["pred"][0], oof_pred[0]
            fixed = int(((bp != y) & (npred == y) & done).sum())
            broken = int(((bp == y) & (npred != y) & done).sum())
            lines += ["", f"Compared with {args.compare.parent.name} (same clips, first repeat): "
                          f"{fixed} clips now right that were wrong, {broken} now wrong that were right."]
    text = "\n".join(lines)
    (args.out / "cv_results.txt").write_text(text + "\n", encoding="utf-8")
    np.savez(args.out / "cv_oof.npz", y=y, pred=np.stack(oof_pred), prob=np.stack(oof_prob), top5=np.stack(oof_top5))
    print("\n" + text)


if __name__ == "__main__":
    main()
