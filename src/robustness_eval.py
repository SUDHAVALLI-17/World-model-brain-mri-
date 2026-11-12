"""Stage 6 - Robustness & bias evaluation on the held-out test set.

Compares one or more trained runs (e.g. standard vs adversarially trained):
  * white-box attacks (FGSM, PGD-k) at several eps -> robust balanced accuracy curve
  * natural corruptions MRI actually suffers (noise, bias field, contrast, blur, motion-like rotation)
  * site probe: how well a linear model recovers the acquisition site from the
    classifier's features (lower = less scanner shortcut; chance = 1 / #sites)
  * a figure of one adversarial example (clean | perturbation | attacked)

  python src/robustness_eval.py \
      --run standard "runs/cls/fold*/best.pt" \
      --run adversarial "runs/cls_adv/fold*/best.pt" \
      --out runs/robustness
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from adversarial import build_attribute_labels, fgsm, pgd
from common import LABEL2IDX, LABELS, get_device, get_logger, set_seed
from data import ScanDataset, load_split
from evaluate import load_classifier
from finetune_classify import metrics


class Ensemble(nn.Module):
    """Averages fold models' probabilities; returns log-probs so attacks can differentiate it."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, demo, return_feats=False):
        p = torch.stack([F.softmax(m(x, demo), 1) for m in self.models]).mean(0)
        return torch.log(p + 1e-8)


def _rotate(x, deg):
    r = math.radians(deg)
    theta = torch.tensor([[math.cos(r), -math.sin(r), 0, 0], [math.sin(r), math.cos(r), 0, 0], [0, 0, 1, 0]],
                         dtype=x.dtype, device=x.device)[None].expand(x.shape[0], -1, -1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


def _bias(x, strength):
    g = torch.Generator().manual_seed(0)
    coarse = torch.randn(x.shape[0], 1, 4, 4, 4, generator=g).to(x.device) * strength
    return x * torch.exp(F.interpolate(coarse, size=x.shape[2:], mode="trilinear", align_corners=True))


CORRUPTIONS = {
    "gaussian_noise_0.03": lambda x: x + 0.03 * torch.randn_like(x) * (x > 0),
    "gaussian_noise_0.06": lambda x: x + 0.06 * torch.randn_like(x) * (x > 0),
    "bias_field": lambda x: _bias(x, 0.3),
    "gamma_0.7": lambda x: x.clamp(min=0) ** 0.7,
    "gamma_1.5": lambda x: x.clamp(min=0) ** 1.5,
    "blur": lambda x: F.avg_pool3d(x, 3, 1, 1),
    "rotate_8deg": lambda x: _rotate(x, 8),
}


def run_eval(model, dl, dev, transform=None, attack=None):
    probs, ys = [], []
    for b in dl:
        x, d, y = b["x"].to(dev), b["demo"].to(dev), b["y"].to(dev)
        if transform:
            x = transform(x)
        if attack:
            kind, eps, steps = attack
            x = fgsm(model, x, d, y, eps) if kind == "fgsm" else pgd(model, x, d, y, eps, steps)
        with torch.no_grad():
            probs.append(F.softmax(model(x, d), 1).cpu())
        ys.append(y.cpu())
    return metrics(torch.cat(ys).numpy(), torch.cat(probs).numpy())


@torch.no_grad()
def site_probe(model, df, key, dev, attr="site"):
    try:
        sites, classes = build_attribute_labels(df, attr)
    except ValueError:
        return None
    if len(classes) < 2:
        return None
    feats = []
    for b in DataLoader(ScanDataset(df, None, key), 4):
        feats.append(model(b["x"].to(dev), b["demo"].to(dev), return_feats=True)[1].cpu())
    feats = torch.cat(feats).numpy()
    k = min(5, df.subject.nunique())
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    try:
        pred = cross_val_predict(clf, feats, sites, cv=GroupKFold(k), groups=df.subject.values)
    except ValueError as e:  # e.g. every scanner belongs to a single subject
        return {"not_estimable": str(e).split(":")[0], "n_groups": len(classes)}
    return {f"{attr}_probe_bal_acc": float(balanced_accuracy_score(sites, pred)),
            "chance": 1 / len(classes), f"n_{attr}_groups": len(classes)}


def adversarial_figure(model, ds, dev, eps, steps, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = ds[0]
    x, d, y = s["x"][None].to(dev), s["demo"][None].to(dev), torch.tensor([s["y"]], device=dev)
    xa = pgd(model, x, d, y, eps, steps)
    with torch.no_grad():
        pc, pa = F.softmax(model(x, d), 1)[0], F.softmax(model(xa, d), 1)[0]
    c = x.shape[2] // 2
    clean, adv = x[0, 0, c].cpu().numpy(), xa[0, 0, c].cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.8))
    ax[0].imshow(np.flipud(clean), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title(f"clean: {LABELS[pc.argmax()]} ({pc.max():.2f})")
    ax[1].imshow(np.flipud(adv - clean), cmap="seismic", vmin=-eps, vmax=eps)
    ax[1].set_title(f"perturbation (eps={eps})")
    ax[2].imshow(np.flipud(adv), cmap="gray", vmin=0, vmax=1)
    ax[2].set_title(f"attacked: {LABELS[pa.argmax()]} ({pa.max():.2f})")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{title} | true {LABELS[s['y']]}")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--run", nargs=2, action="append", metavar=("NAME", "CKPT_GLOB"), required=True)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--no_corruptions", action="store_true")
    ap.add_argument("--no_site_probe", action="store_true")
    ap.add_argument("--out", default="runs/robustness")
    ap.add_argument("--key_file", default=None)
    a = ap.parse_args()
    set_seed(0)
    log = get_logger("robustness", a.out)
    dev = get_device()
    key = None
    if a.key_file:
        from secure_io import load_key
        key = load_key(a.key_file)

    splits = json.load(open(a.splits))
    te = load_split(a.index, splits, part="test")
    dev_df = load_split(a.index, splits, part="dev")
    ds = ScanDataset(te, None, key)
    dl = DataLoader(ds, 4)
    rows, report = [], {}

    for name, pattern in a.run:
        paths = sorted(glob.glob(pattern))
        assert paths, f"no checkpoints for {name}: {pattern}"
        folds = [load_classifier(p, dev) for p in paths]
        ens = Ensemble(folds).to(dev).eval()
        log.info("== %s: ensemble of %d models", name, len(paths))

        res = run_eval(ens, dl, dev)
        rows.append({"run": name, "test": "clean", "eps": 0, **res})
        for eps in a.eps:
            for kind in ("fgsm", "pgd"):
                res = run_eval(ens, dl, dev, attack=(kind, eps, a.steps))
                rows.append({"run": name, "test": f"{kind}{'-' + str(a.steps) if kind == 'pgd' else ''}",
                             "eps": eps, **res})
                log.info("%-12s %-7s eps=%.4f bal_acc=%.3f", name, kind, eps, res["bal_acc"])
        if not a.no_corruptions:
            for cname, fn in CORRUPTIONS.items():
                torch.manual_seed(0)
                res = run_eval(ens, dl, dev, transform=fn)
                rows.append({"run": name, "test": cname, "eps": None, **res})
                log.info("%-12s %-20s bal_acc=%.3f", name, cname, res["bal_acc"])
        if not a.no_site_probe:
            report[name] = {}
            for attr in ("site", "scanner"):
                probe = site_probe(folds[0], dev_df, key, dev, attr)
                report[name][f"{attr}_probe"] = probe
                log.info("%-12s %s probe: %s", name, attr, probe)
        try:
            adversarial_figure(ens, ds, dev, max(a.eps), a.steps,
                               os.path.join(a.out, f"adv_example_{name}.png"), name)
        except ImportError:
            pass

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "robustness.csv"), index=False)
    df["condition"] = [t if e is None or e == 0 or pd.isna(e) else f"{t} eps={e:g}" for t, e in zip(df.test, df.eps)]
    table = df.pivot_table(index="condition", columns="run", values="bal_acc", sort=False)
    log.info("balanced accuracy on test set:\n%s", table.round(3).to_string())
    report["bal_acc_table"] = table.round(4).to_dict()
    json.dump(report, open(os.path.join(a.out, "robustness_report.json"), "w"), indent=2, default=str)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, _ in a.run:
            clean = df[(df.run == name) & (df.test == "clean")].bal_acc.item()
            for kind, ls in (("fgsm", "--"), ("pgd", "-")):
                sub = df[(df.run == name) & df.test.str.startswith(kind)]
                ax.plot([0] + sub.eps.tolist(), [clean] + sub.bal_acc.tolist(), ls, marker="o",
                        label=f"{name} {kind.upper()}")
        ax.axhline(1 / len(LABELS), color="gray", lw=0.8, ls=":", label="chance")
        ax.set_xlabel("eps (L-inf, intensity units)")
        ax.set_ylabel("balanced accuracy")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "robustness_curve.png"), dpi=110)
    except ImportError:
        pass
    log.info("saved to %s", a.out)


if __name__ == "__main__":
    main()
