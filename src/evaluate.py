"""Stage 5 - Held-out test evaluation, fold ensemble, bootstrap CI, 3D Grad-CAM,
and latent trajectory "imagination" for each test brain.

  python src/evaluate.py --ckpts "runs/cls/fold*/best.pt" --out runs/test --gradcam 4
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from common import TASKS, add_extra_columns, extra_matrix, get_device, get_logger, load_volume, task_frame
from data import ScanDataset, load_split
from finetune_classify import metrics, predict
from models import BrainWorldModel, WorldModelClassifier


def ckpt_task(ck):
    """Task info stored in a classifier checkpoint (older checkpoints = 3-class, no extra inputs)."""
    task = ck.get("task", "3class")
    return task, ck.get("classes", TASKS[task]["classes"]), ck.get("extra_feats", []), ck.get("extra_stats", {})


def load_classifier(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    task, classes, feats, _ = ckpt_task(ck)
    m = WorldModelClassifier(BrainWorldModel(**ck["cfg"]), len(classes), ck["imagine"], n_extra=len(feats),
                             pool_mode=ck.get("pool_mode", "attn"), head_norm=ck.get("head_norm", "layer"))
    m.load_state_dict(ck["state"], strict=False)
    m.use_meta = ck.get("use_meta", True)
    m.task, m.classes, m.extra_feats, m.extra_stats = task, classes, feats, ck.get("extra_stats", {})
    return m.to(dev).eval()


def subject_bootstrap(df, y, pred, n=1000, seed=0):
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    rng = np.random.default_rng(seed)
    subs = df.subject.values
    uniq = np.unique(subs)
    idx_by = {s: np.where(subs == s)[0] for s in uniq}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx_by[s] for s in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[pick])) > 1:
            vals.append(balanced_accuracy_score(y[pick], pred[pick]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def gradcam(model, x, demo, cls):
    model.zero_grad()
    with torch.enable_grad():
        for p in model.encoder.parameters():
            p.requires_grad = True
        logits = model(x.requires_grad_(True), demo)
        logits[0, cls].backward()
    fmap = model._fmap
    w = fmap.grad.mean((2, 3, 4), keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def save_overlay(vol, cam, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = [s // 2 for s in vol.shape]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for a, (v, h) in zip(ax, [(vol[c[0]], cam[c[0]]), (vol[:, c[1]], cam[:, c[1]]),
                              (vol[:, :, c[2]], cam[:, :, c[2]])]):
        a.imshow(np.flipud(v), cmap="gray")
        a.imshow(np.flipud(h), cmap="jet", alpha=0.35)
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--ckpts", default="runs/cls/fold*/best.pt")
    ap.add_argument("--out", default="runs/test")
    ap.add_argument("--gradcam", type=int, default=3, help="Grad-CAM maps per class")
    ap.add_argument("--key_file", default=None)
    a = ap.parse_args()
    log = get_logger("evaluate", a.out)
    dev = get_device()
    key = None
    if a.key_file:
        from secure_io import load_key
        key = load_key(a.key_file)

    te = load_split(a.index, json.load(open(a.splits)), part="test")
    paths = sorted(glob.glob(a.ckpts))
    assert paths, f"no checkpoints match {a.ckpts}"
    models = [load_classifier(p, dev) for p in paths]
    task, LABELS = models[0].task, models[0].classes
    if any(m.task != task or m.extra_feats != models[0].extra_feats for m in models):
        raise SystemExit("all checkpoints in the ensemble must share the same --task and --extra_feats")
    te, y = task_frame(add_extra_columns(te), task)
    log.info("test (%s %s): %d scans / %d subjects | ensemble of %d fold models", task, LABELS, len(te),
             te.subject.nunique(), len(paths))
    probs = []
    for m in models:  # each fold model z-scores the extra inputs with its own training statistics
        ds = ScanDataset(te, None, key, TASKS[task]["map"], extra_matrix(te, m.extra_feats, m.extra_stats))
        probs.append(predict(m, DataLoader(ds, 4), dev))
    probs = np.mean(probs, 0)
    pred = probs.argmax(1)
    dl = DataLoader(ScanDataset(te, None, key, TASKS[task]["map"],
                                extra_matrix(te, models[0].extra_feats, models[0].extra_stats)), 4)

    rep = {"task": task, "classes": LABELS, "scan_level": metrics(y, probs), "n_scans": len(te), "n_subjects": int(te.subject.nunique()),
           "bal_acc_95ci_subject_bootstrap": subject_bootstrap(te, y, pred)}
    if te.groupby("subject").label.nunique().max() == 1:  # labels constant per subject
        sub = te.assign(**{f"p{i}": probs[:, i] for i in range(len(LABELS))}).groupby("subject")
        sp = sub[[f"p{i}" for i in range(len(LABELS))]].mean().values
        sy = sub.label.first().map(TASKS[task]["map"]).values
        rep["subject_level"] = metrics(sy, sp)
    cm = confusion_matrix(y, pred, labels=range(len(LABELS)))
    rep["confusion_matrix(rows=true)"] = {LABELS[i]: cm[i].tolist() for i in range(len(LABELS))}
    json.dump(rep, open(os.path.join(a.out, "test_report.json"), "w"), indent=2)
    out = te[["image_id", "subject", "label"]].copy()
    out["pred"] = [LABELS[i] for i in pred]
    for i, l in enumerate(LABELS):
        out[f"p_{l}"] = probs[:, i]

    # --- imagination: roll the world model forward 1..3 years for each test brain
    m0 = models[0]
    drift = []
    with torch.no_grad():
        for b in dl:
            x, d = b["x"].to(dev), b["demo"].to(dev)
            tok, grid = BrainWorldModel.to_tokens(m0.encoder(x))
            g0 = BrainWorldModel.target_space(tok).mean(1)
            row = []
            for yr in (1.0, 2.0, 3.0):
                act = torch.cat([torch.full_like(d[:, :1], yr / 2.0), d[:, :2]], 1)
                _, mu, logvar = m0.dynamics(tok, act, grid)
                row.append(torch.stack([(mu - g0).norm(dim=1), logvar.exp().mean(1)], 1))
            drift.append(torch.cat(row, 1).cpu())
    drift = torch.cat(drift).numpy()
    for k, yr in enumerate((1, 2, 3)):
        out[f"latent_drift_{yr}y"] = drift[:, 2 * k]
        out[f"uncertainty_{yr}y"] = drift[:, 2 * k + 1]
    out.to_csv(os.path.join(a.out, "test_predictions.csv"), index=False)
    log.info("mean predicted 2-year latent drift by true class: %s",
             out.groupby("label").latent_drift_2y.mean().round(4).to_dict())

    log.info("TEST %s", json.dumps(rep, indent=1))

    if a.gradcam:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            log.warning("matplotlib not installed - skipping Grad-CAM pngs")
            return
        os.makedirs(os.path.join(a.out, "gradcam"), exist_ok=True)
        ds = dl.dataset
        for c, lab in enumerate(LABELS):
            for i in np.where(y == c)[0][: a.gradcam]:
                s = ds[i]
                x, d = s["x"][None].to(dev), s["demo"][None].to(dev)
                cam = gradcam(m0, x, d, int(pred[i]))
                np.save(os.path.join(a.out, "gradcam", f"{te.image_id[i]}.npy"), cam.astype(np.float16))
                save_overlay(load_volume(te.npy[i], key).astype(np.float32), cam,
                             os.path.join(a.out, "gradcam", f"{lab}_{te.image_id[i]}.png"),
                             f"true {lab} | pred {LABELS[pred[i]]} ({probs[i].max():.2f})")
        log.info("Grad-CAM maps saved to %s", os.path.join(a.out, "gradcam"))


if __name__ == "__main__":
    main()
