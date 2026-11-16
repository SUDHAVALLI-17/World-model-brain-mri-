"""Frozen-encoder probe with patient-level cross-validation: an honest number without fine-tuning.

Diagnostics on OASIS-2 found that the pretrained encoder's TOKENS carry diagnostic information
(training-only probe, CN vs AD AUC: attention pool 0.61, mean 0.87, mean+max 0.88, all tokens 0.92)
while fine-tuning collapsed the features to one direction (effective rank ~1) and scored at chance.
This script freezes the encoder, extracts features ONCE for every development scan, and fits a
standardised logistic regression per fold, on the SAME subject-level folds as the CNN:

  features   attn | mean | meanmax | tokens (all 216 x E, PCA inside each training fold)
             optionally + tissue_ratio / age / sex from the index (--add)
  tasks      cdr_binary (CDR 0 vs >= 0.5), cn_vs_ad, 3class
  model      StandardScaler -> [PCA] -> LogisticRegression(class_weight=balanced),
             C chosen by an inner subject-grouped CV on the training fold only
  report     pooled out-of-fold AUC and balanced accuracy, 95% CI resampling SUBJECTS, the spread over
             folds, and a paired AUC difference against --ref (a baseline row, e.g. tissue_ratio)

Encoders (--ckpt name=path, repeatable): a world-model checkpoint (pretrain .../last.pt) or a classifier
best.pt. A path containing {fold} is formatted per fold (fold-specific pretraining, see
pretrain_world_model.py --exclude_fold), which keeps every validation subject unseen by its encoder.
Without it, the encoder was pretrained (without labels) on other folds' validation subjects: say so.

  python src/probe_cv.py --index data/proc/index_proc.csv --splits data/splits.json --key_file <key> \
      --ckpt world_model=runs/pretrain/last.pt --task cdr_binary --add tissue_ratio --out runs/probe_cv

The held-out test subjects are never loaded.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from common import TASKS, add_extra_columns, get_device, get_logger, set_seed
from data import ScanDataset
from models import BrainWorldModel, effective_rank

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
log = get_logger("probe")
POOLS = ("attn", "mean", "meanmax", "tokens")


def load_encoder(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    wm = BrainWorldModel(**ck["cfg"])
    if "model" in ck:
        wm.load_state_dict(ck["model"], strict=False)
    else:
        st = ck["state"]
        wm.encoder.load_state_dict({k[8:]: v for k, v in st.items() if k.startswith("encoder.")})
        wm.pool.load_state_dict({k[5:]: v for k, v in st.items() if k.startswith("pool.")})
    return wm.eval()


@torch.no_grad()
def extract(wm, df, key, dev, bs):
    """pooled features for every row of df: dict pool -> (n, d) float32 array."""
    wm.to(dev)
    out = {p: [] for p in POOLS}
    for b in DataLoader(ScanDataset(df, None, key), bs, num_workers=0):
        tok, _ = BrainWorldModel.to_tokens(wm.encoder(b["x"].to(dev)))
        tok = tok.float()
        out["attn"].append(wm.pool(tok).cpu())
        out["mean"].append(tok.mean(1).cpu())
        out["meanmax"].append(torch.cat([tok.mean(1), tok.amax(1)], 1).cpu())
        out["tokens"].append(tok.flatten(1).cpu())
    wm.cpu()
    return {p: torch.cat(v).numpy().astype(np.float32) for p, v in out.items()}


def fit_predict(Xtr, ytr, gtr, Xva, n_cls, pca, seed):
    """Inner subject-grouped CV picks C on the training fold; returns validation probabilities."""
    def pipe(C, n):
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=max(1, min(pca, n - 2)), random_state=seed))
        steps.append(LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
        return make_pipeline(*steps)

    grid, best_c, best = (0.001, 0.01, 0.1, 1.0), 0.01, -np.inf
    n_inner = min(4, len(np.unique(gtr)))
    if n_inner >= 3 and min(np.bincount(ytr, minlength=n_cls)) >= 3:
        for C in grid:
            p = np.zeros((len(ytr), n_cls))
            ok = True
            for itr, iva in GroupKFold(n_inner).split(Xtr, ytr, gtr):
                if len(np.unique(ytr[itr])) < n_cls:
                    ok = False
                    break
                p[iva] = pipe(C, len(itr)).fit(Xtr[itr], ytr[itr]).predict_proba(Xtr[iva])
            if not ok:
                continue
            sc = balanced_accuracy_score(ytr, p.argmax(1))
            if sc > best:
                best, best_c = sc, C
    return pipe(best_c, len(ytr)).fit(Xtr, ytr).predict_proba(Xva), best_c


def auc_of(y, p):
    k = p.shape[1]
    if len(np.unique(y)) < k:
        return np.nan
    return roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro")


def boot(y, p, subj, n, seed, ref=None):
    rng, uniq = np.random.default_rng(seed), np.unique(subj)
    idx = {s: np.where(subj == s)[0] for s in uniq}
    a, b, d = [], [], []
    for _ in range(n):
        t = np.concatenate([idx[s] for s in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[t])) < p.shape[1]:
            continue
        a.append(auc_of(y[t], p[t]))
        b.append(balanced_accuracy_score(y[t], p[t].argmax(1)))
        if ref is not None:
            d.append(a[-1] - auc_of(y[t], ref[t]))
    q = lambda v: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if len(v) else (np.nan, np.nan)
    return q(a), q(b), np.array(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--key_file", default=None)
    ap.add_argument("--ckpt", action="append", default=[], help="name=path ({fold} allowed in path)")
    ap.add_argument("--no_random", action="store_true", help="skip the random-init encoder reference")
    ap.add_argument("--task", default="cdr_binary", choices=list(TASKS))
    ap.add_argument("--pools", nargs="+", default=list(POOLS), choices=POOLS)
    ap.add_argument("--add", nargs="*", default=[], help="index columns appended to the image features "
                    "and also scored alone, e.g. tissue_ratio age")
    ap.add_argument("--pca", type=int, default=64, help="PCA size for 'tokens' (fit inside each training fold)")
    ap.add_argument("--ref", default=None, help="row name for the paired AUC difference (default: first --add alone)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/probe_cv")
    a = ap.parse_args()
    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = get_device()
    key = None
    if a.key_file:
        from secure_io import load_key
        key = load_key(a.key_file)

    splits = json.load(open(a.splits))
    folds = splits["folds"]
    dev_subj = set(s for f in folds for s in f["train"] + f["val"])
    df = add_extra_columns(pd.read_csv(a.index))
    df = df[df.subject.isin(dev_subj) & df.label.isin(TASKS[a.task]["map"])].reset_index(drop=True)
    if "sex" in df:
        df["sex_m"] = (df.sex.astype(str).str.upper().str[0] == "M").astype(float)
    add = ["sex_m" if c == "sex" else c for c in a.add]
    for c in add:
        if c not in df or df[c].notna().sum() == 0:
            raise SystemExit(f"--add {c}: column missing (tissue_ratio needs add_tissue_volume.py)")
    y = df.label.map(TASKS[a.task]["map"]).values.astype(int)
    subj = df.subject.values
    n_cls = len(TASKS[a.task]["classes"])
    log.info("task %s: %d dev scans / %d subjects %s | %d folds", a.task, len(df), df.subject.nunique(),
             df.label.value_counts().to_dict(), len(folds))
    fold_of = {s: k for k, f in enumerate(folds) for s in f["val"]}
    row_fold = np.array([fold_of[s] for s in subj])

    encoders = [tuple(c.split("=", 1)) for c in a.ckpt]
    if not a.no_random:
        cfg = torch.load(encoders[0][1].format(fold=0), map_location="cpu", weights_only=False)["cfg"] \
            if encoders else {}
        encoders = [("random", cfg)] + encoders

    # feature sets: name -> per-fold feature matrix getter
    rows, preds, ranks = [], {}, []
    extra = df[add].apply(pd.to_numeric, errors="coerce") if add else None
    if add:
        extra = extra.fillna(extra.median()).values.astype(np.float32)

    def run(name, getX, pca):
        p = np.full((len(y), n_cls), np.nan)
        per_fold, cs = [], []
        for k in range(len(folds)):
            tr, va = row_fold != k, row_fold == k
            if len(np.unique(y[tr])) < n_cls or va.sum() == 0:
                continue
            X = getX(k)
            p[va], c = fit_predict(X[tr], y[tr], subj[tr], X[va], n_cls, pca, a.seed)
            cs.append(c)
            per_fold.append(auc_of(y[va], p[va]))
        preds[name] = p
        ok = ~np.isnan(p).any(1)
        return p, ok, per_fold, cs

    if add:  # non-imaging reference rows
        for i, c in enumerate(add):
            run(f"{c} alone", lambda k, i=i: extra[:, [i]], None)
        if len(add) > 1:
            run("+".join(add), lambda k: extra, None)

    for name, src in encoders:
        per_fold_feats = {}
        if isinstance(src, dict):  # random init, same for all folds
            torch.manual_seed(a.seed)
            f = extract(BrainWorldModel(**src).eval(), df, key, dev, a.bs) if src else \
                extract(BrainWorldModel().eval(), df, key, dev, a.bs)
            per_fold_feats = {k: f for k in range(len(folds))}
        elif "{fold}" in src:
            for k in range(len(folds)):
                per_fold_feats[k] = extract(load_encoder(src.format(fold=k)), df, key, dev, a.bs)
        else:
            f = extract(load_encoder(src), df, key, dev, a.bs)
            per_fold_feats = {k: f for k in range(len(folds))}
        f0 = per_fold_feats[0]
        for pool in ("attn", "mean", "meanmax"):
            ranks.append({"encoder": name, "pool": pool, "effective_rank": round(effective_rank(f0[pool]), 2),
                          "dims": f0[pool].shape[1]})
        for pool in a.pools:
            pca = a.pca if pool == "tokens" else None
            run(f"{name} | {pool}", lambda k, pool=pool: per_fold_feats[k][pool], pca)
            if add:
                run(f"{name} | {pool} + {'+'.join(add)}",
                    lambda k, pool=pool: np.concatenate([per_fold_feats[k][pool], extra], 1), pca)

    ref = a.ref or (f"{add[0]} alone" if add else None)
    for name, p in preds.items():
        ok = ~np.isnan(p).any(1)
        yy, pp, ss = y[ok], p[ok], subj[ok]
        rp = preds[ref][ok] if ref in preds and name != ref else None
        ci_a, ci_b, d = boot(yy, pp, ss, a.n_boot, a.seed, rp)
        aucs = [auc_of(y[ok & (row_fold == k)], p[ok & (row_fold == k)]) for k in range(len(folds))]
        row = {"features": name, "scans": int(ok.sum()),
               "AUC [95% CI]": f"{auc_of(yy, pp):.3f} [{ci_a[0]:.3f}, {ci_a[1]:.3f}]",
               "bal acc [95% CI]": f"{balanced_accuracy_score(yy, pp.argmax(1)):.3f} [{ci_b[0]:.3f}, {ci_b[1]:.3f}]",
               "AUC fold sd": round(float(np.nanstd(aucs)), 3)}
        if rp is not None and len(d):
            row[f"AUC - {ref}"] = (f"{auc_of(yy, pp) - auc_of(yy, rp):+.3f} "
                                   f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
        rows.append(row)
        out = df.loc[ok, ["image_id", "subject", "label"]].copy()
        out["y"] = yy
        for c, cl in enumerate(TASKS[a.task]["classes"]):
            out[f"p_{cl}"] = pp[:, c]
        safe = name.replace(" | ", "__").replace(" ", "_").replace("+", "_").replace("/", "_")
        out.to_csv(os.path.join(a.out, f"oof_{safe}.csv"), index=False)

    tab, rk = pd.DataFrame(rows), pd.DataFrame(ranks)
    tab.to_csv(os.path.join(a.out, "probe_cv.csv"), index=False)
    md = lambda t: t.to_markdown(index=False)
    exposure = "fold-specific encoders ({fold} paths): every validation subject unseen by its encoder" \
        if any("{fold}" in str(s) for _, s in encoders) else \
        ("single encoder: if it was pretrained on all development subjects, validation subjects were seen "
         "WITHOUT labels during pretraining; use --exclude_fold pretraining for a fully clean estimate")
    open(os.path.join(a.out, "probe_cv.md"), "w").write(
        f"# Frozen-encoder probe, task {a.task}, {len(folds)}-fold patient-level CV\n\n"
        f"{len(df)} development scans, {df.subject.nunique()} subjects; test subjects never loaded.\n\n"
        f"Encoder exposure: {exposure}.\n\n{md(tab)}\n\n## Effective rank of the features (fold-0 encoder)\n\n"
        f"{md(rk)}\n\nAbout 1 = dimensional collapse (every scan on one line).\n")
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(tab.to_string(index=False))
    print()
    print(rk.to_string(index=False))
    print(f"\nwrote {a.out}/probe_cv.md, probe_cv.csv and oof_*.csv")


if __name__ == "__main__":
    main()
