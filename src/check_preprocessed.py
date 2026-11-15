"""Is the preprocessing good? Checks the FINAL preprocessed cubes, without any neural network.

Three questions, each answered from the saved volumes in index_proc.csv:

  1. Are the cubes technically sound?
     shape, NaN/Inf, value range, brain fraction, centre of mass, and ORIENTATION: every cube is
     correlated with the group-average brain, and with its own flipped versions; a cube that matches
     the average better when flipped is mis-oriented. Outliers on any measure are listed and drawn.
  2. Does it look right?
     qc_montage.png: three orthogonal slices of random cubes plus the worst outliers.
     class_means.png: average CN brain, average AD brain, and AD - CN. Alzheimer's atrophy should show
     as darker (more CSF) ventricles and sulci in AD, i.e. a structured difference map, not noise.
  3. Is the disease signal still in the cubes?
     a simple classifier on the cubes themselves (voxels at 24^3 -> PCA -> logistic regression, and a
     few hand-made atrophy measures), cross-validated on the SAME subject-level folds as the CNN.
     Simple voxel classifiers are a fair reference: in published comparisons CNNs did not beat them
     for AD vs CN.

Reading the result:
  voxel/feature AUC good (CDR 0 vs >= 0.5 above ~0.75)  -> the preprocessing keeps the signal; the
                                                           problem is the network or its training
  voxel AUC near 0.5, but OASIS nWBV baseline good       -> the preprocessing loses the signal
                                                           (crop/resize, masks, normalisation)
  both near 0.5                                          -> labels/sample, not preprocessing

  python src/check_preprocessed.py --index data/proc/index_proc.csv --splits data/splits.json \
      --key_file <key> --out runs/check_preprocessed

Only dev folds are used for the classifiers; held-out test subjects are excluded.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import get_logger, load_volume

log = get_logger("checkprep")
SMALL = 24  # voxel classifier works on cubes resized to SMALL^3


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_all(df, key):
    vols, small, rows = [], [], []
    for r in df.itertuples():
        v = load_volume(r.npy, key).astype(np.float32)
        finite = np.isfinite(v).all()
        v = np.nan_to_num(v)
        brain = v > 0
        com = np.argwhere(brain).mean(0) / np.array(v.shape) if brain.any() else np.full(3, np.nan)
        inside = v[brain]
        rows.append({"image_id": r.image_id, "shape": "x".join(map(str, v.shape)), "finite": finite,
                     "min": float(v.min()), "max": float(v.max()),
                     "brain_frac": float(brain.mean()),
                     "com_0": com[0], "com_1": com[1], "com_2": com[2],
                     "median_in_brain": float(np.median(inside)) if inside.size else np.nan,
                     # atrophy proxies inside the mask: share of dark (CSF-like) voxels overall and in a
                     # central box around the ventricles
                     "dark_frac": float((inside < 0.25).mean()) if inside.size else np.nan,
                     "central_dark_frac": central_dark(v)})
        vols.append(v)
        small.append(zoom(v, SMALL / v.shape[0], order=1))
    return vols, np.stack(small), pd.DataFrame(rows)


def central_dark(v):
    n = v.shape[0]
    lo, hi = int(n * 0.35), int(n * 0.65)
    box = v[lo:hi, lo:hi, lo:hi]
    inside = box > 0
    if inside.sum() == 0:
        return np.nan
    return float((box[inside] < 0.25).mean())


def orientation(vols):
    """Correlation of each cube with the mean cube, as-is and flipped along each axis."""
    mean = np.mean(vols, 0)
    out = []
    for v in vols:
        c0 = corr(v, mean)
        flips = {ax: corr(np.flip(v, ax), mean) for ax in range(3)}
        best_ax = max(flips, key=flips.get)
        # axis 2 (left-right after RAS reorientation) is nearly symmetric, so only a clear gain counts
        mis = any(flips[ax] > c0 + 0.05 for ax in (0, 1)) or flips[2] > c0 + 0.15
        out.append({"corr_to_mean": c0, "best_flip_axis": best_ax, "best_flip_corr": flips[best_ax],
                    "misoriented": mis})
    return pd.DataFrame(out), mean


def outliers(tab, cols, z=3.5):
    """Robust z-scores (median / MAD); rows beyond z on any column."""
    flags = []
    for c in cols:
        x = tab[c].astype(float)
        mad = np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826 + 1e-12
        zc = (x - np.nanmedian(x)) / mad
        for i in np.where(np.abs(zc) > z)[0]:
            flags.append((tab.image_id.iloc[i], c, round(float(x.iloc[i]), 4), round(float(zc.iloc[i]), 1)))
    return pd.DataFrame(flags, columns=["image_id", "measure", "value", "robust_z"])


def montage(vols, ids, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(vols)
    fig, ax = plt.subplots(n, 3, figsize=(6, 2 * n))
    ax = np.atleast_2d(ax)
    for i, (v, name) in enumerate(zip(vols, ids)):
        c = [s // 2 for s in v.shape]
        for j, sl in enumerate((v[c[0]], v[:, c[1]], v[:, :, c[2]])):
            ax[i, j].imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=1)
            ax[i, j].axis("off")
        ax[i, 0].set_title(name, fontsize=7, loc="left")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def class_means(vols, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    if not {"CN", "AD"} <= set(labels):
        return False
    cn = np.mean([v for v, l in zip(vols, labels) if l == "CN"], 0)
    ad = np.mean([v for v, l in zip(vols, labels) if l == "AD"], 0)
    d = ad - cn
    lim = np.abs(d).max() + 1e-6
    c = [s // 2 for s in cn.shape]
    fig, ax = plt.subplots(3, 3, figsize=(8, 8))
    for i, (img, name, cm, vmin, vmax) in enumerate([(cn, "mean CN", "gray", 0, 1), (ad, "mean AD", "gray", 0, 1),
                                                     (d, "AD - CN (blue = darker in AD)", "bwr", -lim, lim)]):
        for j, sl in enumerate((img[c[0]], img[:, c[1]], img[:, :, c[2]])):
            ax[i, j].imshow(np.rot90(sl), cmap=cm, vmin=vmin, vmax=vmax)
            ax[i, j].axis("off")
        ax[i, 0].set_title(name, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return True


def subject_ci(X, y, subjects, splits, pca, n=300, seed=0):
    """Out-of-fold predictions once, then a subject bootstrap of the AUC."""
    p = np.full(len(y), np.nan)
    for f in splits["folds"]:
        tr, va = np.isin(subjects, f["train"]), np.isin(subjects, f["val"])
        if len(np.unique(y[tr])) < 2 or va.sum() == 0:
            continue
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=min(pca, tr.sum() - 1), random_state=0))
        steps.append(LogisticRegression(C=0.1 if pca else 1.0, class_weight="balanced", max_iter=5000))
        p[va] = make_pipeline(*steps).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    ok = ~np.isnan(p)
    yy, pp, ss = y[ok], p[ok], subjects[ok]
    rng, uniq = np.random.default_rng(seed), np.unique(ss)
    idx = {s: np.where(ss == s)[0] for s in uniq}
    vals = []
    for _ in range(n):
        t = np.concatenate([idx[s] for s in rng.choice(uniq, len(uniq))])
        if len(np.unique(yy[t])) == 2:
            vals.append(roc_auc_score(yy[t], pp[t]))
    if len(np.unique(yy)) < 2:
        return np.nan, (np.nan, np.nan)
    return roc_auc_score(yy, pp), (np.percentile(vals, 2.5), np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--key_file", default=None)
    ap.add_argument("--out", default="runs/check_preprocessed")
    ap.add_argument("--n_montage", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    key = None
    if a.key_file:
        from secure_io import load_key
        key = load_key(a.key_file)

    df = pd.read_csv(a.index)
    splits = json.load(open(a.splits))
    test = set(splits.get("test", []))
    log.info("loading %d cubes ...", len(df))
    vols, small, tab = load_all(df, key)
    ori, mean_cube = orientation(vols)
    tab = pd.concat([tab, ori], axis=1)
    tab.insert(1, "subject", df.subject.values)
    tab.insert(2, "label", df.label.values)
    for c in ("mask_ml", "etiv", "nwbv", "age"):
        if c in df:
            tab[c] = pd.to_numeric(df[c], errors="coerce").values
    tab.to_csv(os.path.join(a.out, "per_scan.csv"), index=False)

    # ---- 1. technical checks
    lines = ["# Preprocessing check", "", f"{len(tab)} cubes from {tab.subject.nunique()} subjects", "",
             "## 1. Technical checks", ""]
    shapes = tab["shape"].value_counts().to_dict()
    lines += [f"- shapes: {shapes}",
              f"- non-finite values: {int((~tab.finite).sum())} cubes",
              f"- value range: min {tab['min'].min():.3f}, max {tab['max'].max():.3f} (expected 0-1)",
              f"- brain fraction of the cube: median {tab.brain_frac.median():.3f} "
              f"(range {tab.brain_frac.min():.3f}-{tab.brain_frac.max():.3f})",
              f"- correlation with the average brain: median {tab.corr_to_mean.median():.3f}, "
              f"lowest {tab.corr_to_mean.min():.3f}",
              f"- possibly mis-oriented (match the average better when flipped): "
              f"{int(tab.misoriented.sum())} cubes", ""]
    # dark_frac is left out on purpose: strong atrophy (real AD) is an outlier there, not an error
    out = outliers(tab, ["brain_frac", "com_0", "com_1", "com_2", "median_in_brain", "corr_to_mean"])
    if len(tab.misoriented) and tab.misoriented.any():
        mis = tab[tab.misoriented][["image_id"]].assign(measure="flip", value=np.nan, robust_z=np.nan)
        out = pd.concat([out, mis], ignore_index=True)
    out.insert(1, "label", out.image_id.map(dict(zip(tab.image_id, tab.label))))
    if len(out):
        try:
            lines += ["Outliers (robust z > 3.5) and mis-oriented cubes:", "", out.to_markdown(index=False), ""]
        except ImportError:
            lines += ["Outliers:", "", "```", out.to_string(index=False), "```", ""]
    else:
        lines += ["No outliers.", ""]
    out.to_csv(os.path.join(a.out, "outliers.csv"), index=False)

    if tab.mask_ml.notna().any() if "mask_ml" in tab else False:
        if "nwbv" in tab and "etiv" in tab and tab.nwbv.notna().sum() > 5:
            r = tab[["mask_ml", "nwbv", "etiv"]].dropna()
            rr = np.corrcoef(r.mask_ml, r.nwbv * r.etiv)[0, 1]
            lines += [f"- our brain-mask volume vs OASIS brain volume (nWBV x eTIV): r = {rr:.3f} "
                      "(above ~0.8 = our skull stripping agrees with OASIS's)", ""]

    # ---- 2. pictures
    rng = np.random.default_rng(0)
    pick = list(rng.choice(len(vols), min(a.n_montage, len(vols)), replace=False))
    montage([vols[i] for i in pick], [tab.image_id[i] + " " + tab.label[i] for i in pick],
            os.path.join(a.out, "qc_montage.png"), "random cubes (axial / coronal / sagittal through the centre)")
    worst = list(dict.fromkeys(out.image_id.tolist()))[:12]
    if worst:
        idx = [int(np.where(tab.image_id == w)[0][0]) for w in worst]
        montage([vols[i] for i in idx], [tab.image_id[i] + " " + tab.label[i] for i in idx],
                os.path.join(a.out, "qc_outliers.png"), "outlier cubes")
    dev = ~tab.subject.isin(test).values
    has_means = class_means([v for v, d in zip(vols, dev) if d], tab.label[dev].values,
                            os.path.join(a.out, "class_means.png"))
    lines += ["## 2. Pictures", "", "- `qc_montage.png`: random cubes; all should share pose, size and contrast",
              "- `qc_outliers.png`: the outliers above" if worst else "- no outlier picture (no outliers)",
              "- `class_means.png`: average CN, average AD and AD - CN; atrophy = blue (darker) ventricles "
              "and sulci in a structured, symmetric pattern" if has_means else "- class means skipped (no CN/AD)", ""]

    # ---- 3. is the signal there? (dev folds only)
    d = tab[dev].reset_index(drop=True)
    Xs = small[dev].reshape(dev.sum(), -1)
    subj = d.subject.values
    feats = ["dark_frac", "central_dark_frac", "brain_frac"]
    if "mask_ml" in d and "etiv" in d and d.etiv.notna().any():
        d["vol_ratio"] = d.mask_ml / d.etiv
        feats.append("vol_ratio")
    rows = []
    for task, y_map in [("CDR 0 vs >=0.5", {"CN": 0, "MCI": 1, "AD": 1}), ("CN vs AD", {"CN": 0, "AD": 1})]:
        keep = d.label.isin(y_map).values
        y = d.label[keep].map(y_map).values
        if len(np.unique(y)) < 2:
            continue
        sets = [(f"voxels {SMALL}^3 -> PCA 50 -> logistic", Xs[keep], 50),
                ("atrophy measures from the cubes (" + ", ".join(feats) + ")",
                 d.loc[keep, feats].fillna(d[feats].median()).values, None)]
        if "age" in d and d.age.notna().any():
            sets.append(("age only (confound reference)", d.loc[keep, ["age"]].fillna(d.age.median()).values, None))
        for name, X, pca in sets:
            auc, ci = subject_ci(X, y, subj[keep], splits, pca)
            rows.append({"task": task, "input": name, "AUC": round(auc, 3),
                         "95% CI (subjects)": f"{ci[0]:.2f}-{ci[1]:.2f}"})
    sig = pd.DataFrame(rows)
    sig.to_csv(os.path.join(a.out, "signal.csv"), index=False)
    try:
        sig_md = sig.to_markdown(index=False)
    except ImportError:
        sig_md = "```\n" + sig.to_string(index=False) + "\n```"
    lines += ["## 3. Is the disease signal in the cubes? (same subject-level folds as the CNN, dev only)", "",
              sig_md, "",
              "Simple classifiers on the cubes are the floor the CNN must beat. If they reach a useful AUC "
              "(CDR 0 vs >= 0.5 above ~0.75) and the CNN does not, the preprocessing is fine and the network "
              "or its training is the problem. If they sit near 0.5 while OASIS's own nWBV separates the "
              "classes (diagnose_labels.py), the preprocessing loses the signal.", ""]
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {a.out}/report.md, per_scan.csv, outliers.csv, signal.csv and PNGs")


if __name__ == "__main__":
    main()
