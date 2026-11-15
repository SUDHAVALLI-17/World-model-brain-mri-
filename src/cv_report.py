"""Pool cross-validation predictions into one honest number per experiment.

Each run of finetune_classify.py writes <run>/val_predictions.csv (out-of-fold predictions for one
fold and seed). This script, per experiment:
  1. per seed, concatenates its folds -> every dev subject predicted exactly once -> pooled metrics
  2. reports mean and sd of those metrics across seeds (seed-to-seed noise)
  3. averages the probabilities over seeds per scan and gives a 95% CI that resamples SUBJECTS
     (a person's visits are correlated; resampling scans would make the CI far too narrow)
  4. with --ref, a paired subject-bootstrap of the AUC difference against a reference experiment on the
     same scans: the honest test of "does the world model help?" or "does the fixed crop help?"

  python src/cv_report.py --exp scratch="runs/cv/cdr_binary/scratch/seed*/fold*" \
                          --exp wm_ft="runs/cv/cdr_binary/wm_ft/seed*/fold*" \
                          --ref scratch --out runs/cv/cdr_binary/report

Never use test-set predictions here; this is for model development on the dev folds only.
"""
import argparse
import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def md(frame):
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def load_exp(pattern):
    runs = sorted(glob.glob(pattern))
    frames = []
    for r in runs:
        f = os.path.join(r, "val_predictions.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        m = re.search(r"seed(\d+)", r)
        d["seed"] = int(m.group(1)) if m else 0
        d["run"] = r
        frames.append(d)
    if not frames:
        raise SystemExit(f"no val_predictions.csv under {pattern}")
    df = pd.concat(frames, ignore_index=True)
    pcols = [c for c in df.columns if c.startswith("p_")]
    if "y" not in df:  # older runs (3-class only, no y column)
        df["y"] = df.label.map({"CN": 0, "MCI": 1, "AD": 2})
    return df, pcols


def auc(y, p):
    k = p.shape[1]
    if len(np.unique(y)) < k:
        return np.nan
    return roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro")


def cn_vs_ad_auc(y, p):
    if p.shape[1] != 3 or not {0, 2}.issubset(set(y)):
        return np.nan
    m = np.isin(y, [0, 2])
    return roc_auc_score(y[m] == 2, p[m][:, 2] / (p[m][:, 0] + p[m][:, 2] + 1e-8))


def scores(y, p):
    pred = p.argmax(1)
    out = {"bal_acc": balanced_accuracy_score(y, pred), "auc": auc(y, p)}
    if p.shape[1] == 3:
        out["auc_cn_vs_ad"] = cn_vs_ad_auc(y, p)
    for c in range(p.shape[1]):
        out[f"recall_{c}"] = float((pred[y == c] == c).mean()) if (y == c).any() else np.nan
    return out


def bootstrap(subj, y, p, n, seed, extra=None):
    """Subject bootstrap. extra=(y2, p2) on the SAME rows -> also the paired AUC difference p - p2."""
    rng, uniq = np.random.default_rng(seed), np.unique(subj)
    idx = {s: np.where(subj == s)[0] for s in uniq}
    res = {"bal_acc": [], "auc": [], "auc_cn_vs_ad": [], "d_auc": []}
    for _ in range(n):
        t = np.concatenate([idx[s] for s in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[t])) < p.shape[1]:
            continue
        sc = scores(y[t], p[t])
        for k in ("bal_acc", "auc", "auc_cn_vs_ad"):
            if k in sc:
                res[k].append(sc[k])
        if extra is not None:
            res["d_auc"].append(sc["auc"] - auc(y[t], extra[t]))
    ci = lambda v: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if len(v) else (np.nan, np.nan)
    return {k: ci(v) for k, v in res.items()}, res["d_auc"]


def seed_mean_probs(df, pcols):
    g = df.groupby("image_id")
    m = g[pcols].mean()
    first = g[["subject", "y", "label"]].first()
    return first.join(m).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", action="append", required=True, help='name="glob of run dirs"')
    ap.add_argument("--ref", default=None, help="experiment name to compare every other one against")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/cv/report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    exps = {}
    for e in a.exp:
        name, pat = e.split("=", 1)
        exps[name] = load_exp(pat.strip('"'))

    pooled = {n: seed_mean_probs(df, pc) for n, (df, pc) in exps.items()}
    rows, details = [], {}
    for name, (df, pcols) in exps.items():
        classes = [c[2:] for c in pcols]
        per_seed = []
        for s, d in df.groupby("seed"):
            dup = d.image_id.duplicated().sum()
            if dup:
                print(f"WARNING {name} seed {s}: {dup} scans predicted twice - overlapping folds?")
            per_seed.append(scores(d.y.values, d[pcols].values))
        ps = pd.DataFrame(per_seed)
        pm = pooled[name]
        y, p, subj = pm.y.values.astype(int), pm[pcols].values, pm.subject.values
        ref_p = None
        if a.ref and a.ref != name and a.ref in pooled:
            r = pooled[a.ref].set_index("image_id").reindex(pm.image_id)
            if r[pcols].isna().any().any() or list(pooled[a.ref].columns) != list(pm.columns):
                print(f"WARNING {name} vs {a.ref}: different scans or classes - no paired test")
            else:
                ref_p = r[pcols].values
        ci, d_auc = bootstrap(subj, y, p, a.n_boot, 0, ref_p)
        sc = scores(y, p)
        row = {"experiment": name, "seeds": df.seed.nunique(), "runs": df.run.nunique(),
               "scans": len(pm), "subjects": pm.subject.nunique(),
               "bal_acc": f"{sc['bal_acc']:.3f} [{ci['bal_acc'][0]:.3f}, {ci['bal_acc'][1]:.3f}]",
               "bal_acc seed sd": round(ps.bal_acc.std(ddof=1), 3) if len(ps) > 1 else np.nan,
               "auc": f"{sc['auc']:.3f} [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}]",
               "auc seed sd": round(ps.auc.std(ddof=1), 3) if len(ps) > 1 else np.nan}
        if len(classes) == 3:
            row["auc CN-AD"] = f"{sc['auc_cn_vs_ad']:.3f} [{ci['auc_cn_vs_ad'][0]:.3f}, {ci['auc_cn_vs_ad'][1]:.3f}]"
        for c, cl in enumerate(classes):
            row[f"recall {cl}"] = round(sc[f"recall_{c}"], 3)
        if ref_p is not None:
            dv = np.array(d_auc)
            row[f"AUC - {a.ref}"] = (f"{sc['auc'] - auc(y, ref_p):+.3f} "
                                     f"[{np.percentile(dv, 2.5):+.3f}, {np.percentile(dv, 97.5):+.3f}]")
            row[f"P(no gain over {a.ref})"] = round(float((dv <= 0).mean()), 3)
        rows.append(row)
        cm = confusion_matrix(y, p.argmax(1), labels=range(len(classes)))
        details[name] = {"classes": classes,
                         "confusion(rows=true, seed-averaged probs)": {cl: cm[i].tolist() for i, cl in enumerate(classes)},
                         "per_seed": ps.round(4).to_dict("records")}

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.out, "cv_report.csv"), index=False)
    json.dump(details, open(os.path.join(a.out, "cv_details.json"), "w"), indent=1)
    lines = ["# Cross-validation report (dev folds only)", "",
          "Metrics on out-of-fold predictions, probabilities averaged over seeds; "
          "[95% CI] resamples subjects. 'seed sd' = spread of the pooled metric across seeds.", "",
          md(tab), ""]
    if a.ref:
        lines += [f"A gain over **{a.ref}** is supported only if its CI excludes 0 "
               f"(P(no gain) < 0.05) and it holds across seeds.", ""]
    open(os.path.join(a.out, "cv_report.md"), "w").write("\n".join(lines))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(tab.to_string(index=False))
    print(f"\nwrote {a.out}/cv_report.md, cv_report.csv, cv_details.json")


if __name__ == "__main__":
    main()
