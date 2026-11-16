"""Label audit + non-imaging baselines on the SAME subject-level folds the CNN uses.

Answers, before any more deep-learning runs:
  1. Are the visit-level labels what we think? (CDR -> CN/MCI/AD counts, OASIS "Group" vs our label,
     subjects whose CDR changes, missing values)
  2. How many AD scans AND AD subjects does each validation fold really contain?
     (7 AD scans can be 2 people; a per-fold AD score is then mostly noise)
  3. Is the signal there at all? Logistic regression on OASIS's own nWBV / eTIV, and on our
     preprocessing's mask_ml, scored on the identical folds. If nWBV separates CN vs AD and the CNN
     does not, the image pipeline (per-brain crop + resize) is throwing the signal away.
  4. Confounds: does age + sex alone predict the label? If so, image results must beat that.

  python src/diagnose_labels.py --index data/proc/index_proc.csv --splits data/splits.json --out runs/diagnose

Uses only dev folds (never the held-out test subjects).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["CN", "MCI", "AD"]
TASKS = {  # name -> (label mapping, classes kept)
    "3class": ({"CN": 0, "MCI": 1, "AD": 2}, None),
    "CN_vs_AD": ({"CN": 0, "AD": 1}, ["CN", "AD"]),
    "CN_vs_impaired": ({"CN": 0, "MCI": 1, "AD": 1}, None),   # CDR 0 vs CDR >= 0.5
}
FEATURES = {
    "age_sex (confound)": ["age", "sex_m"],
    "nWBV (OASIS)": ["nwbv"],
    "nWBV+eTIV+age+sex": ["nwbv", "etiv", "age", "sex_m"],
    "mask_ml/eTIV (ours)": ["our_ratio"],
    "mask_ml/eTIV+age+sex": ["our_ratio", "age", "sex_m"],
    "tissue/eTIV (ours)": ["tissue_ratio"],
    "tissue/eTIV+age+sex": ["tissue_ratio", "age", "sex_m"],
}


def md(frame, index=True):
    """Markdown table (falls back to plain text if `tabulate` is not installed)."""
    try:
        return frame.to_markdown(index=index)
    except ImportError:
        return "```\n" + frame.to_string(index=index) + "\n```"


def load(index):
    df = pd.read_csv(index)
    for c in ("nwbv", "etiv", "cdglobal", "mask_ml", "mmscore", "age", "tissue_ml"):
        if c not in df:
            df[c] = np.nan
    df["sex_m"] = (df.sex.astype(str).str.upper().str[0] == "M").astype(float)
    for c in ("nwbv", "etiv", "cdglobal", "mask_ml", "mmscore"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["our_ratio"] = df.mask_ml / df.etiv
    df["tissue_ratio"] = pd.to_numeric(df.tissue_ml, errors="coerce") / df.etiv
    return df.sort_values(["subject", "acq_date"])


def audit(df, splits, out):
    lines = ["# Label audit", ""]
    lines += ["## Scan-level CDR -> label", "",
              md(df.groupby([df.cdglobal.fillna("missing"), "label"]).size().rename("scans").reset_index(),
                 index=False), ""]
    if "group_raw" in df:
        lines += ["## OASIS Group vs our label (scans)", "",
                  md(pd.crosstab(df.group_raw.fillna("missing"), df.label)), "",
                  "OASIS calls CDR 0.5 'very mild dementia'. If most 'Demented' scans are labelled MCI here, "
                  "the MCI class is mostly very mild AD, and MCI vs AD is a severity split, not a diagnosis.", ""]
    per_subj = df.groupby("subject").label.agg(lambda s: "->".join(dict.fromkeys(s)))
    lines += ["## Label paths per subject (order of first appearance)", "",
              md(per_subj.value_counts().rename("subjects")), ""]
    miss = df[["age", "sex", "nwbv", "etiv", "mask_ml"]].isna().sum()
    lines += ["## Missing values", "", md(miss.rename("missing")), ""]
    if df.mask_ml.notna().any() and df.nwbv.notna().any():
        r = df[["mask_ml", "nwbv", "etiv"]].dropna()
        corr = np.corrcoef(r.mask_ml, r.nwbv * r.etiv)[0, 1] if len(r) > 2 else np.nan
        lines += [f"Correlation of our mask volume with OASIS brain volume (nWBV x eTIV): r = {corr:.2f} "
                  "(expect > 0.8 if skull stripping is consistent).", ""]

    rows = []
    for k, f in enumerate(splits["folds"]):
        for part in ("train", "val"):
            d = df[df.subject.isin(f[part])]
            row = {"fold": k, "part": part, "subjects": d.subject.nunique(), "scans": len(d)}
            for c in CLASSES:
                row[f"{c} scans"] = int((d.label == c).sum())
                row[f"{c} subj"] = d.loc[d.label == c, "subject"].nunique()
            row["age mean"] = round(d.age.mean(), 1)
            row["male %"] = round(100 * d.sex_m.mean())
            rows.append(row)
    fold_tab = pd.DataFrame(rows)
    lines += ["## Class counts per fold (scans and distinct subjects)", "", md(fold_tab, index=False), "",
              "A class with fewer than ~5 subjects in a validation fold gives a per-fold score that one "
              "patient can swing by 20+ points.", ""]
    open(os.path.join(out, "label_audit.md"), "w").write("\n".join(lines))
    fold_tab.to_csv(os.path.join(out, "fold_counts.csv"), index=False)
    return fold_tab


def score(y, p, n_cls):
    pred = p.argmax(1)
    out = {"bal_acc": balanced_accuracy_score(y, pred)}
    try:
        out["auc"] = roc_auc_score(y, p[:, 1]) if n_cls == 2 else \
            roc_auc_score(y, p, multi_class="ovr", average="macro")
    except ValueError:
        out["auc"] = np.nan
    return out


def subject_bootstrap(y, p, subj, n_cls, n=1000, seed=0):
    """95% CI resampling SUBJECTS (their scans are correlated), on pooled out-of-fold predictions."""
    rng, uniq = np.random.default_rng(seed), np.unique(subj)
    idx = {s: np.where(subj == s)[0] for s in uniq}
    vals = {"bal_acc": [], "auc": []}
    for _ in range(n):
        take = np.concatenate([idx[s] for s in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[take])) < n_cls:
            continue
        for k, v in score(y[take], p[take], n_cls).items():
            vals[k].append(v)
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if v else (np.nan, np.nan)
            for k, v in vals.items()}


def baselines(df, splits, out):
    res = []
    for task, (mapping, keep) in TASKS.items():
        d = df[df.label.isin(keep)] if keep else df
        n_cls = len(set(mapping.values()))
        for fname, cols in FEATURES.items():
            dd = d.dropna(subset=cols)
            if dd.empty or dd.subject.nunique() < 10:
                continue
            oof_y, oof_p, oof_s, per_fold = [], [], [], []
            for f in splits["folds"]:
                tr, va = dd[dd.subject.isin(f["train"])], dd[dd.subject.isin(f["val"])]
                ytr, yva = tr.label.map(mapping).values, va.label.map(mapping).values
                if len(np.unique(ytr)) < n_cls or len(va) == 0:
                    continue
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(class_weight="balanced", max_iter=2000))
                m.fit(tr[cols].values, ytr)
                p = m.predict_proba(va[cols].values)
                per_fold.append(score(yva, p, n_cls))
                oof_y.append(yva), oof_p.append(p), oof_s.append(va.subject.values)
            if not per_fold:
                continue
            y, p, s = np.concatenate(oof_y), np.concatenate(oof_p), np.concatenate(oof_s)
            pooled, ci = score(y, p, n_cls), subject_bootstrap(y, p, s, n_cls)
            pf = pd.DataFrame(per_fold)
            res.append({"task": task, "features": fname, "n_scans": len(y),
                        "bal_acc": round(pooled["bal_acc"], 3),
                        "bal_acc 95% CI": f"{ci['bal_acc'][0]:.2f}-{ci['bal_acc'][1]:.2f}",
                        "auc": round(pooled["auc"], 3),
                        "auc 95% CI": f"{ci['auc'][0]:.2f}-{ci['auc'][1]:.2f}",
                        "auc fold sd": round(pf.auc.std(), 3)})
    tab = pd.DataFrame(res)
    tab.to_csv(os.path.join(out, "baselines.csv"), index=False)
    open(os.path.join(out, "baselines.md"), "w").write(
        "# Non-imaging baselines (same folds, pooled out-of-fold, subject-bootstrap CI)\n\n"
        + md(tab, index=False)
        + "\n\nRead: the CNN must beat 'age_sex (confound)' to show it uses the image, and should approach "
          "'nWBV (OASIS)'. If nWBV separates CN vs AD but the CNN is near 0.5, the image pipeline is losing "
          "the global atrophy signal. If 'mask_ml/eTIV (ours)' is much weaker than nWBV, check skull stripping.\n")
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out", default="runs/diagnose")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, splits = load(a.index), json.load(open(a.splits))
    test = set(splits.get("test", []))
    df = df[~df.subject.isin(test)]                      # never touch held-out test subjects
    fold_tab = audit(df, splits, a.out)
    tab = baselines(df, splits, a.out)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(fold_tab.to_string(index=False))
    print()
    print(tab.to_string(index=False))
    print(f"\nwrote {a.out}/label_audit.md, fold_counts.csv, baselines.md")


if __name__ == "__main__":
    main()
