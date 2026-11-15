"""Summarise the sanity runs written by run_cv.sh (SANITY=1) into one table with a verdict per run.

  overfit   train on N balanced scans, score on the SAME scans. Should reach ~100%:
            balanced accuracy >= 0.90 and AUC >= 0.95. Lower = the model cannot fit even a small set:
            look at feature collapse (feature_spread.py), the learning rate, the input data.
  shuffled  training labels randomly permuted, validation labels real, scored at the LAST epoch (the
            best epoch is chosen with real validation labels, which would bias it upwards). Should be
            near chance. AUC clearly ABOVE chance (> 0.65) = the label reaches the model some other way
            (a subject in two folds, a join bug, the label in the metadata). Below chance happens when the
            random shuffle itself happens to agree with the true labels less than chance ('agreement'
            column, chance = 1 / number of classes) and the images carry real signal; it is not a bug.

  python src/sanity_report.py --dir runs/cv/cdr_binary/sanity
"""
import argparse
import glob
import json
import os

import pandas as pd


def verdict(kind, m, agreement=None, n_cls=2):
    auc = m.get("auc", m.get("macro_auc"))
    bal = m.get("bal_acc")
    if auc is None or bal is None:
        return "no metrics"
    if kind == "overfit":
        return "OK: can fit" if bal >= 0.90 and auc >= 0.95 else "FAIL: cannot fit a small set"
    if kind == "shuffled_labels":
        if auc > 0.65:
            return "CHECK: above chance - possible leakage or label bug"
        if auc < 0.35:
            if agreement is not None and agreement < 1 / n_cls - 0.02:
                return "OK: below chance, explained by this shuffle's agreement"
            return "CHECK: below chance without explanation - rerun with another --seed"
        return "OK: chance"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*", "run_info.json"))):
        info = json.load(open(f))
        kind = info.get("sanity") or "?"
        # both checks are read at the last epoch (no selection on validation labels)
        m = info.get("last_metrics") or {}
        agr, k = info.get("shuffle_agreement"), info.get("n_classes", 2)
        rows.append({"run": os.path.basename(os.path.dirname(f)), "test": kind,
                     "epochs": info["args"].get("epochs"), "bal_acc": round(m.get("bal_acc", float("nan")), 3),
                     "auc": round(m.get("auc", m.get("macro_auc", float("nan"))), 3),
                     "agreement": None if agr is None else round(agr, 3),
                     "verdict": verdict(kind, m, agr, k)})
    if not rows:
        raise SystemExit(f"no run_info.json under {a.dir}")
    tab = pd.DataFrame(rows)
    try:
        body = tab.to_markdown(index=False)
    except ImportError:
        body = "```\n" + tab.to_string(index=False) + "\n```"
    open(os.path.join(a.dir, "sanity.md"), "w").write(
        "# Sanity checks\n\n" + body + "\n\noverfit = scored on its own training scans (must be ~1.0); "
        "shuffled = trained on permuted labels, scored on real validation labels (must be ~0.5).\n")
    print(tab.to_string(index=False))
    print(f"\nwrote {a.dir}/sanity.md")


if __name__ == "__main__":
    main()
