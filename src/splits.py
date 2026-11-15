"""Stage 2 - Subject-level splits (no subject ever appears in two splits).

ADNI subjects have several visits; splitting by scan leaks the same brain
into train and test and inflates accuracy by 10-30 points. We hold out a
test set by subject, then build stratified-group CV folds on the rest.

  python src/splits.py --index data/proc/index_proc.csv --out data/splits.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify", default="baseline", choices=["baseline", "worst"],
                    help="label used to balance subjects across folds: their first visit (default, keeps "
                         "existing splits) or their most severe visit (spreads people who reach AD only at "
                         "follow-up evenly across folds)")
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline test on tiny data: every subject in train, val AND test (results invalid)")
    a = ap.parse_args()
    log = get_logger("splits")

    df = pd.read_csv(a.index)
    if a.smoke:
        subs = sorted(df.subject.unique())
        json.dump({"test": subs, "folds": [{"train": subs, "val": subs}], "smoke": True},
                  open(a.out, "w"), indent=1)
        log.warning("SMOKE SPLIT: %d subjects used for train, val and test - for checking that the "
                    "code runs only; any accuracy reported from this split is meaningless", len(subs))
        return
    # stratify on each subject's baseline (or most severe) label
    first = df.sort_values("acq_date").groupby("subject").first().reset_index()
    if a.stratify == "worst":
        sev = {"CN": 0, "MCI": 1, "AD": 2}
        worst = df.assign(_s=df.label.map(sev)).groupby("subject")._s.max().map({v: k for k, v in sev.items()})
        first["label"] = first.subject.map(worst)
    if first.subject.nunique() < 10 or first.label.value_counts().min() < 2:
        raise SystemExit(f"only {first.subject.nunique()} subjects / class counts "
                         f"{first.label.value_counts().to_dict()}: too few for a real split. "
                         "Download more subjects, or use --smoke to test that the code runs.")
    subj, y = first.subject.values, first.label.values

    n_test = max(2, int(round(1 / a.test_frac)))
    sgk = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=a.seed)
    tr_idx, te_idx = next(sgk.split(subj, y, groups=subj))
    test_s, dev_s, dev_y = subj[te_idx], subj[tr_idx], y[tr_idx]

    k = min(a.folds, int(pd.Series(dev_y).value_counts().min()))
    k = max(k, 2)
    folds = []
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=a.seed)
    for tr, va in sgk.split(dev_s, dev_y, groups=dev_s):
        folds.append({"train": dev_s[tr].tolist(), "val": dev_s[va].tolist()})

    json.dump({"test": test_s.tolist(), "folds": folds}, open(a.out, "w"), indent=1)
    log.info("test subjects: %d %s", len(test_s),
             pd.Series(y[te_idx]).value_counts().to_dict())
    log.info("%d CV folds on %d dev subjects (folds reduced if a class is tiny)", k, len(dev_s))
    if k < a.folds:
        log.warning("smallest class has only %d dev subjects -> using %d folds", k, k)


if __name__ == "__main__":
    main()
