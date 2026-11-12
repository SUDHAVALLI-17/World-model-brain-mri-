"""Choose which ADNI subjects / images to download from IDA.

Uses the MRI key (image catalogue) and, if given, DXSUM/ADNIMERGE for per-visit labels.
Picks subjects with at least --min_visits 3D T1 visits (so the world-model dynamics has
real follow-ups), balanced by baseline diagnosis when labels are available, and writes:

  download_subjects.txt   comma-separated subject IDs -> paste into IDA Advanced Search
  download_images.csv     the exact T1 image IDs, visits, dates and scanners
  plan_summary.txt        counts per class, visits, scanners, estimated download size

  python src/plan_download.py --mri_key All_Subjects_Key_MRI_10Nov2026.csv \
      --dxsum DXSUM.csv --per_class 40 --min_visits 3 --out data/plan
"""
import argparse
import os

import numpy as np
import pandas as pd

from adni_tables import attach_dx, load_dx, load_mri_key, t1_visits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mri_key", required=True)
    ap.add_argument("--dxsum", default=None, help="DXSUM or ADNIMERGE csv (optional)")
    ap.add_argument("--per_class", type=int, default=40, help="subjects per baseline class")
    ap.add_argument("--n_subjects", type=int, default=120, help="used when no labels are given")
    ap.add_argument("--min_visits", type=int, default=3)
    ap.add_argument("--max_visits", type=int, default=4, help="cap visits per subject (download size)")
    ap.add_argument("--phases", nargs="*", default=None, help="e.g. ADNIGO/ADNI2 ADNI3")
    ap.add_argument("--field", type=float, default=None, help="e.g. 3.0 to keep one field strength")
    ap.add_argument("--include", nargs="*", default=[], help="subjects you already have (always kept)")
    ap.add_argument("--mb_per_scan", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/plan")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    t1 = t1_visits(load_mri_key(a.mri_key))
    if a.phases:
        t1 = t1[t1.mri_protocol_phase.isin(a.phases)]
    if a.field:
        t1 = t1[np.isclose(t1.magnetic_field_strength, a.field)]
    t1 = t1.rename(columns={"subject_id": "subject", "image_visit": "visit", "image_date": "acq_date"})

    labelled = a.dxsum is not None
    if labelled:
        t1 = attach_dx(t1, load_dx(a.dxsum))
        t1 = t1.dropna(subset=["dx_visit"])
    counts = t1.groupby("subject").size()
    ok = counts[counts >= a.min_visits].index
    t1 = t1[t1.subject.isin(ok) | t1.subject.isin(a.include)]

    if labelled:
        base = t1.sort_values("acq_date").groupby("subject").dx_visit.first()
        conv = t1.sort_values("acq_date").groupby("subject").dx_visit.agg(lambda s: s.iloc[0] != s.iloc[-1])
        picks = list(a.include)
        for cls in ("CN", "MCI", "AD"):
            pool = base[base == cls].index.difference(picks)
            # for MCI, prefer a mix of converters and stable subjects (converters are the interesting cases)
            if cls == "MCI":
                c, s = pool[conv[pool].values], pool[~conv[pool].values]
                k = min(len(c), a.per_class // 2)
                chosen = list(rng.choice(c, k, replace=False)) + \
                    list(rng.choice(s, min(len(s), a.per_class - k), replace=False))
            else:
                chosen = list(rng.choice(pool, min(len(pool), a.per_class), replace=False))
            picks += chosen
    else:
        pool = counts[counts >= a.min_visits].index.difference(a.include)
        picks = list(a.include) + list(rng.choice(pool, min(len(pool), a.n_subjects), replace=False))

    sel = t1[t1.subject.isin(picks)].sort_values(["subject", "acq_date"])
    sel = sel.groupby("subject").head(a.max_visits)  # earliest visits first
    cols = [c for c in ("subject", "image_id", "visit", "acq_date", "dx_visit", "dx_match",
                        "mri_protocol_phase", "series_description", "acceleration", "scanner_manufacturer",
                        "magnetic_field_strength") if c in sel.columns]
    sel[cols].to_csv(os.path.join(a.out, "download_images.csv"), index=False)
    subs = sorted(sel.subject.unique())
    open(os.path.join(a.out, "download_subjects.txt"), "w").write(",".join(subs))

    lines = [f"subjects: {len(subs)}   T1 scans: {len(sel)}   "
             f"estimated download: ~{len(sel) * a.mb_per_scan / 1000:.1f} GB (T1 only)",
             f"visits per subject: {sel.groupby('subject').size().value_counts().sort_index().to_dict()}",
             f"protocol phase: {sel.mri_protocol_phase.value_counts().to_dict()}",
             f"scanner: {sel.scanner_manufacturer.value_counts().to_dict()}",
             f"field strength: {sel.magnetic_field_strength.value_counts().to_dict()}"]
    if labelled:
        b = sel.groupby("subject").dx_visit.first()
        lines.append(f"baseline diagnosis (subjects): {b.value_counts().to_dict()}")
        lines.append(f"scan-level labels: {sel.dx_visit.value_counts().to_dict()}")
        lines.append(f"diagnosis changes during follow-up: {int(sel.groupby('subject').dx_visit.nunique().gt(1).sum())} subjects")
    else:
        lines.append("NO LABELS: pass --dxsum (Diagnostic Summary) or ADNIMERGE to balance classes")
    open(os.path.join(a.out, "plan_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {a.out}/download_subjects.txt  (paste into IDA Advanced Search > Subject ID)")


if __name__ == "__main__":
    main()
