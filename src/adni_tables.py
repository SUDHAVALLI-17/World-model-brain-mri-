"""Readers for ADNI study-data tables and joining them to images.

MRI key    All_Subjects_Key_MRI_<date>.csv : one row per image (image_id, subject_id,
           image_visit, image_date, series_type, protocol phase, scanner ...). No diagnosis.
DXSUM      Diagnostic Summary table: diagnosis per subject per visit. Recent versions use
           DIAGNOSIS (1 = CN, 2 = MCI, 3 = Dementia); ADNI1 rows may only have DXCURREN
           (1 = NL, 2 = MCI, 3 = AD) or DXCHANGE (1-3 stable, 4-6 conversions, 7-9 reversions).
           ADNIMERGE.csv (column DX) also works.

Labels are attached per image by (subject, visit code); if the visit code does not match
(screening vs baseline, ADNI4 codes), the DXSUM row with the nearest exam date within
`max_days` is used instead.
"""
import re

import numpy as np
import pandas as pd

from common import map_group

DX_CODE = {1: "CN", 2: "MCI", 3: "AD"}
DXCHANGE = {1: "CN", 2: "MCI", 3: "AD", 4: "MCI", 5: "AD", 6: "AD", 7: "CN", 8: "MCI", 9: "CN"}


def _norm(df):
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_") for c in df.columns]
    return df


def norm_visit(v):
    """ADNI visit code. Screening carries the baseline diagnosis; IDA XML names such as
    'ADNI1/GO Month 36' or 'ADNI2 Baseline' are converted to codes (m36, bl)."""
    v = str(v).strip().lower() if v is not None and not (isinstance(v, float) and np.isnan(v)) else ""
    m = re.search(r"month\s*(\d+)", v)
    if m:
        return f"m{int(m.group(1)):02d}"
    if "baseline" in v or "screening" in v:
        return "bl"
    return "bl" if v in ("sc", "scmri", "bl", "init", "v01", "4_sc", "4_bl", "4_init") else v


def load_mri_key(path):
    k = _norm(pd.read_csv(path, low_memory=False))
    k["image_id"] = "I" + k["image_id"].astype(str).str.replace(r"^I", "", regex=True)
    k["image_date"] = pd.to_datetime(k.image_date, errors="coerce")
    if "scanner_manufacturer" in k.columns:  # 'Siemens' vs 'SIEMENS' etc. are the same vendor
        k["scanner_manufacturer"] = k.scanner_manufacturer.astype(str).str.upper().str.split().str[0]
    return k


def t1_visits(key, accel_ok=True):
    """One 3D T1w image per subject-visit (prefers the non-repeat, then unaccelerated scan)."""
    t1 = key[(key.series_type == "T1w") & (key.acquisition_type == "3D")].copy()
    if not accel_ok:
        t1 = t1[t1.acceleration.fillna("").str.lower().str.startswith("unacc")]
    t1["_rep"] = t1.series_description.fillna("").str.contains("repeat", case=False).astype(int)
    t1["_acc"] = (~t1.acceleration.fillna("").str.lower().str.startswith("unacc")).astype(int)
    return (t1.sort_values(["subject_id", "image_date", "_rep", "_acc", "image_id"])
              .drop_duplicates(["subject_id", "image_date"]).drop(columns=["_rep", "_acc"]))


def load_dx(path):
    """Return DataFrame(subject, visit, exam_date, dx) from DXSUM or ADNIMERGE."""
    d = _norm(pd.read_csv(path, low_memory=False))
    subj = next(c for c in ("ptid", "subject_id", "subject") if c in d.columns)
    vis = next((c for c in ("viscode2", "viscode", "visit") if c in d.columns), None)
    date = next((c for c in ("examdate", "exam_date", "userdate") if c in d.columns), None)
    dx = pd.Series([None] * len(d), index=d.index, dtype=object)
    if "diagnosis" in d.columns:
        dx = dx.fillna(pd.to_numeric(d.diagnosis, errors="coerce").map(DX_CODE))
    if "dxcurren" in d.columns:
        dx = dx.fillna(pd.to_numeric(d.dxcurren, errors="coerce").map(DX_CODE))
    if "dxchange" in d.columns:
        dx = dx.fillna(pd.to_numeric(d.dxchange, errors="coerce").map(DXCHANGE))
    if "dx" in d.columns:  # ADNIMERGE text labels
        dx = dx.fillna(d.dx.map(map_group))
    out = pd.DataFrame({"subject": d[subj].astype(str), "dx": dx,
                        "visit": d[vis].map(norm_visit) if vis else "",
                        "exam_date": pd.to_datetime(d[date], errors="coerce") if date else pd.NaT})
    out = out.dropna(subset=["dx"])
    if out.empty:
        raise ValueError(f"no diagnosis column recognised in {path}: {list(d.columns)[:30]}")
    return out


def attach_dx(df, dx, max_days=180):
    """df needs subject, visit (code), acq_date. Adds dx_visit + how it was matched."""
    df = df.copy()
    by_visit = dx.drop_duplicates(["subject", "visit"]).set_index(["subject", "visit"]).dx
    labels, how = [], []
    dates = pd.to_datetime(df.acq_date, errors="coerce")
    for (_, r), dt in zip(df.iterrows(), dates):
        key = (str(r.subject), norm_visit(r.get("visit")))
        if key in by_visit.index:
            labels.append(by_visit[key]), how.append("visit")
            continue
        cand = dx[(dx.subject == key[0]) & dx.exam_date.notna()]
        if len(cand) and pd.notna(dt):
            gap = (cand.exam_date - dt).abs().dt.days
            j = gap.idxmin()
            if gap[j] <= max_days:
                labels.append(cand.dx[j]), how.append("date")
                continue
        labels.append(None), how.append(None)
    df["dx_visit"], df["dx_match"] = labels, how
    return df
