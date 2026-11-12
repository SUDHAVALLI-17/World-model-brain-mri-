"""Metadata defences (Paper 1, RQ-M4): which layer catches which metadata attack.

Layer 1  HMAC signature over the clinically used header fields (subject, image id,
         age, sex, acquisition date). Catches *any* edit made after signing, e.g. in
         transit or in PACS. Does NOT catch an insider who edits before signing.
Layer 2  Plausibility rules. No key needed; catches clumsy edits:
           - age outside [40, 100], sex not M/F, missing fields
           - acquisition date unparseable or in the future
           - per subject: age difference between visits must match date difference
             (+- tol years), and gaps must be in [0, 15] years
Layer 3  Image-metadata consistency (brain-age head, see paper1_eval.py): flags scans
         whose stated age is far from the age the image looks like.

CLI:
  python src/metadata_guard.py sign   --index data/proc/index_proc.csv --key_file ~/.adni.key
  python src/metadata_guard.py verify --index data/proc/index_proc.csv --key_file ~/.adni.key
"""
import argparse
import hashlib
import hmac
import json

import numpy as np
import pandas as pd

FIELDS = ["subject", "image_id", "age", "sex", "acq_date"]


def _canonical(row):
    vals = {}
    for f in FIELDS:
        v = row.get(f)
        if isinstance(v, float) and np.isnan(v):
            v = None
        elif isinstance(v, (float, np.floating)):
            v = round(float(v), 2)
        elif v is not None:
            v = str(v)
        vals[f] = v
    return json.dumps(vals, sort_keys=True)


def sign_row(row, key):
    return hmac.new(key, _canonical(row).encode(), hashlib.sha256).hexdigest()


def sign_df(df, key):
    df = df.copy()
    df["meta_sig"] = [sign_row(r, key) for r in df.to_dict("records")]
    return df


def verify_df(df, key):
    """Boolean Series: True = signature valid."""
    return pd.Series([hmac.compare_digest(sign_row(r, key), str(r.get("meta_sig")))
                      for r in df.to_dict("records")], index=df.index)


def plausibility(df, age_range=(40, 100), max_gap=15.0, tol=1.5, today=None):
    """Per-row list of violated rules (empty list = plausible)."""
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now()
    d = df.copy()
    d["_date"] = pd.to_datetime(d.acq_date, errors="coerce")
    issues = {i: [] for i in d.index}
    for i, r in d.iterrows():
        age = r.get("age")
        if age is None or pd.isna(age):
            issues[i].append("age_missing")
        elif not age_range[0] <= float(age) <= age_range[1]:
            issues[i].append("age_out_of_range")
        if str(r.get("sex", "")).upper()[:1] not in ("M", "F"):
            issues[i].append("sex_invalid")
        if pd.isna(r._date):
            issues[i].append("date_invalid")
        elif r._date > today:
            issues[i].append("date_in_future")
    for _, g in d.dropna(subset=["_date"]).groupby("subject"):
        g = g.sort_values("_date")
        idx = g.index.tolist()
        for a, b in zip(idx[:-1], idx[1:]):
            gap = (d.at[b, "_date"] - d.at[a, "_date"]).days / 365.25
            if gap > max_gap:
                issues[b].append("gap_too_long")
            aa, ab = d.at[a, "age"], d.at[b, "age"]
            if not (pd.isna(aa) or pd.isna(ab)) and abs((float(ab) - float(aa)) - gap) > tol:
                issues[a].append("age_date_mismatch")
                issues[b].append("age_date_mismatch")
    return pd.Series(issues)


# ------------------------------------------------------------------ tampering
def tamper(df, idx, kind, rng, years=5.0):
    """Apply one metadata attack to row idx of a copy of df (for catch-rate evaluation)."""
    d = df.copy()
    r = d.loc[idx]
    if kind == "age_shift":
        d.at[idx, "age"] = float(r.age) + rng.choice([-1, 1]) * years
    elif kind == "sex_flip":
        d.at[idx, "sex"] = "F" if str(r.sex).upper().startswith("M") else "M"
    elif kind == "date_shift":  # changes dt to neighbouring visits
        d.at[idx, "acq_date"] = (pd.to_datetime(r.acq_date) +
                                 pd.Timedelta(days=int(rng.choice([-1, 1]) * years * 365.25))).strftime("%Y-%m-%d")
    elif kind == "date_and_age_shift":  # careful attacker keeps age and date consistent
        s = rng.choice([-1, 1]) * years
        d.at[idx, "acq_date"] = (pd.to_datetime(r.acq_date) + pd.Timedelta(days=int(s * 365.25))).strftime("%Y-%m-%d")
        d.at[idx, "age"] = float(r.age) + s
        # shift the whole subject so between-visit consistency also holds
        for j in d.index[(d.subject == r.subject) & (d.index != idx)]:
            d.at[j, "acq_date"] = (pd.to_datetime(d.at[j, "acq_date"]) +
                                   pd.Timedelta(days=int(s * 365.25))).strftime("%Y-%m-%d")
            d.at[j, "age"] = float(d.at[j, "age"]) + s
    elif kind == "missing":
        d.at[idx, "age"] = np.nan
        d.at[idx, "sex"] = None
    elif kind == "inconsistent_age":
        d.at[idx, "age"] = 55.0 if float(r.age) >= 73 else 92.0
    else:
        raise ValueError(kind)
    return d


def catch_rates(df, key, kinds, n=50, seed=0, years=5.0):
    """For each attack kind: fraction of tampered rows flagged by plausibility and by HMAC."""
    rng = np.random.default_rng(seed)
    signed = sign_df(df, key)
    rows = []
    cand = signed.index[signed.age.notna()]
    base = plausibility(signed)
    for kind in kinds:
        hit_p = hit_h = 0
        picks = rng.choice(cand, size=min(n, len(cand)), replace=False)
        for idx in picks:
            t = tamper(signed, idx, kind, rng, years)
            changed = [i for i in t.index if _canonical(t.loc[i].to_dict()) != _canonical(signed.loc[i].to_dict())]
            plaus = plausibility(t)
            hit_p += any(len(plaus[i]) > len(base[i]) for i in changed) if changed else 0
            v = verify_df(t, key)
            hit_h += any(not v[i] for i in changed)
        rows.append({"attack": kind, "n": len(picks), "plausibility_catch": hit_p / len(picks),
                     "hmac_catch": hit_h / len(picks)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from secure_io import load_key
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sign", "verify", "check"])
    ap.add_argument("--index", required=True)
    ap.add_argument("--key_file", default="~/.adni.key")
    a = ap.parse_args()
    df = pd.read_csv(a.index)
    if a.cmd == "sign":
        sign_df(df, load_key(a.key_file)).to_csv(a.index, index=False)
        print(f"signed {len(df)} rows -> {a.index} (column meta_sig)")
    elif a.cmd == "verify":
        bad = (~verify_df(df, load_key(a.key_file))).sum()
        print("OK - all metadata signatures valid" if bad == 0 else f"TAMPERED: {bad} rows fail HMAC")
    else:
        p = plausibility(df)
        flagged = p[p.map(len) > 0]
        print(f"{len(flagged)} / {len(df)} rows violate plausibility rules")
        for i, v in flagged.head(20).items():
            print(" ", df.loc[i, "image_id"], v)
