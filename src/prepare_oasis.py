"""Build the scan index from OASIS-1 (cross-sectional) and/or OASIS-2 (longitudinal).

Produces the same index.csv as prepare_index.py, so every later stage (preprocess, splits,
world model, classifiers, attacks, web app) runs unchanged.

Expected layout after unpacking the downloads from oasis-brains.org (sub-folders such as
disc1/ or OAS2_RAW_PART1/ are fine; the search is recursive):

  OASIS-1   OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.hdr|.img
            OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr|.img
            + oasis_cross-sectional.csv
  OASIS-2   OAS2_0001_MR1/RAW/mpr-1.nifti.hdr|.img
            + oasis_longitudinal.csv

Labels come from the Clinical Dementia Rating at each session:
  CDR 0 -> CN,  CDR 0.5 -> MCI (very mild impairment),  CDR >= 1 -> AD (mild/moderate dementia)
OASIS has no ADNI-style MCI diagnosis, so "MCI" here means CDR 0.5; say so when reporting results.
OASIS has no scan dates: acq_date = 2000-01-01 + "MR Delay" days (OASIS-2) so visit order and the
time step between visits are exact; OASIS-1 sessions all get 2000-01-01.

  python src/prepare_oasis.py --oasis2_dir data/raw/OASIS2 --oasis2_csv oasis_longitudinal.csv \
      --oasis1_dir data/raw/OASIS1 --oasis1_csv oasis_cross-sectional.csv --out data/index.csv
"""
import argparse
import glob
import os
import re

import pandas as pd

BASE = pd.Timestamp("2000-01-01")


def read_table(path):
    """OASIS demographics come as .csv or .xlsx (e.g. oasis_longitudinal_demographics.xlsx)."""
    df = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def cdr_label(cdr):
    if pd.isna(cdr):
        return None
    return "CN" if cdr == 0 else ("MCI" if cdr == 0.5 else "AD")


def _sessions(root, prefix):
    """{session id: session dir} for every OAS1_xxxx_MRn / OAS2_xxxx_MRn folder under root."""
    out = {}
    for d in glob.glob(os.path.join(root, "**", f"{prefix}_*_MR*"), recursive=True):
        if os.path.isdir(d):
            out.setdefault(os.path.basename(d), d)
    return out


def _pick(paths, scan):
    """Prefer mpr-<scan>, else the lowest-numbered mpr."""
    def num(p):
        m = re.search(r"mpr-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 99
    paths = sorted(paths, key=num)
    exact = [p for p in paths if num(p) == scan]
    return (exact or paths or [None])[0]


def oasis1(root, csv, image, scan, min_age, include_reliability):
    df = read_table(csv)
    ses = _sessions(root, "OAS1")
    rows, missing = [], 0
    for r in df.to_dict("records"):
        sid = r["ID"]
        if sid.endswith("MR2") and not include_reliability:
            continue  # reliability re-scan within 90 days of MR1
        d = ses.get(sid)
        if d is None:
            missing += 1
            continue
        if image == "t88":
            path = (glob.glob(os.path.join(d, "PROCESSED", "MPRAGE", "T88_111", "*t88_masked_gfc.hdr")) or [None])[0]
        else:
            path = _pick(glob.glob(os.path.join(d, "RAW", "*mpr-*.hdr")), scan)
        lab = cdr_label(r.get("CDR"))
        if path is None or lab is None or r["Age"] < min_age:
            continue
        rows.append({"image_id": sid, "path": path, "is_dicom": False, "subject": "_".join(sid.split("_")[:2]),
                     "group_raw": f"CDR {r['CDR']}", "sex": r["M/F"], "age": float(r["Age"]),
                     "visit": sid.split("_")[-1], "acq_date": BASE.date().isoformat(),
                     "description": "MPRAGE T88 masked" if image == "t88" else "MPRAGE",
                     "mmscore": r.get("MMSE"), "cdglobal": r.get("CDR"), "educ": r.get("Educ"),
                     "etiv": r.get("eTIV"), "nwbv": r.get("nWBV"), "dataset": "OASIS-1",
                     "manufacturer": "SIEMENS", "scanner_manufacturer": "SIEMENS",
                     "field_strength": 1.5, "magnetic_field_strength": 1.5, "label": lab})
    return rows, missing


def oasis2(root, csv, scan):
    df = read_table(csv)
    ses = _sessions(root, "OAS2")
    rows, missing = [], 0
    for r in df.to_dict("records"):
        sid = r["MRI ID"]
        d = ses.get(sid)
        if d is None:
            missing += 1
            continue
        path = _pick(glob.glob(os.path.join(d, "RAW", "mpr-*.hdr")), scan)
        lab = cdr_label(r.get("CDR"))
        if path is None or lab is None:
            continue
        rows.append({"image_id": sid, "path": path, "is_dicom": False, "subject": r["Subject ID"],
                     "group_raw": r.get("Group"), "sex": r["M/F"], "age": float(r["Age"]),
                     "visit": f"v{int(r['Visit'])}",
                     "acq_date": (BASE + pd.Timedelta(days=int(r.get("MR Delay") or 0))).date().isoformat(),
                     "description": "MPRAGE", "mmscore": r.get("MMSE"), "cdglobal": r.get("CDR"),
                     "educ": r.get("EDUC"), "etiv": r.get("eTIV"), "nwbv": r.get("nWBV"), "dataset": "OASIS-2",
                     "manufacturer": "SIEMENS", "scanner_manufacturer": "SIEMENS",
                     "field_strength": 1.5, "magnetic_field_strength": 1.5, "label": lab})
    return rows, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oasis1_dir")
    ap.add_argument("--oasis1_csv")
    ap.add_argument("--oasis2_dir")
    ap.add_argument("--oasis2_csv")
    ap.add_argument("--oasis1_image", default="raw", choices=["raw", "t88"],
                    help="raw = run our own preprocessing (as for OASIS-2); t88 = OASIS skull-stripped "
                         "atlas images (then preprocess with --skullstrip premasked --no_neck_crop)")
    ap.add_argument("--scan", type=int, default=1, help="which repeat MPRAGE per session (OASIS has 3-4)")
    ap.add_argument("--min_age", type=float, default=60, help="OASIS-1 includes young adults without CDR")
    ap.add_argument("--include_reliability", action="store_true", help="keep OASIS-1 MR2 re-scans")
    ap.add_argument("--out", default="data/index.csv")
    a = ap.parse_args()
    if not (a.oasis1_dir or a.oasis2_dir):
        ap.error("give --oasis1_dir/--oasis1_csv and/or --oasis2_dir/--oasis2_csv")

    rows = []
    if a.oasis2_dir:
        r, miss = oasis2(a.oasis2_dir, a.oasis2_csv, a.scan)
        print(f"OASIS-2: {len(r)} sessions indexed ({miss} CSV rows with no image folder)")
        rows += r
    if a.oasis1_dir:
        r, miss = oasis1(a.oasis1_dir, a.oasis1_csv, a.oasis1_image, a.scan, a.min_age, a.include_reliability)
        print(f"OASIS-1: {len(r)} sessions indexed ({miss} CSV rows with no image folder)")
        rows += r
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no scans found - check the folder paths and CSV files")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    first = df.sort_values("acq_date").groupby("subject").first()
    print(f"{len(df)} scans, {df.subject.nunique()} subjects -> {a.out}")
    print("baseline labels:", first.label.value_counts().to_dict(), "| scan labels:", df.label.value_counts().to_dict())
    print("visits per subject:", df.groupby("subject").size().value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
