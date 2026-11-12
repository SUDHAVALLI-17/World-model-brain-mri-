"""Stage 0 - Build an index of the ADNI subset you have.

Scans the IDA download folder for NIfTI files / DICOM series, extracts the
image ID (I12345) from the path, and joins it with the IDA collection CSV
(the CSV you download alongside the images). Optionally overrides the
screening "Group" with the per-visit diagnosis from ADNIMERGE.csv.

Example:
  python src/prepare_index.py --raw_dir data/raw/ADNI \
      --meta_csv data/raw/ADNI1_subset_10_30_2026.csv \
      --adnimerge data/raw/ADNIMERGE.csv --out data/index.csv
"""
import argparse
import os
import re

import pandas as pd

from common import LABELS, get_logger, map_group

IMG_ID_RE = re.compile(r"(?:^|[_/\\])I(\d{3,})(?=[_./\\]|$)")


def find_images(raw_dir):
    recs = []
    for root, _, files in os.walk(raw_dir):
        nii = [f for f in files if f.endswith((".nii", ".nii.gz"))]
        dcm = [f for f in files if f.lower().endswith(".dcm")]
        for f in nii:
            p = os.path.join(root, f)
            m = IMG_ID_RE.findall(p)
            recs.append(dict(image_id=f"I{m[-1]}" if m else None, path=p, is_dicom=False))
        if dcm:
            m = IMG_ID_RE.findall(root)
            recs.append(dict(image_id=f"I{m[-1]}" if m else None, path=root, is_dicom=True))
    return pd.DataFrame(recs, columns=["image_id", "path", "is_dicom"])


def norm_cols(df):
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_") for c in df.columns]
    return df


def norm_image_id(v):
    s = str(v).strip()
    return s if s.upper().startswith("I") else f"I{s}"


def attach_adnimerge(df, adnimerge_csv, max_days, log):
    am = norm_cols(pd.read_csv(adnimerge_csv, low_memory=False))
    need = {"ptid", "examdate", "dx"}
    if not need.issubset(am.columns):
        log.warning("ADNIMERGE missing columns %s - skipping per-visit DX", need - set(am.columns))
        return df
    am = am[["ptid", "examdate", "dx"]].dropna()
    am["examdate"] = pd.to_datetime(am["examdate"], errors="coerce")
    am = am.dropna()
    out = []
    for _, r in df.iterrows():
        cand = am[am.ptid == r.subject]
        dx = None
        if len(cand) and pd.notna(r.acq_date):
            gap = (cand.examdate - r.acq_date).abs().dt.days
            j = gap.idxmin()
            if gap[j] <= max_days:
                dx = map_group(cand.loc[j, "dx"])
        out.append(dx)
    df["dx_visit"] = out
    n = df.dx_visit.notna().sum()
    log.info("ADNIMERGE per-visit DX matched for %d / %d scans", n, len(df))
    df["label"] = df.dx_visit.fillna(df.label)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--meta_csv", default=None, help="IDA collection CSV")
    ap.add_argument("--meta_xml_dir", default=None,
                    help="folder of IDA per-image metadata XMLs (alternative or addition to --meta_csv)")
    ap.add_argument("--adnimerge", default=None, help="optional ADNIMERGE.csv for per-visit DX")
    ap.add_argument("--mri_key", default=None, help="All_Subjects_Key_MRI_*.csv: visit codes, phase, scanner")
    ap.add_argument("--dxsum", default=None, help="Diagnostic Summary (DXSUM) or ADNIMERGE: per-visit labels")
    ap.add_argument("--max_days", type=int, default=180)
    ap.add_argument("--desc_regex", default=None,
                    help="keep only series whose Description matches, e.g. 'MPRAGE|MP-RAGE|IR-SPGR'")
    ap.add_argument("--keep_repeats", action="store_true",
                    help="keep same-session repeat scans as extra images (same subject -> same split)")
    ap.add_argument("--out", default="data/index.csv")
    a = ap.parse_args()
    log = get_logger("prepare_index")

    imgs = find_images(a.raw_dir)
    log.info("found %d image files/series on disk", len(imgs))
    if imgs.image_id.isna().any():
        log.warning("%d files had no image ID in their path - ignored", imgs.image_id.isna().sum())
        imgs = imgs.dropna(subset=["image_id"])
    imgs = imgs.drop_duplicates("image_id")

    assert a.meta_csv or a.meta_xml_dir, "give --meta_csv and/or --meta_xml_dir"
    keep = {"subject": "subject", "group": "group_raw", "sex": "sex", "age": "age",
            "visit": "visit", "acq_date": "acq_date", "description": "description"}
    parts = []
    if a.meta_csv:
        m = norm_cols(pd.read_csv(a.meta_csv))
        id_col = next(c for c in m.columns if c in ("image_data_id", "image_id", "imageuid", "loni_image"))
        m["image_id"] = m[id_col].map(norm_image_id)
        parts.append(m.rename(columns={k: v for k, v in keep.items() if k in m.columns}))
    if a.meta_xml_dir:
        from ida_xml import parse_dir
        x = parse_dir(a.meta_xml_dir)
        log.info("IDA XML metadata: %d images, %d subjects with labels", len(x), x.subject.nunique())
        parts.append(x)
    meta = pd.concat(parts, ignore_index=True).drop_duplicates("image_id", keep="last")
    extra = [c for c in ("apoe_a1", "apoe_a2", "mmscore", "cdglobal", "faqtotal", "manufacturer",
                         "mfg_model", "field_strength") if c in meta.columns]
    for v in keep.values():
        if v not in meta.columns:
            meta[v] = None

    df = imgs.merge(meta[["image_id"] + list(keep.values()) + extra], on="image_id", how="inner")
    log.info("%d images matched to metadata", len(df))
    if a.desc_regex:
        df = df[df.description.fillna("").str.contains(a.desc_regex, case=False, regex=True)]
        log.info("%d after description filter", len(df))

    if a.mri_key:
        from adni_tables import load_mri_key
        k = load_mri_key(a.mri_key)[["image_id", "subject_id", "image_visit", "image_date", "mri_protocol_phase",
                                      "scanner_manufacturer", "magnetic_field_strength"]]
        df = df.merge(k, on="image_id", how="left")
        df["subject"] = df.subject.fillna(df.subject_id)
        df["visit"] = df.image_visit.fillna(df.visit)  # visit codes (sc, m06, ...) for DXSUM matching
        df["acq_date"] = df.acq_date.fillna(df.image_date.dt.strftime("%Y-%m-%d"))
        df = df.drop(columns=["subject_id", "image_visit", "image_date"])
        log.info("MRI key matched %d / %d images", df.mri_protocol_phase.notna().sum(), len(df))
    df["acq_date"] = pd.to_datetime(df.acq_date, errors="coerce")
    df["label"] = df.group_raw.map(map_group)
    if a.dxsum:
        from adni_tables import attach_dx, load_dx
        df = attach_dx(df, load_dx(a.dxsum))
        log.info("DXSUM labels: %d by visit code, %d by nearest date, %d unmatched",
                 (df.dx_match == "visit").sum(), (df.dx_match == "date").sum(), df.dx_match.isna().sum())
        df["label"] = df.dx_visit.fillna(df.label)
    if a.adnimerge:
        df = attach_adnimerge(df, a.adnimerge, a.max_days, log)
    df = df[df.label.isin(LABELS)]
    # one scan per subject per date (ADNI often has repeat MPRAGEs in the same session)
    # prefer the original over a "repeat" acquisition when both exist
    df["_rep"] = df.description.fillna("").str.contains("repeat", case=False).astype(int)
    df = df.sort_values(["subject", "acq_date", "_rep", "image_id"])
    if not a.keep_repeats:
        df = df.drop_duplicates(["subject", "acq_date"])
    df = df.drop(columns="_rep")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    per_subj = df.groupby("subject").size()
    log.info("saved %s: %d scans, %d subjects", a.out, len(df), df.subject.nunique())
    log.info("scans per class: %s", df.label.value_counts().to_dict())
    log.info("subjects with >=2 visits (usable for dynamics): %d", int((per_subj >= 2).sum()))


if __name__ == "__main__":
    main()
