"""Add a brain-TISSUE volume to index_proc.csv, from the preprocessed cubes (no reprocessing).

Why: the skull-strip mask (mask_ml) covers the whole brain INCLUDING the CSF in the sulci and
ventricles. When tissue is lost with atrophy, CSF takes its place inside the same mask, so mask_ml
barely changes. On OASIS-2, mask_ml / eTIV separated CN from AD with AUC 0.66, OASIS's nWBV (tissue
only) with 0.83. This counts tissue instead:

  tissue_frac  = share of in-brain voxels brighter than the CSF / tissue threshold of that scan
                 (Otsu threshold on the in-brain intensities of the normalised T1 cube; CSF is dark)
  tissue_ml    = mask_ml x tissue_frac      (the cube is resized, but a FRACTION does not depend on scale)
  tissue_ratio = tissue_ml / eTIV           (our analogue of nWBV, when eTIV is in the index)

  python src/add_tissue_volume.py --index data/proc/index_proc.csv --key_file <key>

Writes the columns into the same file (a copy of the original is kept as <index>.bak once).
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

from common import get_logger, load_volume

log = get_logger("tissue")


def otsu(values, bins=256):
    """Otsu threshold of a 1-D array (two classes, maximal between-class variance)."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-12)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between[:-1])])


def tissue_fraction(vol):
    brain = vol[vol > 0]
    if brain.size < 1000:
        return np.nan, np.nan
    t = otsu(brain)
    return float((brain > t).mean()), t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--key_file", default=None)
    ap.add_argument("--force", action="store_true", help="recompute even if the columns exist")
    a = ap.parse_args()
    df = pd.read_csv(a.index)
    if "tissue_frac" in df and df.tissue_frac.notna().all() and not a.force:
        log.info("tissue columns already present in %s (use --force to recompute)", a.index)
        return
    key = None
    if a.key_file:
        from secure_io import load_key
        key = load_key(a.key_file)
    fr, th = [], []
    for i, p in enumerate(df.npy):
        f, t = tissue_fraction(load_volume(p, key).astype(np.float32))
        fr.append(f)
        th.append(t)
        if (i + 1) % 50 == 0:
            log.info("%d / %d", i + 1, len(df))
    if not os.path.exists(a.index + ".bak"):
        shutil.copy(a.index, a.index + ".bak")
    df["tissue_frac"] = fr
    df["csf_threshold"] = th
    if "mask_ml" in df:
        df["tissue_ml"] = pd.to_numeric(df.mask_ml, errors="coerce") * df.tissue_frac
        if "etiv" in df:
            df["tissue_ratio"] = df.tissue_ml / pd.to_numeric(df.etiv, errors="coerce")
    df.to_csv(a.index, index=False)
    d = df.dropna(subset=["tissue_frac"])
    msg = f"tissue_frac median {d.tissue_frac.median():.3f}"
    if "nwbv" in d and "tissue_ratio" in d:
        r = d[["tissue_ratio", "nwbv"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(r) > 5:
            msg += f" | correlation of tissue_ratio with OASIS nWBV: r = {np.corrcoef(r.tissue_ratio, r.nwbv)[0, 1]:.3f}"
    for lab, g in d.groupby("label"):
        msg += f" | {lab} {g.tissue_frac.mean():.3f}"
    log.info(msg)
    log.info("wrote tissue_frac, tissue_ml, tissue_ratio to %s", a.index)


if __name__ == "__main__":
    main()
