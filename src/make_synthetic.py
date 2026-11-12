"""Create a fake ADNI-shaped dataset (IDA folder layout + IDA CSV) to smoke-test
the pipeline before real data arrives. Atrophy (bigger ventricles, smaller
hippocampus-like blobs) grows with diagnosis and with time between visits.

  python src/make_synthetic.py --out data_synth --subjects 30
"""
import argparse
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk


def ellipsoid(shape, c, r):
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return ((z - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((x - c[2]) / r[2]) ** 2 <= 1


def head(shape, atrophy, rng):
    c = np.array(shape) / 2 + rng.normal(0, 2, 3)
    vol = np.zeros(shape, np.float32)
    vol[ellipsoid(shape, c, (40, 46, 38))] = 0.9  # scalp/skull
    vol[ellipsoid(shape, c, (36, 42, 34))] = 0.15  # CSF
    vol[ellipsoid(shape, c, (34 - 3 * atrophy, 40 - 3 * atrophy, 32 - 3 * atrophy))] = 0.55  # GM
    vol[ellipsoid(shape, c, (26 - 2 * atrophy, 32 - 2 * atrophy, 24 - 2 * atrophy))] = 0.8  # WM
    for s in (-1, 1):
        vol[ellipsoid(shape, c + (0, 0, s * 6), (5 + 4 * atrophy, 12 + 3 * atrophy, 3 + 2 * atrophy))] = 0.15
        vol[ellipsoid(shape, c + (-12, 4, s * 18), (4 - 1.5 * atrophy, 9 - 3 * atrophy, 3 - atrophy))] = 0.6
    zz = np.linspace(-1, 1, shape[0])[:, None, None]
    bias = 1 + 0.2 * zz + 0.1 * np.linspace(-1, 1, shape[2])[None, None]
    return np.clip(vol * bias + rng.normal(0, 0.03, shape), 0, None) * 1000


SITES = ["002", "011", "023", "941"]
SCANNER = {"002": (1.0, 5), "011": (0.8, 25), "023": (1.25, 10), "941": (0.9, 40)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows, iid = [], 10000
    groups = ["CN", "MCI", "AD"]
    for s in range(a.subjects):
        g = groups[s % 3]
        site = SITES[s % len(SITES)]
        subj = f"{site}_S_{1000 + s:04d}"
        sex, age = rng.choice(["M", "F"]), float(rng.integers(60, 88))
        base = {"CN": 0.0, "MCI": 0.6, "AD": 1.2}[g] + rng.normal(0, 0.15)
        rate = {"CN": 0.02, "MCI": 0.12, "AD": 0.25}[g]
        n_vis = int(rng.integers(1, 4))
        for v in range(n_vis):
            iid += 1
            yrs = [0, 1, 2][v]
            date = pd.Timestamp("2006-03-01") + pd.Timedelta(days=int(365.25 * yrs + s * 3))
            vol = head((96, 110, 96), max(0, base + rate * yrs), rng)
            gam, noise = SCANNER[site]  # site-specific contrast + noise: the confound to debias
            vol = (vol / 1000) ** gam * 1000 + rng.normal(0, noise, vol.shape) * (vol > 50)
            img = sitk.GetImageFromArray(vol)
            img.SetSpacing((1.2, 1.2, 1.2))
            d = os.path.join(a.out, "raw", "ADNI", subj, "MPRAGE", date.strftime("%Y-%m-%d_08_00_00.0"),
                             f"I{iid}")
            os.makedirs(d, exist_ok=True)
            sitk.WriteImage(img, os.path.join(d, f"ADNI_{subj}_MR_MPRAGE_br_raw_S{iid}_I{iid}.nii.gz"))
            rows.append({"Image Data ID": f"I{iid}", "Subject": subj, "Group": g, "Sex": sex,
                         "Age": age + yrs, "Visit": ["sc", "m12", "m24"][v], "Modality": "MRI",
                         "Description": "MPRAGE", "Type": "Original",
                         "Acq Date": date.strftime("%m/%d/%Y"), "Format": "NiFTI"})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "raw", "IDA_collection.csv"), index=False)
    print(f"{len(rows)} scans for {a.subjects} subjects -> {a.out}/raw")


if __name__ == "__main__":
    main()
