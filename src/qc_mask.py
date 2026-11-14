"""Visual check of orientation, neck crop and skull-strip mask, BEFORE crop/resize.

For each scan it writes one PNG with three rows:
  1. the raw file as stored (array axes, no orientation applied) - to see which way is up
  2. the oriented image (superior up, anterior right in sagittal) with the neck-crop window
     drawn as yellow lines: everything below the lower line would be removed
  3. the brain mask as a red outline on 3 sagittal, 3 coronal and 3 axial slices through the
     mask centre (+-15 mm), on exactly the image the mask was computed on

and a CSV with mask volume, mask size in mm and warnings (mask touching the bottom of the
image after the neck crop = brain was probably cut; volume outside 900-2,000 ml).

  python src/qc_mask.py --index data/index.csv --n 4 --out qc_mask --no_neck_crop
  python src/qc_mask.py --index data/index.csv --ids OAS2_0001_MR1 OAS2_0002_MR1 --out qc_mask
  python src/qc_mask.py --index data/index.csv --n 2 --compare_neck --out qc_mask   # both settings

Look at row 3: the red line should follow the outer edge of the brain (grey matter / CSF) all
round, include the cerebellum and the bottom of the temporal lobes, and exclude skull, scalp,
eyes and neck.
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import SimpleITK as sitk

from preprocess import apply_geometry, crop_neck, prepare_and_mask, read_image


def _header_info(path):
    try:
        import nibabel as nib
        im = nib.load(path)
        h = im.header
        codes = "".join(nib.aff2axcodes(im.affine))
        q = int(h["qform_code"]) if "qform_code" in h else -1
        s = int(h["sform_code"]) if "sform_code" in h else -1
        return f"{type(h).__name__} qform={q} sform={s} axcodes={codes} shape={im.shape[:3]}"
    except Exception as e:  # DICOM folders etc.
        return f"header: {e.__class__.__name__}"


def _mid(arr, ax):
    return np.take(arr, arr.shape[ax] // 2, axis=ax)


def qc_one(row, a, tag, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opts = SimpleNamespace(min_extent_mm=100.0, no_n4=a.no_n4, spacing=a.spacing, no_denoise=a.no_denoise,
                           no_neck_crop=a.no_neck_crop if tag is None else (tag == "no_neck_crop"),
                           skullstrip=a.skullstrip, bet_f=a.bet_f)
    raw = sitk.GetArrayFromImage(read_image(row["path"], bool(row["is_dicom"])))
    # full oriented image (no crop) for row 2
    full = apply_geometry(read_image(row["path"], bool(row["is_dicom"])), row.get("geometry"))
    full = sitk.DICOMOrient(full, "RAS")
    from preprocess import resample_iso
    full = resample_iso(full, opts.spacing)
    _, lo, top = crop_neck(full)
    fa = sitk.GetArrayFromImage(full)

    img, m, tool, info = prepare_and_mask(row, opts)
    arr, mask = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(m).astype(bool)
    sp = img.GetSpacing()[0]
    ml = float(mask.sum() * np.prod(img.GetSpacing()) / 1000)
    zz, yy, xx = np.where(mask) if mask.any() else (np.array([0]),) * 3
    size_mm = [(v.max() - v.min() + 1) * sp for v in (xx, yy, zz)]
    warn = []
    if not 900 <= ml <= 2000:
        warn.append(f"volume {ml:.0f} ml outside 900-2000")
    if not opts.no_neck_crop and zz.min() <= 1:
        warn.append("mask touches the bottom of the cropped image - neck crop probably cut the brain")
    if size_mm[2] < 95:
        warn.append(f"mask only {size_mm[2]:.0f} mm tall (expect ~110-140)")

    fig = plt.figure(figsize=(15, 13))
    g = fig.add_gridspec(5, 9, height_ratios=[1, 1.2, 1, 1, 1])
    for i in range(3):  # row 1: raw array axes
        ax = fig.add_subplot(g[0, 3 * i:3 * i + 3])
        ax.imshow(_mid(raw, i), cmap="gray", origin="lower", aspect="auto")
        ax.set_title(f"raw file, middle of array axis {i} (shape {raw.shape})", fontsize=8)
        ax.axis("off")
    cz, cy, cx = (int(v) for v in np.argwhere(mask).mean(0)) if mask.any() else (s // 2 for s in arr.shape)
    fcx = fa.shape[2] // 2
    views = [(fa[:, :, fcx], "sagittal: superior up, anterior right"),
             (fa[:, fa.shape[1] // 2, :], "coronal: superior up"),
             (fa[(lo + top) // 2], "axial (between crop lines)")]
    for i, (sl, t) in enumerate(views):  # row 2: oriented, crop window
        ax = fig.add_subplot(g[1, 3 * i:3 * i + 3])
        ax.imshow(sl, cmap="gray", origin="lower")
        if i < 2:
            ax.axhline(top, color="yellow", lw=1)
            ax.axhline(lo, color="yellow", lw=1, ls="--")
        ax.set_title(t + ("  | yellow: neck-crop window" if i < 2 else ""), fontsize=8)
        ax.axis("off")
    off = int(round(15 / sp))
    panels = [("sag", lambda k: (arr[:, :, k], mask[:, :, k]), cx),
              ("cor", lambda k: (arr[:, k, :], mask[:, k, :]), cy),
              ("ax", lambda k: (arr[k], mask[k]), cz)]
    for r, (name, get, c) in enumerate(panels):  # rows 3-5: mask outline
        lim = arr.shape[{"sag": 2, "cor": 1, "ax": 0}[name]] - 1
        for j, k in enumerate([c - off, c, c + off]):
            k = int(np.clip(k, 0, lim))
            im, mk = get(k)
            ax = fig.add_subplot(g[2 + r, 3 * j:3 * j + 3])
            ax.imshow(im, cmap="gray", origin="lower")
            if mk.any():
                ax.contour(mk, levels=[0.5], colors="r", linewidths=0.8)
            ax.set_title(f"{name} slice {k}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"{row['image_id']}  |  {tag or ('no neck crop' if opts.no_neck_crop else 'neck crop')}  |  "
                 f"stripper={tool}  mask={ml:.0f} ml  size L-R/A-P/S-I = "
                 f"{size_mm[0]:.0f}/{size_mm[1]:.0f}/{size_mm[2]:.0f} mm\n{_header_info(row['path'])}"
                 + ("\nWARNING: " + "; ".join(warn) if warn else ""), fontsize=9,
                 color="red" if warn else "black")
    fig.tight_layout()
    name = f"{row['image_id']}_{tag or ('nocrop' if opts.no_neck_crop else 'crop')}.png"
    fig.savefig(os.path.join(out_dir, name), dpi=80)
    plt.close(fig)
    return {"image_id": row["image_id"], "setting": tag or ("no_neck_crop" if opts.no_neck_crop else "neck_crop"),
            "stripper": tool, "mask_ml": round(ml, 1), "lr_mm": round(size_mm[0]), "ap_mm": round(size_mm[1]),
            "si_mm": round(size_mm[2]), "neck_crop": info.get("neck_crop"), "warnings": "; ".join(warn),
            "png": name}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="data/index.csv (before preprocessing)")
    ap.add_argument("--ids", nargs="*", help="image_id values to check (default: first --n)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="qc_mask")
    ap.add_argument("--skullstrip", default="auto",
                    choices=["auto", "synthstrip", "hdbet", "bet", "otsu", "premasked"])
    ap.add_argument("--bet_f", type=float, default=0.6)
    ap.add_argument("--no_neck_crop", action="store_true")
    ap.add_argument("--compare_neck", action="store_true", help="make both versions for each scan")
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--no_n4", action="store_true")
    ap.add_argument("--no_denoise", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.index)
    df = df[df.image_id.isin(a.ids)] if a.ids else df.head(a.n)
    rows = []
    for r in df.to_dict("records"):
        for tag in (["neck_crop", "no_neck_crop"] if a.compare_neck else [None]):
            res = qc_one(r, a, tag, a.out)
            rows.append(res)
            print(f"{res['image_id']:<18} {res['setting']:<13} {res['stripper']:<6} {res['mask_ml']:>7.0f} ml  "
                  f"L-R {res['lr_mm']} A-P {res['ap_mm']} S-I {res['si_mm']} mm  {res['warnings']}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "qc_mask.csv"), index=False)
    print(f"-> {a.out}/  (one PNG per scan and setting, plus qc_mask.csv)")


if __name__ == "__main__":
    main()
