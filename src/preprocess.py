"""Stage 1 - Digital image processing (DIP) pipeline for T1 MRI.

Per scan:
  1. read DICOM series / NIfTI, take first volume if 4D
  2. reorient to RAS (consistent axes across scanners)
  3. N4 bias-field correction (removes low-frequency intensity inhomogeneity)
  4. resample to isotropic spacing (linear interpolation)
  5. edge-preserving denoising (curvature anisotropic diffusion)
  6. brain/head mask: SynthStrip or HD-BET if installed, else Otsu threshold
     + morphology (opening, largest connected component, closing, hole fill)
  7. crop to mask, pad to cube, resize to target^3
  8. intensity windowing (0.5-99.5 pct inside mask) -> [0, 1], background 0
  9. save float16 .npy (optionally AES-encrypted) + QC png

Example:
  python src/preprocess.py --index data/index.csv --out_dir data/proc \
      --target 96 --spacing 1.5 --skullstrip auto --workers 4
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom

from common import get_logger


def read_image(path, is_dicom):
    if is_dicom:
        r = sitk.ImageSeriesReader()
        ids = r.GetGDCMSeriesIDs(path)
        files = r.GetGDCMSeriesFileNames(path, ids[0]) if ids else sorted(glob(os.path.join(path, "*.dcm")))
        r.SetFileNames(files)
        img = r.Execute()
    else:
        img = sitk.ReadImage(path)
    if img.GetDimension() == 4:
        size = list(img.GetSize())
        img = sitk.Extract(img, size[:3] + [0], [0, 0, 0, 0])
    return sitk.Cast(img, sitk.sitkFloat32)


def n4_correct(img, shrink=4, iters=(50, 50, 30)):
    arr_min = float(sitk.GetArrayViewFromImage(img).min())
    img = img - arr_min + 1.0  # N4 works in log space -> needs positive values
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    small = sitk.Shrink(img, [shrink] * 3)
    small_mask = sitk.Shrink(mask, [shrink] * 3)
    f = sitk.N4BiasFieldCorrectionImageFilter()
    f.SetMaximumNumberOfIterations(list(iters))
    f.Execute(small, small_mask)
    log_bias = f.GetLogBiasFieldAsImage(img)
    return img / sitk.Exp(log_bias)


def resample_iso(img, spacing):
    new_size = [int(round(sz * sp / spacing)) for sz, sp in zip(img.GetSize(), img.GetSpacing())]
    return sitk.Resample(img, new_size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(),
                         [spacing] * 3, img.GetDirection(), 0.0, sitk.sitkFloat32)


def otsu_mask(img):
    m = sitk.OtsuThreshold(img, 0, 1, 200)
    m = sitk.BinaryMorphologicalOpening(m, [2, 2, 2])
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(m), sortByObjectSize=True)
    m = sitk.Cast(cc == 1, sitk.sitkUInt8)
    m = sitk.BinaryMorphologicalClosing(m, [3, 3, 3])
    return sitk.BinaryFillhole(m)


def external_mask(img, tool):
    """Deep-learning skull stripping via SynthStrip or HD-BET command-line tools."""
    with tempfile.TemporaryDirectory() as td:
        src, out, msk = (os.path.join(td, n) for n in ("in.nii.gz", "out.nii.gz", "mask.nii.gz"))
        sitk.WriteImage(img, src)
        if tool == "synthstrip":
            subprocess.run(["mri_synthstrip", "-i", src, "-o", out, "-m", msk], check=True,
                           capture_output=True)
        else:  # hd-bet
            subprocess.run(["hd-bet", "-i", src, "-o", out, "-device", "cpu", "-mode", "fast", "-tta", "0"],
                           check=True, capture_output=True)
            msk = out.replace(".nii.gz", "_mask.nii.gz")
        m = sitk.ReadImage(msk)
        m.CopyInformation(img)
        return sitk.Cast(m > 0, sitk.sitkUInt8)


def bet_mask(img, f=0.6):
    """Classical FSL-BET-style surface deformation (pure Python `brainextractor` package).
    No deep-learning weights needed - the fallback when SynthStrip / HD-BET are unavailable."""
    import contextlib
    import io
    import nibabel as nib
    from brainextractor import BrainExtractor
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "in.nii.gz")
        sitk.WriteImage(img, p)
        with contextlib.redirect_stdout(io.StringIO()):
            bet = BrainExtractor(nib.load(p), bt=f)
            bet.run(iterations=1000)
            m = bet.compute_mask()
    m = sitk.GetImageFromArray((m.transpose(2, 1, 0) > 0).astype(np.uint8))  # nibabel xyz -> itk zyx
    m.CopyInformation(img)
    return sitk.BinaryFillhole(m)


def crop_neck(img, keep_mm=170.0):
    """Keep only the top `keep_mm` of the head along the superior axis (like FSL robustfov).
    ADNI sagittal MPRAGEs include a lot of neck, which otherwise pulls skull stripping down."""
    arr = sitk.GetArrayViewFromImage(img)
    head = arr > np.percentile(arr, 60)
    rows = np.where(head.reshape(head.shape[0], -1).mean(1) > 0.01)[0]
    if len(rows) == 0:
        return img
    top = int(rows.max())
    lo = max(0, top - int(keep_mm / img.GetSpacing()[2]))
    return img[:, :, lo:top + 1]


def pick_stripper(choice):
    if choice != "auto":
        return choice
    if shutil.which("mri_synthstrip"):
        return "synthstrip"
    if shutil.which("hd-bet"):
        return "hdbet"
    try:
        import brainextractor  # noqa: F401
        return "bet"
    except ImportError:
        return "otsu"


def crop_pad_resize(arr, mask, target, margin=2):
    zz, yy, xx = np.where(mask)
    lo = np.maximum([zz.min() - margin, yy.min() - margin, xx.min() - margin], 0)
    hi = np.minimum([zz.max() + margin + 1, yy.max() + margin + 1, xx.max() + margin + 1], arr.shape)
    arr = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    mask = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    side = max(arr.shape)
    pads = [((side - s) // 2, side - s - (side - s) // 2) for s in arr.shape]
    arr = np.pad(arr, pads)
    mask = np.pad(mask, pads)
    f = target / side
    return zoom(arr, f, order=1), zoom(mask.astype(np.uint8), f, order=0).astype(bool)


def normalize(arr, mask, lo_p=0.5, hi_p=99.5):
    lo, hi = np.percentile(arr[mask], [lo_p, hi_p])
    out = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    out[~mask] = 0
    return out.astype(np.float32)


def save_qc(arr, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    c = [s // 2 for s in arr.shape]
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for a, sl in zip(ax, [arr[c[0]], arr[:, c[1]], arr[:, :, c[2]]]):
        a.imshow(np.flipud(sl), cmap="gray")
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def process_one(row, a):
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(a.threads)
    img = read_image(row["path"], bool(row["is_dicom"]))
    extent = [sz * sp for sz, sp in zip(img.GetSize(), img.GetSpacing())]
    if min(extent) < a.min_extent_mm:  # e.g. an interrupted download with 20 of 170 slices
        raise ValueError(f"truncated series: extent {[round(e) for e in extent]} mm < {a.min_extent_mm} mm")
    img = sitk.DICOMOrient(img, "RAS")
    if not a.no_n4:
        img = n4_correct(img)
    img = resample_iso(img, a.spacing)
    if not a.no_denoise:
        img = sitk.CurvatureAnisotropicDiffusion(img, timeStep=0.03, conductanceParameter=1.5,
                                                 numberOfIterations=5)
    if not a.no_neck_crop:
        img = crop_neck(img)
    tool = pick_stripper(a.skullstrip)
    if tool in ("synthstrip", "hdbet"):
        m = external_mask(img, tool)
    elif tool == "bet":
        m = bet_mask(img, a.bet_f)
    else:
        m = otsu_mask(img)
    arr = sitk.GetArrayFromImage(img)
    mask = sitk.GetArrayFromImage(m).astype(bool)
    brain_ml = float(mask.sum() * np.prod(img.GetSpacing()) / 1000)
    if tool != "otsu":
        arr = arr * mask
    arr, mask = crop_pad_resize(arr, mask, a.target)
    vol = normalize(arr, mask).astype(np.float16)

    out = os.path.join(a.out_dir, "vols", f"{row['image_id']}.npy")
    if a.encrypt:
        from secure_io import load_key, save_encrypted
        out += ".enc"
        save_encrypted(out, vol, load_key(a.key_file))
    else:
        np.save(out, vol)
    if a.qc:
        save_qc(vol.astype(np.float32), os.path.join(a.out_dir, "qc", f"{row['image_id']}.png"))
    return out, tool, brain_ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out_dir", default="data/proc")
    ap.add_argument("--target", type=int, default=96, help="output cube size (must be divisible by 16)")
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--skullstrip", default="auto", choices=["auto", "synthstrip", "hdbet", "bet", "otsu"])
    ap.add_argument("--no_neck_crop", action="store_true")
    ap.add_argument("--bet_f", type=float, default=0.6, help="BET fractional threshold (higher = tighter mask)")
    ap.add_argument("--min_extent_mm", type=float, default=100.0, help="reject truncated series")
    ap.add_argument("--brain_ml_range", type=float, nargs=2, default=[900, 2000],
                    help="QC: flag masks whose volume (ml) is outside this range")
    ap.add_argument("--no_n4", action="store_true", help="skip if you downloaded N3m-corrected images")
    ap.add_argument("--no_denoise", action="store_true")
    ap.add_argument("--encrypt", action="store_true", help="AES-encrypt volumes at rest")
    ap.add_argument("--key_file", default=None)
    ap.add_argument("--qc", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2, help="ITK threads per worker")
    a = ap.parse_args()
    assert a.target % 16 == 0, "--target must be divisible by 16"
    log = get_logger("preprocess", a.out_dir)
    os.makedirs(os.path.join(a.out_dir, "vols"), exist_ok=True)
    if a.qc:
        os.makedirs(os.path.join(a.out_dir, "qc"), exist_ok=True)

    df = pd.read_csv(a.index)
    log.info("preprocessing %d scans | skullstrip=%s", len(df), pick_stripper(a.skullstrip))
    results, vols = {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_one, r.to_dict(), a): r.image_id for _, r in df.iterrows()}
        for i, f in enumerate(as_completed(futs), 1):
            iid = futs[f]
            try:
                results[iid], _, vols[iid] = f.result()
                lo, hi = a.brain_ml_range
                if not lo <= vols[iid] <= hi:
                    log.warning("QC %s: mask volume %.0f ml outside [%g, %g] - check skull stripping",
                                iid, vols[iid], lo, hi)
            except ValueError as e:
                log.warning("SKIPPED %s: %s", iid, e)
            except Exception:
                log.error("FAILED %s\n%s", iid, traceback.format_exc(limit=2))
            if i % 10 == 0 or i == len(futs):
                log.info("%d / %d done", i, len(futs))
    df["npy"] = df.image_id.map(results)
    df["mask_ml"] = df.image_id.map(vols)
    df = df.dropna(subset=["npy"])
    out_csv = os.path.join(a.out_dir, "index_proc.csv")
    df.to_csv(out_csv, index=False)
    log.info("saved %s (%d scans)", out_csv, len(df))
    if a.encrypt:
        from secure_io import write_manifest
        write_manifest(os.path.join(a.out_dir, "vols"), os.path.join(a.out_dir, "manifest.json"))
        log.info("integrity manifest written")


if __name__ == "__main__":
    main()
