"""Inference engine: one uploaded scan -> diagnosis, forecast, checks.

Uses the same preprocessing and model code as the research pipeline (src/), so the web
app sees scans exactly the way the models were trained on them.
"""
import base64
import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from datetime import date, datetime

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

SRC = os.environ.get("SRC_DIR", os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.abspath(SRC))
from common import LABELS, encode_action  # noqa: E402
from data import brain_parenchymal_fraction  # noqa: E402
from evaluate import gradcam, load_classifier  # noqa: E402
from models import BrainWorldModel  # noqa: E402
from preprocess import (bet_mask, crop_neck, crop_pad_resize, external_mask, n4_correct,  # noqa: E402
                        normalize, otsu_mask, pick_stripper, read_image, resample_iso)
import trajectory_attacks as TA  # noqa: E402
from header_anomaly import HeaderDetector, read_header, with_form  # noqa: E402

MAX_ZIP_FILES = 3000
MAX_UNZIPPED_BYTES = 1_500_000_000
T1_RE = re.compile(r"RAGE|SPGR|T1", re.I)


class InputError(ValueError):
    """Problem with the uploaded file: shown to the user as a 400."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------ reading uploads
def safe_extract(zip_path, dest):
    """Extract a zip without zip-slip, links or zip bombs."""
    try:
        z = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise InputError("not a valid zip file")
    infos = z.infolist()
    if len(infos) > MAX_ZIP_FILES:
        raise InputError(f"zip has {len(infos)} entries (limit {MAX_ZIP_FILES})")
    if sum(i.file_size for i in infos) > MAX_UNZIPPED_BYTES:
        raise InputError("zip expands to more than 1.5 GB")
    root = os.path.realpath(dest)
    for i in infos:
        target = os.path.realpath(os.path.join(root, i.filename))
        if not target.startswith(root + os.sep) and target != root:
            raise InputError(f"unsafe path in zip: {i.filename!r}")
        if (i.external_attr >> 16) & 0o170000 == 0o120000:
            raise InputError("zip contains a symbolic link")
    z.extractall(root)


def _tag(reader, key):
    return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else None


def find_t1_series(folder):
    """Pick the T1 series from an extracted DICOM folder; return (dir, header dict)."""
    best = None
    for d, _, files in os.walk(folder):
        dcm = [f for f in files if f.lower().endswith(".dcm") or "." not in f]
        if len(dcm) < 20:
            continue
        r = sitk.ImageFileReader()
        r.SetFileName(os.path.join(d, sorted(dcm)[0]))
        r.LoadPrivateTagsOff()
        try:
            r.ReadImageInformation()
        except RuntimeError:
            continue
        desc = _tag(r, "0008|103e") or os.path.basename(os.path.dirname(d)) or ""
        score = (bool(T1_RE.search(desc)) and "repeat" not in desc.lower(), bool(T1_RE.search(desc)), len(dcm))
        hdr = {"description": desc, "age": _tag(r, "0010|1010"), "sex": _tag(r, "0010|0040"),
               "date": _tag(r, "0008|0022") or _tag(r, "0008|0020"), "manufacturer": _tag(r, "0008|0070"),
               "field_strength": _tag(r, "0018|0087"), "n_slices": len(dcm), "raw": read_header(r, len(dcm))}
        if best is None or score > best[0]:
            best = (score, d, hdr)
    if best is None:
        raise InputError("no DICOM series with 20 or more slices found in the zip")
    return best[1], best[2]


def parse_dicom_age(s):
    m = re.fullmatch(r"(\d{1,3})([YMWD])", (s or "").strip().upper())
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2)
    return {"Y": n, "M": n / 12, "W": n / 52, "D": n / 365}[u]


def parse_date(s):
    if not s:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    return None


# ------------------------------------------------------------------ engine
class Engine:
    def __init__(self, model_dir, device="cpu", max_jobs=1):
        self.dir = model_dir
        man_path = os.path.join(model_dir, "manifest.json")
        if not os.path.exists(man_path):
            raise RuntimeError(f"{man_path} missing - build it with deploy/package_models.py")
        self.manifest = json.load(open(man_path))
        for rel, digest in self.manifest["sha256"].items():  # integrity check of model files
            if sha256_file(os.path.join(model_dir, rel)) != digest:
                raise RuntimeError(f"model file {rel} failed its SHA-256 check - refusing to start")
        self.dev = torch.device(device)
        self.target = int(self.manifest.get("target", 96))
        self.classifiers = [load_classifier(os.path.join(model_dir, p), self.dev)
                            for p in self.manifest["classifiers"]]
        if any(m.task != "3class" or m.extra_feats for m in self.classifiers):
            raise RuntimeError("the web app serves 3-class image(+age/sex) models; package a model trained "
                               "with --task 3class and no --extra_feats")
        ck = torch.load(os.path.join(model_dir, self.manifest["world_model"]), map_location="cpu", weights_only=False)
        self.wm = BrainWorldModel(**ck["cfg"])
        self.wm.load_state_dict(ck["model"], strict=False)
        self.wm = self.wm.to(self.dev).eval()
        hd = self.manifest.get("header_detector")  # optional: DICOM header anomaly detector
        self.header_det = HeaderDetector.load(os.path.join(model_dir, hd)) if hd else None
        self.sem = threading.Semaphore(max_jobs)
        self.cfg = {"spacing": 1.5, "bet_f": 0.6, "brain_age_tol": float(os.environ.get("BRAIN_AGE_TOL", 12)),
                    "noise_copies": 4, "noise_sigma": 0.01,
                    "uncertainty_threshold": float(os.environ["UNC_THRESHOLD"]) if os.environ.get("UNC_THRESHOLD") else None}

    # ---------------------------------------------------------------- loading
    def load_upload(self, path, filename, workdir):
        name = filename.lower()
        if name.endswith((".nii", ".nii.gz")):
            dst = os.path.join(workdir, "scan.nii.gz" if name.endswith(".gz") else "scan.nii")
            shutil.copy(path, dst)
            try:
                img = read_image(dst, False)
            except RuntimeError:
                raise InputError("could not read the NIfTI file")
            return img, {"source": "nifti"}
        if name.endswith(".zip"):
            ex = os.path.join(workdir, "dicom")
            os.makedirs(ex)
            safe_extract(path, ex)
            series, hdr = find_t1_series(ex)
            try:
                img = read_image(series, True)
            except RuntimeError:
                raise InputError("could not read the DICOM series")
            hdr["source"] = "dicom"
            return img, hdr
        raise InputError("upload a .nii, .nii.gz or a .zip of one DICOM study")

    def preprocess(self, img):
        extent = [sz * sp for sz, sp in zip(img.GetSize(), img.GetSpacing())]
        if min(extent) < 100:
            raise InputError(f"scan covers only {min(extent):.0f} mm in one direction - incomplete series?")
        img = sitk.DICOMOrient(img, "RAS")
        img = n4_correct(img)
        img = resample_iso(img, self.cfg["spacing"])
        img = sitk.CurvatureAnisotropicDiffusion(img, timeStep=0.03, conductanceParameter=1.5, numberOfIterations=5)
        img = crop_neck(img)
        tool = pick_stripper("auto")
        if tool in ("synthstrip", "hdbet"):
            m = external_mask(img, tool)
        elif tool == "bet":
            m = bet_mask(img, self.cfg["bet_f"])
        else:
            m = otsu_mask(img)
        arr = sitk.GetArrayFromImage(img)
        mask = sitk.GetArrayFromImage(m).astype(bool)
        brain_ml = float(mask.sum() * np.prod(img.GetSpacing()) / 1000)
        if tool != "otsu":
            arr = arr * mask
        arr, mask = crop_pad_resize(arr, mask, self.target)
        return normalize(arr, mask), {"skull_strip": tool, "mask_ml": round(brain_ml)}

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def _probs(self, x, d):
        ps = []
        for m in self.classifiers:
            p = F.softmax(m(x, d), 1) + F.softmax(m(x.flip(-1), d), 1)
            ps.append(p / 2)
        return torch.stack(ps).mean(0)[0].cpu().numpy()

    def analyze(self, vol, age, sex):
        x = torch.from_numpy(vol)[None, None].to(self.dev)
        demo = torch.from_numpy(encode_action(0.0, age if age is not None else np.nan, sex)[1:])[None].to(self.dev)
        probs = self._probs(x, demo)
        pred = int(probs.argmax())

        # stability under tiny noise (cheap screen for adversarial / borderline inputs)
        agree = 0
        g = torch.Generator().manual_seed(0)
        for _ in range(self.cfg["noise_copies"]):
            n = torch.randn(x.shape, generator=g).to(self.dev) * self.cfg["noise_sigma"] * (x > 0)
            agree += int(self._probs((x + n).clamp(min=0), demo).argmax() == pred)
        stability = agree / self.cfg["noise_copies"]

        with torch.no_grad():
            tok, _ = self.wm.encode(x)
            brain_age = float(self.wm.brain_age(tok)[0] * 8 + 73)
            forecast = []
            for h in (1.0, 2.0, 3.0):
                r = TA.readouts(self.wm, x, TA.make_action(demo, h))
                forecast.append({"years": h, "predicted_atrophy": float(r["pred_atrophy"][0]),
                                 "uncertainty": float(r["uncertainty"][0])})

        cam = gradcam(self.classifiers[0], x.clone(), demo, pred)
        return {"probabilities": {l: float(p) for l, p in zip(LABELS, probs)}, "prediction": LABELS[pred],
                "stability": stability, "brain_age": round(brain_age, 1), "forecast": forecast,
                "bpf": round(brain_parenchymal_fraction(vol), 4), "figure": self.figure(vol, cam, LABELS[pred], probs[pred])}

    @staticmethod
    def figure(vol, cam, label, p):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = [s // 2 for s in vol.shape]
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.2), facecolor="#14213D")
        for a, (v, h) in zip(ax, [(vol[c[0]], cam[c[0]]), (vol[:, c[1]], cam[:, c[1]]), (vol[:, :, c[2]], cam[:, :, c[2]])]):
            a.imshow(np.flipud(v), cmap="gray", vmin=0, vmax=1)
            a.imshow(np.flipud(h), cmap="inferno", alpha=0.35)
            a.axis("off")
        fig.suptitle(f"Grad-CAM for {label} ({p:.2f})", color="#F2F1EC", fontsize=11)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, facecolor=fig.get_facecolor())
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # ---------------------------------------------------------------- checks
    def checks(self, header, form, used, result):
        out = []

        def add(name, status, detail):
            out.append({"check": name, "status": status, "detail": detail})

        age, sex, d = used["age"], used["sex"], used["date"]
        if age is None:
            add("Metadata present", "warn", "No age given or in the header - classified with population-average age")
        elif not 40 <= age <= 100:
            add("Age plausible", "fail", f"Age {age:g} is outside 40-100")
        else:
            add("Age plausible", "ok", f"Age {age:g}")
        if sex not in ("M", "F"):
            add("Sex present", "warn", "Sex missing or not M/F - treated as unknown")
        if d and d > date.today():
            add("Scan date", "fail", f"Scan date {d} is in the future")

        # header vs form: two sources that disagree (Paper 1: inconsistent metadata)
        h_age = parse_dicom_age(header.get("age"))
        if form.get("age") is not None and h_age is not None and abs(form["age"] - h_age) > 1:
            add("Header vs form", "warn", f"Form says age {form['age']:g}, DICOM header says {h_age:g}")
        h_sex = (header.get("sex") or "").upper()[:1]
        if form.get("sex") and h_sex in ("M", "F") and form["sex"] != h_sex:
            add("Header vs form", "warn", f"Form says sex {form['sex']}, DICOM header says {h_sex}")
        if header.get("source") == "dicom" and header.get("age") is None:
            add("DICOM header", "warn", "The header has no PatientAge field")

        # header anomaly detector (rules + Isolation Forest + autoencoder), DICOM uploads only
        if self.header_det is not None:
            if header.get("raw") is None:
                add("Header anomaly detector", "warn", "Not run - NIfTI files carry no DICOM header")
            else:
                r = self.header_det.check(with_form(header["raw"], form.get("age"), form.get("sex"), form.get("date")))
                if r["anomalous"]:
                    why = "; ".join(r["rules"][:3]) or "unusual combination of header values"
                    add("Header anomaly detector", "warn",
                        f"Flagged by {', '.join(r['flagged_by'])}: {why} - check the scan's origin")
                else:
                    add("Header anomaly detector", "ok",
                        f"Header looks like the training scans (IF {r['iforest']:.2f} <= {r['iforest_thr']:.2f}, "
                        f"AE {r['ae']:.3f} <= {r['ae_thr']:.3f})")

        # image vs metadata: brain age
        if age is not None:
            gap = abs(result["brain_age"] - age)
            st = "ok" if gap <= self.cfg["brain_age_tol"] else "warn"
            add("Brain age vs stated age", st, f"Image looks {result['brain_age']:.0f}, stated {age:g} (gap {gap:.0f} y)")

        st = "ok" if result["stability"] == 1 else "warn"
        add("Stability under tiny noise", st, f"{result['stability']:.0%} of noisy copies kept the same prediction"
            + ("" if st == "ok" else " - possible tampered or borderline scan, refer to a radiologist"))
        unc = result["forecast"][-1]["uncertainty"]
        thr = self.cfg["uncertainty_threshold"]
        if thr is not None:
            add("Forecast uncertainty", "ok" if unc <= thr else "warn", f"{unc:.3f} (threshold {thr:.3f})")
        lo, hi = 900, 2000
        ml = used["mask_ml"]
        add("Skull-strip volume", "ok" if lo <= ml <= hi else "warn", f"{ml} ml (expected {lo}-{hi})")
        return out

    # ---------------------------------------------------------------- top level
    def run(self, upload_path, filename, form):
        t0 = time.time()
        with self.sem, tempfile.TemporaryDirectory() as wd:
            img, header = self.load_upload(upload_path, filename, wd)
            vol, pre = self.preprocess(img)
            age = form.get("age") if form.get("age") is not None else parse_dicom_age(header.get("age"))
            sex = form.get("sex") or ((header.get("sex") or "").upper()[:1] or None)
            d = form.get("date") or parse_date(header.get("date"))
            used = {"age": age, "sex": sex, "date": d, "mask_ml": pre["mask_ml"]}
            res = self.analyze(vol.astype(np.float32), age, sex)
            res["checks"] = self.checks(header, form, used, res)
        res["input"] = {"source": header.get("source"), "series": header.get("description"),
                        "manufacturer": header.get("manufacturer"), "field_strength": header.get("field_strength"),
                        "age_used": age, "sex_used": sex, "date_used": d.isoformat() if d else None,
                        "skull_strip": pre["skull_strip"]}
        res["model"] = {"version": self.manifest.get("version"), "n_classifiers": len(self.classifiers),
                        "trained_on": self.manifest.get("trained_on")}
        res["seconds"] = round(time.time() - t0, 1)
        return res
