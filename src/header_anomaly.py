"""DICOM header anomaly detection (Paper 1, metadata-tampering defence without a trusted hash).

A hash or HMAC catches every change to signed metadata, but only when a trusted signature exists.
For legacy or external scans, or tampering that happened before signing, the scan has to be judged
on its own. This module learns what clean ADNI T1 headers look like and flags headers that do not
fit, using three detectors and their combination:

  rules     structural/physical consistency (Larmor frequency vs field strength, TE < TR,
            header vs form, date order, plausible ranges, vendor-model pairs seen in training)
  iforest   Isolation Forest on numeric + categorical header features
  ae        small autoencoder; reconstruction error as anomaly score
  hybrid    rules OR iforest OR ae          (+ hash, when a trusted hash is available)

Thresholds are set on CLEAN validation headers only (target false-positive rate, default 5 %),
so the learned detectors never see an attack before evaluation.

  python src/header_anomaly.py extract  --index data/proc/index_proc.csv --out data/headers.csv
  python src/header_anomaly.py train    --headers data/headers.csv --splits data/splits.json \
         --fold 0 --out runs/header_anomaly
  python src/header_anomaly.py evaluate --headers data/headers.csv --splits data/splits.json \
         --detector runs/header_anomaly/header_detector.joblib --out runs/header_anomaly

`evaluate` applies controlled header attacks to held-out test headers and reports recall,
precision, F1, FPR and AUROC per attack and per detector (header_results.csv, header_summary.md,
header_recall_heatmap.png).
"""
import argparse
import copy
import hashlib
import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

GAMMA_MHZ_PER_T = 42.577  # proton gyromagnetic ratio

TAGS = {
    "manufacturer": "0008|0070", "model": "0008|1090", "field_strength": "0018|0087",
    "imaging_freq": "0018|0084", "scanning_seq": "0018|0020", "seq_variant": "0018|0021",
    "tr": "0018|0080", "te": "0018|0081", "ti": "0018|0082", "flip": "0018|1314",
    "slice_thickness": "0018|0050", "slice_spacing": "0018|0088", "pixel_spacing": "0028|0030",
    "rows": "0028|0010", "cols": "0028|0011", "sex": "0010|0040", "age": "0010|1010",
    "weight": "0010|1030", "study_date": "0008|0020", "series_date": "0008|0021",
    "acq_date": "0008|0022", "description": "0008|103e",
}
NUMERIC = ["field_strength", "imaging_freq", "tr", "te", "ti", "flip", "slice_thickness",
           "slice_spacing", "rows", "cols", "weight"]
DATES = ["study_date", "series_date", "acq_date"]
FORM = ["form_age", "form_sex", "form_date"]


# ------------------------------------------------------------------------------ parsing helpers
def _num(v):
    try:
        return float(str(v).split("\\")[0])
    except (TypeError, ValueError):
        return None


def _date(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (date, datetime)):
        return v if isinstance(v, date) else v.date()
    s = str(v).strip()[:10]
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s if fmt != "%Y%m%d" else s[:8], fmt).date()
        except ValueError:
            pass
    return None


def _age(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    if s[-1:] in ("Y", "M", "W", "D") and s[:-1].isdigit():
        n = int(s[:-1])
        return {"Y": n, "M": n / 12, "W": n / 52, "D": n / 365}[s[-1]]
    return _num(s)


def _sex(v):
    s = (str(v).strip().upper()[:1] if v is not None else "")
    return s if s in ("M", "F") else None


def _vendor(v):
    s = (v or "").upper()
    for k in ("SIEMENS", "GE", "PHILIPS"):
        if k in s:
            return k
    return "OTHER" if s else None


def read_header(reader, n_slices=None):
    """Canonical header dict from a SimpleITK ImageFileReader after ReadImageInformation()."""
    g = {k: (reader.GetMetaData(t).strip() if reader.HasMetaDataKey(t) else None) for k, t in TAGS.items()}
    h = {k: (v if v not in ("",) else None) for k, v in g.items()}
    ps = (h.pop("pixel_spacing") or "").split("\\")
    h["pixel_row"], h["pixel_col"] = (_num(ps[0]), _num(ps[1])) if len(ps) == 2 else (None, None)
    for k in NUMERIC:
        h[k] = _num(h[k])
    h["age"] = _age(h["age"])
    h["sex"] = _sex(h["sex"])
    h["n_slices"] = n_slices
    return h


def read_series_header(series_dir):
    import SimpleITK as sitk
    files = sorted(f for f in os.listdir(series_dir) if f.lower().endswith(".dcm") or "." not in f)
    if not files:
        return None
    r = sitk.ImageFileReader()
    r.SetFileName(os.path.join(series_dir, files[0]))
    r.LoadPrivateTagsOff()
    r.ReadImageInformation()
    return read_header(r, len(files))


def with_form(h, age=None, sex=None, scan_date=None):
    """Attach the metadata that came with the scan (IDA form / web form) to a header dict."""
    h = dict(h)
    h["form_age"] = None if age is None or (isinstance(age, float) and np.isnan(age)) else float(age)
    h["form_sex"] = _sex(sex)
    d = _date(scan_date)
    h["form_date"] = d.isoformat() if d else None
    return h


# ------------------------------------------------------------------------------ hash baseline
def header_digest(h):
    """SHA-256 over every header + form field (the 'trusted hash available' baseline)."""
    return hashlib.sha256(json.dumps(h, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------------------ rules
def rule_violations(h, known_pairs=None, known_vendors=None):
    """Structural / physical consistency rules. Returns a list of (rule, message)."""
    v = []
    for k in ("manufacturer", "field_strength", "tr", "te", "acq_date", "sex"):
        if h.get(k) in (None, ""):
            v.append(("R1_required", f"{k} missing"))
    b0, f0 = h.get("field_strength"), h.get("imaging_freq")
    if b0 is not None and not 0.2 <= b0 <= 7.5:
        v.append(("R2_field", f"field strength {b0} T out of range"))
    if b0 and f0:
        ratio = f0 / (GAMMA_MHZ_PER_T * b0)
        if abs(ratio - 1) > 0.03:
            v.append(("R3_larmor", f"imaging frequency {f0:.1f} MHz does not match {b0} T"))
    tr, te = h.get("tr"), h.get("te")
    if tr and te and te >= tr:
        v.append(("R4_timing", f"TE {te} >= TR {tr}"))
    # no TI < TR rule: GE/Philips 3D IR-SPGR/MPRAGE store the echo spacing as TR (TR 8.9 ms, TI 1000 ms)
    for k, lo, hi in (("slice_thickness", 0.4, 3.5), ("pixel_row", 0.4, 3.5), ("pixel_col", 0.4, 3.5),
                      ("flip", 1, 90), ("weight", 25, 250)):
        x = h.get(k)
        if x is not None and x > 0 and not lo <= x <= hi:
            v.append(("R5_range", f"{k} {x} outside {lo}-{hi}"))
    ds = {k: _date(h.get(k)) for k in DATES}
    if ds["study_date"] and ds["acq_date"] and abs((ds["acq_date"] - ds["study_date"]).days) > 1:
        v.append(("R6_dates", "acquisition date differs from study date"))
    if ds["acq_date"] and ds["acq_date"] > date.today():
        v.append(("R6_dates", "acquisition date in the future"))
    fd = _date(h.get("form_date"))
    if fd and ds["acq_date"] and abs((fd - ds["acq_date"]).days) > 31:
        v.append(("R7_header_vs_form", f"form date {fd} vs header {ds['acq_date']}"))
    if h.get("form_sex") and h.get("sex") and h["form_sex"] != h["sex"]:
        v.append(("R7_header_vs_form", f"form sex {h['form_sex']} vs header {h['sex']}"))
    fa, ha = h.get("form_age"), h.get("age")
    if fa is not None and ha is not None and abs(fa - ha) > 1.5:
        v.append(("R7_header_vs_form", f"form age {fa:g} vs header {ha:g}"))
    if fa is not None and not 40 <= fa <= 100:
        v.append(("R8_age", f"age {fa:g} outside 40-100"))
    if known_pairs is not None and h.get("manufacturer") and h.get("model"):
        if (_vendor(h["manufacturer"]), str(h["model"]).upper()) not in known_pairs:
            v.append(("R9_vendor_model", f"{h['manufacturer']} / {h['model']} never seen in training"))
    if known_vendors is not None and h.get("manufacturer") and _vendor(h["manufacturer"]) not in known_vendors:
        v.append(("R9_vendor_model", f"unknown vendor {h['manufacturer']}"))
    return v


# ------------------------------------------------------------------------------ features
class FeatureSpec:
    """Turns header dicts into a fixed numeric vector; fitted on clean training headers."""

    def fit(self, H):
        df = pd.DataFrame(H)
        self.vendors = sorted({_vendor(x) for x in df.manufacturer if _vendor(x)})
        self.seqs = sorted({str(x) for x in df.scanning_seq.dropna()})[:8]
        self.pairs = {(_vendor(m), str(md).upper()) for m, md in zip(df.manufacturer, df.model) if m and md}
        raw = np.array([self._raw(h) for h in H], dtype=float)
        self.median = np.nanmedian(raw, axis=0)
        self.median = np.where(np.isnan(self.median), 0.0, self.median)
        X = self._fill(raw)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def _raw(self, h):
        b0, f0 = h.get("field_strength"), h.get("imaging_freq")
        ds = {k: _date(h.get(k)) for k in DATES}
        fd = _date(h.get("form_date"))
        gap = lambda a, b: abs((a - b).days) if a and b else np.nan  # noqa: E731
        lg = lambda x: np.log(x) if x and x > 0 else np.nan  # noqa: E731
        nsl, sp = h.get("n_slices"), h.get("slice_spacing") or h.get("slice_thickness")
        num = [lg(h.get("tr")), lg(h.get("te")), lg(h.get("ti")), h.get("flip"), h.get("slice_thickness"), sp,
               h.get("pixel_row"), h.get("pixel_col"), h.get("rows"), h.get("cols"), nsl,
               nsl * sp if nsl and sp else np.nan, b0,
               f0 / (GAMMA_MHZ_PER_T * b0) if b0 and f0 else np.nan, h.get("weight"), h.get("form_age"),
               (h["form_age"] - h["age"]) if h.get("form_age") is not None and h.get("age") is not None else 0.0,
               float(bool(h.get("form_sex") and h.get("sex") and h["form_sex"] != h["sex"])),
               gap(ds["acq_date"], ds["study_date"]), gap(fd, ds["acq_date"])]
        num = [np.nan if x is None else float(x) for x in num]
        miss = [float(h.get(k) in (None, "")) for k in ("tr", "te", "field_strength", "imaging_freq",
                                                        "sex", "acq_date", "manufacturer", "form_age")]
        v = _vendor(h.get("manufacturer"))
        cat = [float(v == x) for x in self.vendors] + [float(v not in self.vendors)]
        cat += [float(h.get("sex") == "M"), float(h.get("form_sex") == "M")]
        cat += [float(str(h.get("scanning_seq")) == s) for s in self.seqs]
        pair = (v, str(h.get("model") or "").upper())
        cat += [float(pair not in self.pairs)]
        return num + miss + cat

    def _fill(self, raw):
        return np.where(np.isnan(raw), self.median, raw)

    def transform(self, H):
        raw = np.array([self._raw(h) for h in H], dtype=float)
        return (self._fill(raw) - self.mu) / self.sd


# ------------------------------------------------------------------------------ detectors
def _ae_model(d):
    import torch.nn as nn
    h = max(4, min(32, d))
    return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, max(2, h // 4)), nn.ReLU(),
                         nn.Linear(max(2, h // 4), h), nn.ReLU(), nn.Linear(h, d))


class HeaderDetector:
    """rules + Isolation Forest + autoencoder; thresholds from clean validation headers."""

    def fit(self, H_train, H_val, target_fpr=0.05, seed=0, ae_epochs=400):
        import torch
        from sklearn.ensemble import IsolationForest
        self.target_fpr = target_fpr
        self.spec = FeatureSpec().fit(H_train)
        X = self.spec.transform(H_train)
        self.iforest = IsolationForest(n_estimators=300, random_state=seed).fit(X)
        torch.manual_seed(seed)
        ae = _ae_model(X.shape[1])
        opt = torch.optim.Adam(ae.parameters(), lr=3e-3, weight_decay=1e-4)
        xt = torch.tensor(X, dtype=torch.float32)
        for _ in range(ae_epochs):
            noisy = xt + 0.05 * torch.randn_like(xt)  # denoising AE: less identity mapping
            loss = ((ae(noisy) - xt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.ae_state = {k: v.detach().clone() for k, v in ae.state_dict().items()}
        self.dim = X.shape[1]
        calib = H_val if len(H_val) >= 20 else H_train + H_val
        self.calibrated_on = "val" if len(H_val) >= 20 else "train+val (val < 20 headers)"
        s = self.scores(calib)
        q = 1 - target_fpr
        self.thr = {"iforest": float(np.quantile(s["iforest"], q)), "ae": float(np.quantile(s["ae"], q))}
        self.n_train, self.n_calib = len(H_train), len(calib)
        return self

    def _ae(self):
        ae = _ae_model(self.dim)
        ae.load_state_dict(self.ae_state)
        return ae.eval()

    def scores(self, H):
        import torch
        X = self.spec.transform(H)
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32)
            ae = ((self._ae()(xt) - xt) ** 2).mean(1).numpy()
        rules = [rule_violations(h, self.spec.pairs, set(self.spec.vendors)) for h in H]
        return {"iforest": -self.iforest.score_samples(X), "ae": ae,
                "rules": np.array([len(r) for r in rules], dtype=float), "rule_msgs": rules}

    def flags(self, H, s=None):
        s = s or self.scores(H)
        f = {"rules": s["rules"] > 0, "iforest": s["iforest"] > self.thr["iforest"], "ae": s["ae"] > self.thr["ae"]}
        f["iforest+ae"] = f["iforest"] | f["ae"]
        f["rules+iforest"] = f["rules"] | f["iforest"]
        f["rules+ae"] = f["rules"] | f["ae"]
        f["hybrid"] = f["rules"] | f["iforest"] | f["ae"]
        return f, s

    def check(self, h):
        """Single header -> dict for the web app."""
        f, s = self.flags([h])
        return {"anomalous": bool(f["hybrid"][0]), "rules": [m for _, m in s["rule_msgs"][0]],
                "iforest": float(s["iforest"][0]), "iforest_thr": self.thr["iforest"],
                "ae": float(s["ae"][0]), "ae_thr": self.thr["ae"],
                "flagged_by": [k for k in ("rules", "iforest", "ae") if f[k][0]]}

    def save(self, path):
        import joblib
        joblib.dump({"version": 1, "obj": self}, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)["obj"]


# ------------------------------------------------------------------------------ attacks
def _shift_date(v, days):
    d = _date(v)
    if not d:
        return v
    nd = d + timedelta(days=days)
    return nd.strftime("%Y%m%d") if len(str(v).strip()) == 8 else nd.isoformat()


def _other(x, options):
    return [o for o in options if o != x][0]


ATTACKS = {
    # name: (function(h, rng) -> h, stealthy?)  stealthy = changes stay mutually consistent
    "age_shift_5y": (lambda h, r: {**h, "form_age": (h["form_age"] or 75) + r.choice([-5, 5])}, True),
    "age_shift_20y": (lambda h, r: {**h, "form_age": (h["form_age"] or 75) + r.choice([-20, 20])}, False),
    "age_implausible": (lambda h, r: {**h, "form_age": float(r.choice([17, 118]))}, False),
    "sex_flip_form": (lambda h, r: {**h, "form_sex": _other(h.get("form_sex") or "M", "MF")}, False),
    "sex_flip_consistent": (lambda h, r: {**h, "form_sex": _other(h.get("form_sex") or "M", "MF"),
                                          "sex": _other(h.get("sex") or "M", "MF")}, True),
    "date_inconsistent": (lambda h, r: {**h, "acq_date": _shift_date(h.get("acq_date"), 730)}, False),
    "date_shift_consistent": (lambda h, r: {**h, **{k: _shift_date(h.get(k), 730) for k in DATES + ["form_date"]}}, True),
    "field_strength_swap": (lambda h, r: {**h, "field_strength": 3.0 if (h.get("field_strength") or 3) < 2 else 1.5}, False),
    "vendor_swap": (lambda h, r: {**h, "manufacturer": _other(_vendor(h.get("manufacturer")) or "GE",
                                                              ["SIEMENS", "GE", "PHILIPS"])}, False),
    "protocol_tamper": (lambda h, r: {**h, "tr": (h.get("tr") or 2300) * 2.5, "te": (h.get("te") or 3) * 3}, False),
    "geometry_tamper": (lambda h, r: {**h, "slice_thickness": (h.get("slice_thickness") or 1.2) * 2,
                                      "pixel_row": (h.get("pixel_row") or 1) * 1.6,
                                      "pixel_col": (h.get("pixel_col") or 1) * 1.6}, False),
    "missing_fields": (lambda h, r: {**h, "sex": None, "field_strength": None, "acq_date": None}, False),
}


# ------------------------------------------------------------------------------ evaluation
def _metrics(y, flag, score=None):
    from sklearn.metrics import roc_auc_score
    tp = int((flag & y).sum()); fp = int((flag & ~y).sum()); fn = int((~flag & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"recall": rec, "precision": prec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
           "fpr": fp / max(1, int((~y).sum()))}
    if score is not None and len(set(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, score))
    return out


def evaluate(det, H_test, seed=0, repeats=3):
    rng = np.random.default_rng(seed)
    clean_f, clean_s = det.flags(H_test)
    clean_dig = [header_digest(h) for h in H_test]
    rows = []
    for name, (fn, stealthy) in ATTACKS.items():
        H_adv = [fn(copy.deepcopy(h), rng) for _ in range(repeats) for h in H_test]
        adv_f, adv_s = det.flags(H_adv)
        n_c, n_a = len(H_test), len(H_adv)
        y = np.r_[np.zeros(n_c, bool), np.ones(n_a, bool)]
        for k in adv_f:
            flag = np.r_[clean_f[k], adv_f[k]]
            sc = {"iforest": "iforest", "ae": "ae", "rules": "rules"}.get(k)
            m = _metrics(y, flag, np.r_[clean_s[sc], adv_s[sc]] if sc else None)
            rows.append({"attack": name, "stealthy": stealthy, "detector": k, **m})
        hash_flag = np.r_[np.zeros(n_c, bool),
                          np.array([header_digest(a) != clean_dig[i % n_c] for i, a in enumerate(H_adv)])]
        rows.append({"attack": name, "stealthy": stealthy, "detector": "hash (trusted)", **_metrics(y, hash_flag)})
        rows.append({"attack": name, "stealthy": stealthy, "detector": "hybrid+hash",
                     **_metrics(y, np.r_[clean_f["hybrid"], adv_f["hybrid"]] | hash_flag)})
    return pd.DataFrame(rows), clean_s


def write_report(df, clean_s, det, out):
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "header_results.csv"), index=False)
    order = ["rules", "iforest", "ae", "iforest+ae", "hybrid", "hash (trusted)", "hybrid+hash"]
    rec = df.pivot_table(index="attack", columns="detector", values="recall").reindex(columns=order)
    rec = rec.reindex(list(ATTACKS))
    fpr = df[df.attack == df.attack.iloc[0]].set_index("detector").fpr.reindex(order)
    f1 = df.groupby("detector").f1.mean().reindex(order)
    au = df.groupby("detector").auroc.mean().reindex(["rules", "iforest", "ae"]) if "auroc" in df else None
    stealth = df[df.stealthy].groupby("detector").recall.mean().reindex(order)
    obvious = df[~df.stealthy].groupby("detector").recall.mean().reindex(order)
    lines = ["# Header anomaly detection results", "",
             f"Train headers: {det.n_train}; thresholds from {det.calibrated_on} ({det.n_calib} headers) "
             f"at target FPR {det.target_fpr:.0%}. Each test header is attacked 3 times per attack.", "",
             "## Recall per attack (fraction of tampered headers flagged)", "",
             rec.round(2).to_markdown(), "",
             "## Summary", "",
             pd.DataFrame({"FPR on clean test": fpr, "mean F1": f1, "recall: obvious attacks": obvious,
                           "recall: stealthy attacks": stealth}).round(3).to_markdown(), ""]
    if au is not None:
        lines += ["Mean AUROC (score-based): " + ", ".join(f"{k} {v:.3f}" for k, v in au.items() if pd.notna(v)), ""]
    lines += ["Stealthy attacks keep the fields mutually consistent (small age shift, sex flipped in both",
              "header and form, all dates shifted together), so header-only rules have little to go on;",
              "compare them with the trusted hash/HMAC and with the image-based brain-age check.",
              "Clean-test false positives are often legitimate but rare scanners or protocols (rule R9 flags",
              "vendor/model pairs never seen in training) - list them from header_flags_clean.csv.", ""]
    open(os.path.join(out, "header_summary.md"), "w").write("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        im = ax.imshow(rec.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
        ax.set_yticks(range(len(rec)), rec.index)
        for i in range(rec.shape[0]):
            for j in range(rec.shape[1]):
                v = rec.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="black" if v > 0.6 else "white")
        fig.colorbar(im, ax=ax, label="recall")
        ax.set_title("Header tampering detection: recall per attack and detector")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "header_recall_heatmap.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print("\n".join(lines))


# ------------------------------------------------------------------------------ CLI
def load_headers(path):
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["image_id", "subject"]), []
    H = []
    for r in df.to_dict("records"):
        h = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        H.append(h)
    return df, H


def split_headers(df, H, splits_path, fold):
    s = json.load(open(splits_path))
    f = s["folds"][fold]
    pick = lambda subs: [h for h, sub in zip(H, df.subject) if sub in set(subs)]  # noqa: E731
    return pick(f["train"]), pick(f["val"]), pick(s["test"])


def cmd_extract(a):
    idx = pd.read_csv(a.index)
    rows, skipped = [], 0
    for r in idx.to_dict("records"):
        if not r.get("is_dicom", True) or not os.path.isdir(str(r["path"])):
            skipped += 1
            continue
        try:
            h = read_series_header(r["path"])
        except RuntimeError:
            h = None
        if h is None:
            skipped += 1
            continue
        h = with_form(h, r.get("age"), r.get("sex"), r.get("acq_date"))
        rows.append({"image_id": r["image_id"], "subject": r["subject"], **h})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} DICOM headers -> {a.out}  ({skipped} NIfTI/unreadable rows skipped)")


def cmd_train(a):
    df, H = load_headers(a.headers)
    tr, va, _ = split_headers(df, H, a.splits, a.fold) if len(df) else ([], [], [])
    if len(tr) < 5:
        print(f"only {len(tr)} DICOM headers in the training split - header detector skipped "
              "(NIfTI-only data has no DICOM headers)")
        return
    det = HeaderDetector().fit(tr, va, a.target_fpr, a.seed)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "header_detector.joblib")
    det.save(p)
    print(f"trained on {len(tr)} headers, thresholds {det.thr} ({det.calibrated_on}) -> {p}")


def cmd_evaluate(a):
    if not os.path.exists(a.detector):
        print(f"{a.detector} not found - header evaluation skipped")
        return
    df, H = load_headers(a.headers)
    _, _, te = split_headers(df, H, a.splits, a.fold)
    det = HeaderDetector.load(a.detector)
    res, clean_s = evaluate(det, te, a.seed)
    write_report(res, clean_s, det, a.out)
    test_rows = df[df.subject.isin(set(json.load(open(a.splits))["test"]))]
    pd.DataFrame({"image_id": test_rows.image_id.values, "iforest": clean_s["iforest"], "ae": clean_s["ae"],
                  "rules": ["; ".join(m for _, m in r) for r in clean_s["rule_msgs"]]}).to_csv(
        os.path.join(a.out, "header_flags_clean.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--index", required=True)
    e.add_argument("--out", default="data/headers.csv")
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--headers", required=True)
        p.add_argument("--splits", required=True)
        p.add_argument("--fold", type=int, default=0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default="runs/header_anomaly")
        if name == "train":
            p.add_argument("--target_fpr", type=float, default=0.05)
        else:
            p.add_argument("--detector", required=True)
    a = ap.parse_args()
    {"extract": cmd_extract, "train": cmd_train, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    # run through the importable module so pickled detectors load as header_anomaly.HeaderDetector
    import header_anomaly
    header_anomaly.main()
