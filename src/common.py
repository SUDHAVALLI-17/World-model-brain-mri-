"""Shared helpers: label maps, seeding, device, logging, volume loading."""
import logging
import os
import random
import sys

import numpy as np
import pandas as pd

LABELS = ["CN", "MCI", "AD"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

# Classification tasks. The index keeps the visit-level label (CN / MCI / AD); a task maps it to
# class indices and drops the labels it does not use. On OASIS the labels come from CDR, so
# CN = CDR 0, MCI = CDR 0.5 ("very mild dementia" in OASIS, NOT a clinical MCI diagnosis),
# AD = CDR >= 1 - report the classes under the CDR names.
TASKS = {
    "3class": {"classes": ["CN", "MCI", "AD"], "map": {"CN": 0, "MCI": 1, "AD": 2},
               "oasis_names": ["CDR 0", "CDR 0.5", "CDR >=1"]},
    "cdr_binary": {"classes": ["CN", "Impaired"], "map": {"CN": 0, "MCI": 1, "AD": 1},
                   "oasis_names": ["CDR 0", "CDR >=0.5"]},
    "cn_vs_ad": {"classes": ["CN", "AD"], "map": {"CN": 0, "AD": 1},
                 "oasis_names": ["CDR 0", "CDR >=1"]},
}


def task_frame(df, task):
    """Rows of df used by `task` (e.g. cn_vs_ad drops MCI) and their class indices."""
    m = TASKS[task]["map"]
    d = df[df.label.isin(m)].reset_index(drop=True)
    return d, d.label.map(m).values.astype(int)


# Extra scalar inputs for the classifier head (z-scored with TRAINING-fold statistics).
#   vol_ratio = our skull-stripped brain volume / OASIS eTIV: a normalised whole-brain volume that the
#               per-brain crop + resize removes from the image itself
#   nwbv      = OASIS's own normalised whole-brain volume (from their processing; a reference only)
#   tissue_ratio = our brain-TISSUE volume / eTIV (add_tissue_volume.py): unlike vol_ratio it excludes the
#               CSF inside the mask, which is where atrophy shows
EXTRA_FEATS = ("vol_ratio", "nwbv", "tissue_ratio")


def add_extra_columns(df):
    df = df.copy()
    if "mask_ml" in df and "etiv" in df:
        df["vol_ratio"] = pd.to_numeric(df.mask_ml, errors="coerce") / pd.to_numeric(df.etiv, errors="coerce")
    if "nwbv" in df:
        df["nwbv"] = pd.to_numeric(df.nwbv, errors="coerce")
    if "tissue_ml" in df and "etiv" in df:
        df["tissue_ratio"] = pd.to_numeric(df.tissue_ml, errors="coerce") / pd.to_numeric(df.etiv, errors="coerce")
    return df


def extra_stats(df, feats):
    """Mean / sd of each extra feature on the training rows (missing values ignored)."""
    out = {}
    for f in feats:
        if f not in df or df[f].notna().sum() < 2:
            raise SystemExit(f"extra feature {f!r} is missing from the index (vol_ratio needs mask_ml + etiv; "
                             "tissue_ratio needs add_tissue_volume.py)")
        out[f] = [float(df[f].mean()), float(df[f].std() + 1e-8)]
    return out


def extra_matrix(df, feats, stats):
    """z-scored extra features; a missing value becomes 0 (= the training mean)."""
    if not feats:
        return np.zeros((len(df), 0), np.float32)
    cols = [((df[f] - stats[f][0]) / stats[f][1]).fillna(0.0).values for f in feats]
    return np.stack(cols, 1).astype(np.float32)

# ADNI uses several names for the same diagnostic group across phases / files.
GROUP_MAP = {
    "CN": "CN", "NL": "CN", "SMC": "CN", "NORMAL": "CN",
    "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI",
    "AD": "AD", "DEMENTIA": "AD",
}


def map_group(g):
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return None
    return GROUP_MAP.get(str(g).strip().upper())


def set_seed(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name, out_dir=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(out_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_volume(path, key=None):
    """Load a preprocessed volume (.npy, or AES-encrypted .npy.enc)."""
    if path.endswith(".enc"):
        from secure_io import load_encrypted
        return load_encrypted(path, key)
    return np.load(path)


def encode_action(dt_years, age, sex):
    """Action vector fed to the world model's dynamics: [time step, age, sex]."""
    age = 73.0 if age is None or np.isnan(age) else float(age)
    s = str(sex).upper()[:1] if sex is not None else ""
    sex_v = 0.0 if s == "M" else 1.0 if s == "F" else 0.5
    return np.array([dt_years / 2.0, (age - 73.0) / 8.0, sex_v], dtype=np.float32)
