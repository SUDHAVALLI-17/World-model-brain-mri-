"""Shared helpers: label maps, seeding, device, logging, volume loading."""
import logging
import os
import random
import sys

import numpy as np

LABELS = ["CN", "MCI", "AD"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

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
