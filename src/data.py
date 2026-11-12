"""Datasets + 3D augmentations (pure PyTorch, no MONAI needed)."""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from common import LABEL2IDX, encode_action, load_volume


# ----------------------------------------------------------------- augment
class Augment3D:
    """Random affine, L-R flip, bias field, gamma, noise. Input (1,D,H,W) in [0,1]."""

    def __init__(self, rot_deg=10, scale=(0.9, 1.1), shift=0.05, p_affine=0.8, p_flip=0.5,
                 p_bias=0.3, p_gamma=0.3, p_noise=0.3):
        self.__dict__.update(locals())

    def _affine(self, x):
        r = [math.radians(np.random.uniform(-self.rot_deg, self.rot_deg)) for _ in range(3)]
        cx, cy, cz = (math.cos(a) for a in r)
        sx, sy, sz = (math.sin(a) for a in r)
        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        A = (Rz @ Ry @ Rx) / np.random.uniform(*self.scale)
        t = torch.empty(3, 1).uniform_(-self.shift * 2, self.shift * 2)
        theta = torch.cat([A, t], 1).float()[None]
        grid = F.affine_grid(theta, x[None].shape, align_corners=False)
        return F.grid_sample(x[None], grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False)[0]

    def _bias(self, x):
        coarse = torch.randn(1, 1, 4, 4, 4) * 0.25
        field = F.interpolate(coarse, size=x.shape[1:], mode="trilinear", align_corners=True)[0]
        return x * torch.exp(field)

    def __call__(self, x):
        if np.random.rand() < self.p_affine:
            x = self._affine(x)
        if np.random.rand() < self.p_flip:
            x = x.flip(-1)  # last axis = left-right after RAS reorientation
        if np.random.rand() < self.p_bias:
            x = self._bias(x)
        if np.random.rand() < self.p_gamma:
            x = x.clamp(min=0) ** np.random.uniform(0.7, 1.5)
        if np.random.rand() < self.p_noise:
            x = x + torch.randn_like(x) * np.random.uniform(0, 0.04) * (x > 0)
        x = x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05) * (x > 0)
        return x


def brain_parenchymal_fraction(vol, tissue_thr=0.35, inside_thr=0.02):
    """BPF = tissue voxels / intracranial voxels (DIP: intensity thresholding).

    After skull stripping and [0,1] windowing, CSF is dark and GM/WM bright, so the
    fraction of bright voxels inside the head falls as the brain atrophies. BPF is a
    standard clinical atrophy measure; here it gives the world model a measurable
    readout of predicted future atrophy.
    """
    v = np.asarray(vol, dtype=np.float32)
    inside = v > inside_thr
    return float((v > tissue_thr).sum() / max(1, inside.sum()))


def _age_known(r):
    a = r.get("age")
    return float(a is not None and not (isinstance(a, float) and np.isnan(a)))


def _to_tensor(vol):
    return torch.from_numpy(np.asarray(vol, dtype=np.float32))[None]


# ----------------------------------------------------------------- datasets
class ScanDataset(Dataset):
    """One scan -> (volume, label, demographics). Used for classification."""

    def __init__(self, df, aug=None, key=None):
        self.df = df.reset_index(drop=True)
        self.aug, self.key = aug, key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        vol = load_volume(r.npy, self.key)
        x = _to_tensor(vol)
        if self.aug:
            x = self.aug(x)
        demo = encode_action(0.0, r.get("age"), r.get("sex"))[1:]  # [age, sex]
        return {"x": x, "y": LABEL2IDX[r.label], "demo": torch.from_numpy(demo), "idx": i,
                "bpf": torch.tensor(brain_parenchymal_fraction(vol)),
                "age_known": torch.tensor(_age_known(r))}


class LongitudinalPairDataset(Dataset):
    """(scan_t, action, scan_t+dt) transitions for training the world model.

    Every ordered pair of visits of the same subject within max_gap years is a
    transition. Every scan also appears as a dt=0 self-transition so single-visit
    subjects still contribute to the JEPA objective.
    """

    def __init__(self, df, aug=None, key=None, max_gap=6.0):
        df = df.copy()
        df["acq_date"] = pd.to_datetime(df.acq_date, errors="coerce")
        self.df = df.reset_index(drop=True)
        self.aug, self.key = aug, key
        pairs = [(i, i, 0.0) for i in range(len(self.df))]
        for _, g in self.df.groupby("subject"):
            g = g.sort_values("acq_date")
            idx, dates = g.index.tolist(), g.acq_date.tolist()
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    if pd.isna(dates[a]) or pd.isna(dates[b]):
                        continue
                    dt = (dates[b] - dates[a]).days / 365.25
                    if 0 < dt <= max_gap:
                        pairs.append((idx[a], idx[b], dt))
        self.pairs = pairs
        self.n_transitions = sum(p[2] > 0 for p in pairs)

    def __len__(self):
        return len(self.pairs)

    def _load(self, i):
        vol = load_volume(self.df.npy[i], self.key)
        x = _to_tensor(vol)
        return (self.aug(x) if self.aug else x), brain_parenchymal_fraction(vol)

    def __getitem__(self, k):
        i, j, dt = self.pairs[k]
        r = self.df.iloc[i]
        act = encode_action(dt, r.get("age"), r.get("sex"))
        x_t, bpf_t = self._load(i)
        x_f, bpf_f = self._load(j)
        return {"x_t": x_t, "x_f": x_f, "action": torch.from_numpy(act),
                "is_transition": torch.tensor(float(dt > 0)),
                "bpf_t": torch.tensor(bpf_t), "bpf_f": torch.tensor(bpf_f),
                "age_known": torch.tensor(_age_known(r))}


def load_split(index_csv, splits, fold=None, part="train"):
    df = pd.read_csv(index_csv)
    if part == "test":
        subs = set(splits["test"])
    elif part == "dev":
        subs = set(df.subject) if splits.get("smoke") else set(df.subject) - set(splits["test"])
    else:
        subs = set(splits["folds"][fold][part])
    out = df[df.subject.isin(subs)].reset_index(drop=True)
    if out.empty:
        raise SystemExit(f"split '{part}' is empty - check {index_csv} against the splits file")
    return out
