"""Adversarial training + adversarial debiasing for the world-model classifier.

1. Robustness (gradient-based attacks in image space)
   - FGSM  : one signed-gradient step (Goodfellow et al., 2015)
   - PGD   : k projected steps with random start (Madry et al., 2018)
   - TRADES: clean CE + beta * KL(adv || clean) (Zhang et al., 2019)
   Perturbations are L-inf bounded (eps, in [0,1] intensity units) and applied
   only inside the brain (voxels > 0), so the background stays black, which is
   a more realistic threat model for MRI than perturbing empty air.

2. Debiasing (domain-adversarial training, Ganin et al., 2016)
   A gradient-reversal layer feeds the classifier's features to an adversary
   that tries to predict a nuisance attribute (ADNI acquisition site by default,
   or sex). The encoder is pushed to make that attribute unpredictable, so the
   diagnosis cannot rely on scanner/site shortcuts.
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ attacks
def brain_mask(x):
    return (x > 0).float()


class _EvalMode:
    """Generate attacks with dropout off, then restore the previous mode."""

    def __init__(self, model):
        self.model, self.was_training = model, model.training

    def __enter__(self):
        self.model.eval()

    def __exit__(self, *a):
        self.model.train(self.was_training)


def _project(x_adv, x, eps, mask):
    delta = (x_adv - x).clamp(-eps, eps) * mask
    return (x + delta).clamp(min=0)


def fgsm(model, x, demo, y, eps):
    mask = brain_mask(x)
    with _EvalMode(model), torch.enable_grad():
        x_adv = x.clone().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv, demo), y)
        g, = torch.autograd.grad(loss, x_adv)
    return _project(x + eps * g.sign(), x, eps, mask).detach()


def pgd(model, x, demo, y, eps, steps=5, alpha=None, random_start=True, p_clean=None):
    """L-inf PGD. If p_clean is given, maximise KL to it (TRADES inner loop) instead of CE."""
    alpha = alpha or 2.5 * eps / steps
    mask = brain_mask(x)
    x_adv = x + (torch.empty_like(x).uniform_(-eps, eps) * mask if random_start else 0)
    x_adv = _project(x_adv, x, eps, mask)
    with _EvalMode(model), torch.enable_grad():
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = model(x_adv, demo)
            if p_clean is None:
                loss = F.cross_entropy(logits, y)
            else:
                loss = F.kl_div(F.log_softmax(logits, 1), p_clean, reduction="batchmean")
            g, = torch.autograd.grad(loss, x_adv)
            x_adv = _project(x_adv + alpha * g.sign(), x, eps, mask)
    return x_adv.detach()


def adversarial_loss(model, x, demo, y, crit, mode, eps, steps=5, adv_ratio=0.5, trades_beta=6.0):
    """Returns (loss, clean_logits, clean_feats). mode in {none, fgsm, pgd, trades}."""
    if mode == "none" or eps <= 0:
        logits, feats = model(x, demo, return_feats=True)
        return crit(logits, y), logits, feats
    if mode == "trades":
        logits, feats = model(x, demo, return_feats=True)
        p_clean = F.softmax(logits, 1).detach()
        x_adv = pgd(model, x, demo, y, eps, steps, p_clean=p_clean)
        kl = F.kl_div(F.log_softmax(model(x_adv, demo), 1), F.softmax(logits, 1), reduction="batchmean")
        return crit(logits, y) + trades_beta * kl, logits, feats
    x_adv = fgsm(model, x, demo, y, eps) if mode == "fgsm" else pgd(model, x, demo, y, eps, steps)
    logits, feats = model(x, demo, return_feats=True)
    loss = (1 - adv_ratio) * crit(logits, y) + adv_ratio * crit(model(x_adv, demo), y)
    return loss, logits, feats


def eps_schedule(epoch, eps, ramp_epochs, start_epoch=0):
    """Linearly ramp eps from 0 -> eps; training is unstable if full eps is used from step 0."""
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return eps
    return eps * min(1.0, (epoch - start_epoch + 1) / ramp_epochs)


# ---------------------------------------------------------------- debiasing
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class AttributeAdversary(nn.Module):
    """GRL -> MLP -> nuisance attribute (site / sex)."""

    def __init__(self, in_dim, n_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, n_out))
        self.lam = 0.0

    def forward(self, feats):
        return self.net(_GradReverse.apply(feats, self.lam))


def dann_lambda(progress, lam_max):
    """Ganin et al. schedule: 0 at start, smoothly -> lam_max."""
    return lam_max * (2 / (1 + math.exp(-10 * progress)) - 1)


def site_of(subject):
    """ADNI PTID '002_S_0295' -> site '002'."""
    return str(subject).split("_")[0]


def scanner_of(df):
    """Vendor + field strength, e.g. 'GE_1.5T', from the MRI key or IDA XML columns."""
    man = None
    for c in ("scanner_manufacturer", "manufacturer"):
        if c in df.columns:
            m = df[c].astype(str).str.upper().str.split().str[0]
            man = m if man is None else man.where(man.notna() & (man != "NAN"), m)
    fs = None
    for c in ("magnetic_field_strength", "field_strength"):
        if c in df.columns:
            f = pd.to_numeric(df[c], errors="coerce")
            fs = f if fs is None else fs.fillna(f)
    if man is None:
        raise ValueError("no scanner columns - index with --mri_key or --meta_xml_dir")
    fs = fs if fs is not None else pd.Series(np.nan, index=df.index)
    return (man.fillna("UNKNOWN").replace("NAN", "UNKNOWN") + "_" +
            fs.map(lambda v: f"{v:g}T" if pd.notna(v) else "?"))


def build_attribute_labels(df, attr, min_count=4):
    """Map each scan to an integer nuisance label; rare sites are pooled into 'other'."""
    if attr == "site":
        vals = df.subject.map(site_of)
        counts = vals.value_counts()
        vals = vals.where(vals.map(counts) >= min_count, "other")
    elif attr == "sex":
        vals = df.sex.fillna("U").astype(str).str.upper().str[:1]
    elif attr == "scanner":
        vals = scanner_of(df)
        counts = vals.value_counts()
        vals = vals.where(vals.map(counts) >= min_count, "other")
    else:
        raise ValueError(attr)
    if attr in ("site", "scanner") and vals.nunique() < 2:  # pooling merged everything: keep raw groups
        raw = df.subject.map(site_of) if attr == "site" else scanner_of(df)
        if raw.nunique() >= 2:
            vals = raw
    classes = sorted(vals.unique())
    return vals.map({c: i for i, c in enumerate(classes)}).values.astype(np.int64), classes
