"""Metadata attacks on the diagnosis classifier (Paper 1, RQ-M1 / RQ-M3).

Metadata = demo = [age_n, sex], age_n = (years - 73) / 8, sex: M=0, F=1, unknown=0.5.
In a real deployment these come from DICOM tags PatientAge (0010,1010),
PatientBirthDate (0010,0030) and PatientSex (0010,0040), which anyone with write
access to a file or the network path can edit without touching the pixels.

Attack families (taxonomy shared with the CT metadata project):
  numerical      age_pgd        shift age within +-budget years (worst case, PGD)
  categorical    sex_flip       flip sex
  missingness    missing        remove age and sex
  inconsistency  inconsistent   age that contradicts the image (far from population mean)
  joint          joint_pgd      pixels (eps) + age (years) optimised together
"""
import numpy as np
import torch
import torch.nn.functional as F

from adversarial import _EvalMode, _project, brain_mask

AGE_SCALE = 8.0  # years per normalised unit


def age_pgd(model, x, demo, y, budget_years, steps=10):
    budget = budget_years / AGE_SCALE
    alpha = 2.5 * budget / steps
    delta = torch.zeros_like(demo[:, 0]).uniform_(-budget, budget)
    with _EvalMode(model), torch.enable_grad():
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            d = demo.clone()
            d[:, 0] = demo[:, 0] + delta
            g, = torch.autograd.grad(F.cross_entropy(model(x, d), y), delta)
            delta = (delta + alpha * g.sign()).clamp(-budget, budget)
    d = demo.clone()
    d[:, 0] = demo[:, 0] + delta.detach()
    return d


def sex_flip(demo):
    d = demo.clone()
    d[:, 1] = 1.0 - d[:, 1]
    return d


def missing(demo):
    d = demo.clone()
    d[:, 0], d[:, 1] = 0.0, 0.5
    return d


def inconsistent(demo, shift_years=20.0):
    """Push the stated age far to the other side of the population mean."""
    d = demo.clone()
    sign = torch.where(d[:, 0] >= 0, -1.0, 1.0)
    d[:, 0] = d[:, 0] + sign * shift_years / AGE_SCALE
    return d


def joint_pgd(model, x, demo, y, eps, budget_years, steps=10):
    mask = brain_mask(x)
    budget = budget_years / AGE_SCALE
    a_px, a_age = 2.5 * eps / steps, 2.5 * budget / steps
    x_adv = _project(x + torch.empty_like(x).uniform_(-eps, eps) * mask, x, eps, mask)
    delta = torch.zeros_like(demo[:, 0])
    with _EvalMode(model), torch.enable_grad():
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            delta = delta.detach().requires_grad_(True)
            d = demo.clone()
            d[:, 0] = demo[:, 0] + delta
            gx, gd = torch.autograd.grad(F.cross_entropy(model(x_adv, d), y), [x_adv, delta])
            x_adv = _project(x_adv + a_px * gx.sign(), x, eps, mask)
            delta = (delta + a_age * gd.sign()).clamp(-budget, budget)
    d = demo.clone()
    d[:, 0] = demo[:, 0] + delta.detach()
    return x_adv.detach(), d


def meta_adversarial_batch(model, x, demo, y, budget_years, p_flip=0.25, steps=3):
    """Training-time worst-case metadata (for --meta_adv): PGD on age + random sex flips."""
    d = age_pgd(model, x, demo, y, budget_years, steps)
    flip = torch.rand(d.shape[0], device=d.device) < p_flip
    d[flip, 1] = 1.0 - d[flip, 1]
    return d


def joint_adversarial_loss(model, x, demo, y, crit, eps, budget_years, steps=5, adv_ratio=0.5, p_flip=0.25):
    """Combined training (image + metadata): one PGD that moves pixels (inside the brain mask, L-inf eps)
    and age (+-budget_years) together, on top of random sex flips; the model learns from the clean
    and the jointly attacked version. Returns (loss, clean_logits, clean_feats)."""
    d0 = demo.clone()
    flip = torch.rand(d0.shape[0], device=d0.device) < p_flip
    d0[flip, 1] = 1.0 - d0[flip, 1]
    x_adv, d_adv = joint_pgd(model, x, d0, y, eps, budget_years, steps)
    logits, feats = model(x, demo, return_feats=True)
    loss = (1 - adv_ratio) * crit(logits, y) + adv_ratio * crit(model(x_adv, d_adv), y)
    return loss, logits, feats


@torch.no_grad()
def minimal_flip_budget(model, x, demo, y, grid, attack):
    """Smallest budget in `grid` that flips each sample's prediction (inf if none).

    attack(budget) -> (x_adv, demo_adv). Returns a numpy array (B,).
    """
    pred0 = model(x, demo).argmax(1)
    out = np.full(x.shape[0], np.inf)
    for b in grid:
        xa, da = attack(b)
        flipped = (model(xa, da).argmax(1) != pred0).cpu().numpy()
        out[np.isinf(out) & flipped] = b
        if not np.isinf(out).any():
            break
    return out
