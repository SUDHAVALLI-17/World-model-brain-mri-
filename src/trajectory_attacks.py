"""Attacks on the world model's *forecast* (Paper 1, RQ1 and RQ-M2).

The world model maps (scan x, action a = [dt/2, age_n, sex]) to a predicted future
global state mu(t+dt) with variance exp(logvar). Readouts:
  drift(h)          = || mu_h - g_0 ||                (how much change is predicted)
  pred_atrophy(h)   = v(g_0) - v(mu_h)                 (predicted loss of brain-parenchymal fraction)

Pixel attacks (image channel)
  hijack    : maximise || mu_h(x+d) - mu_h(x) ||^2 summed over horizons
  targeted  : push predicted atrophy up (direction=+1, "make a healthy brain look like a
              fast decliner") or down (direction=-1, "hide decline")
Action / metadata attacks (action channel)
  dt_attack : change the time step dt within +-budget years (a tampered study date)
  age_attack: change the stated age within +-budget years (a tampered DICOM age/birth date)
All attacks are white-box PGD with an L-inf budget; pixel perturbations stay inside the brain.
"""
import torch
import torch.nn.functional as F

from adversarial import _project, brain_mask


def make_action(demo, dt_years):
    """demo = [age_n, sex] (B,2) -> action [dt/2, age_n, sex]."""
    return torch.cat([torch.full_like(demo[:, :1], dt_years / 2.0), demo[:, :2]], 1)


def readouts(wm, x, action):
    g0, mu, logvar = wm.forecast(x, action)
    return {"g0": g0, "mu": mu, "logvar": logvar,
            "drift": (mu - g0).norm(dim=1),
            "pred_atrophy": wm.volume(g0) - wm.volume(mu),
            "uncertainty": logvar.exp().mean(1)}


def _pgd_pixels(wm, x, objective, eps, steps, alpha=None):
    alpha = alpha or 2.5 * eps / steps
    mask = brain_mask(x)
    x_adv = _project(x + torch.empty_like(x).uniform_(-eps, eps) * mask, x, eps, mask)
    was = wm.training
    wm.eval()
    with torch.enable_grad():
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            g, = torch.autograd.grad(objective(x_adv), x_adv)
            x_adv = _project(x_adv + alpha * g.sign(), x, eps, mask)
    wm.train(was)
    return x_adv.detach()


def hijack(wm, x, demo, horizons, eps, steps=10):
    """Untargeted: move the predicted futures as far as possible from the clean ones."""
    with torch.no_grad():
        clean = [wm.forecast(x, make_action(demo, h))[1] for h in horizons]

    def obj(xa):
        tok, grid = wm.encode(xa)
        return sum(((wm.forecast(xa, make_action(demo, h), tok, grid)[1] - c) ** 2).sum()
                   for h, c in zip(horizons, clean))
    return _pgd_pixels(wm, x, obj, eps, steps)


def targeted_atrophy(wm, x, demo, horizon, eps, steps=10, direction=+1):
    """direction=+1: inflate predicted atrophy (fake decline); -1: hide decline."""
    def obj(xa):
        g0, mu, _ = wm.forecast(xa, make_action(demo, horizon))
        return direction * (wm.volume(g0) - wm.volume(mu)).sum()
    return _pgd_pixels(wm, x, obj, eps, steps)


def action_attack(wm, x, demo, horizon, field, budget_years, steps=10, objective="atrophy", direction=+1):
    """PGD on one action field: 'dt' (study-date tampering) or 'age' (age tampering).

    budget_years is the largest change in years. Returns the attacked action (B,3).
    objective: 'atrophy' (push predicted atrophy in `direction`) or 'drift' (move mu away).
    """
    a0 = make_action(demo, horizon)
    idx, scale = (0, 2.0) if field == "dt" else (1, 8.0)  # action units -> years
    budget = budget_years / scale
    with torch.no_grad():
        tok, grid = wm.encode(x)
        g0 = wm.target_space(tok).mean(1)
        mu_clean = wm.dynamics(tok, a0, grid)[1]
    delta = torch.zeros_like(a0[:, idx]).uniform_(-budget, budget)
    alpha = 2.5 * budget / steps
    was = wm.training
    wm.eval()
    with torch.enable_grad():
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            a = a0.clone()
            a[:, idx] = a0[:, idx] + delta
            mu = wm.dynamics(tok, a, grid)[1]
            if objective == "atrophy":
                loss = direction * (wm.volume(g0) - wm.volume(mu)).sum()
            else:
                loss = ((mu - mu_clean) ** 2).sum()
            g, = torch.autograd.grad(loss, delta)
            delta = (delta + alpha * g.sign()).clamp(-budget, budget)
            if field == "dt":  # time cannot run backwards
                delta = torch.maximum(delta, -a0[:, idx])
    wm.train(was)
    a = a0.clone()
    a[:, idx] = a0[:, idx] + delta.detach()
    return a


def amplification(wm, x, x_adv, demo, horizons):
    """A(h) = ||mu_h(adv) - mu_h(clean)|| / ||g0(adv) - g0(clean)||  (error growth over horizon)."""
    with torch.no_grad():
        tc, gc = wm.encode(x)
        ta, ga = wm.encode(x_adv)
        base = (wm.target_space(ta).mean(1) - wm.target_space(tc).mean(1)).norm(dim=1).clamp(min=1e-6)
        out = {}
        for h in horizons:
            a = make_action(demo, h)
            d = (wm.forecast(x_adv, a, ta, ga)[1] - wm.forecast(x, a, tc, gc)[1]).norm(dim=1)
            out[h] = (d / base).cpu()
    return out


def trajectory_trades_example(wm, x, action, eps, steps=2):
    """Adversarial example used *during pretraining* (Trajectory-TRADES): maximise the
    change of the predicted future for the actual training action."""
    with torch.no_grad():
        mu_c = wm.forecast(x, action)[1]

    def obj(xa):
        return ((wm.forecast(xa, action)[1] - mu_c) ** 2).sum()
    return _pgd_pixels(wm, x, obj, eps, steps)
