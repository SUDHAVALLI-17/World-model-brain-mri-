"""Paper 1 evaluation: pixel + metadata attacks on an action-conditioned brain world model.

Produces one report organised by research question (held-out test subjects only):

  RQ1   Can an invisible pixel change alter the predicted future? Does the error grow
        with the forecast horizon (amplification)? Can it fake / hide decline?
  RQ-M1 Cost: smallest metadata change vs smallest pixel change that flips a diagnosis;
        accuracy under each metadata attack; does metadata make pixel attacks cheaper (joint)?
  RQ-M2 Action channel: how much does tampering with dt (study dates) or age move the
        forecast, compared with pixel attacks, and how does it scale with horizon?
  RQ-M3 Image-only vs image+metadata models under pixel attack (pass both as --run)
  RQ-M4 Defences: brain-age consistency check, forecast-uncertainty detector,
        plausibility rules vs HMAC (catch rate per attack), atrophy-monotonicity violations

  python src/paper1_eval.py --world_model runs/pretrain/last.pt \
      --run img+meta "runs/cls/fold*/best.pt" --run img_only "runs/cls_nometa/fold*/best.pt" \
      --out runs/paper1
"""
import argparse
import glob
import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import metadata_attacks as MA
import trajectory_attacks as TA
from adversarial import pgd
from common import LABEL2IDX, LABELS, get_device, get_logger, set_seed
from data import ScanDataset, load_split
from evaluate import load_classifier
from finetune_classify import metrics
from metadata_guard import catch_rates
from models import BrainWorldModel
from robustness_eval import Ensemble


def batches(ds, dev, bs=4):
    for b in DataLoader(ds, bs):
        yield b["x"].to(dev), b["demo"].to(dev), b["y"].to(dev), b


def auroc(clean_scores, adv_scores):
    y = np.r_[np.zeros(len(clean_scores)), np.ones(len(adv_scores))]
    s = np.r_[clean_scores, adv_scores]
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


# ----------------------------------------------------------- RQ-M1 / RQ-M3
def classification_section(name, ens, ds, dev, a, log):
    def run(make):
        P, Y = [], []
        for x, d, y, _ in batches(ds, dev):
            xa, da = make(x, d, y)
            with torch.no_grad():
                P.append(F.softmax(ens(xa, da), 1).cpu())
            Y.append(y.cpu())
        return metrics(torch.cat(Y).numpy(), torch.cat(P).numpy())["bal_acc"]

    rows = [("clean", "-", run(lambda x, d, y: (x, d)))]
    for eps in a.eps:
        rows.append(("pixel PGD", f"eps={eps:g}", run(lambda x, d, y, e=eps: (pgd(ens, x, d, y, e, a.steps), d))))
    for yrs in a.age_years:
        rows.append(("age PGD", f"+-{yrs:g}y", run(lambda x, d, y, b=yrs: (x, MA.age_pgd(ens, x, d, y, b, a.steps)))))
    rows.append(("sex flip", "-", run(lambda x, d, y: (x, MA.sex_flip(d)))))
    rows.append(("missing", "-", run(lambda x, d, y: (x, MA.missing(d)))))
    rows.append(("inconsistent age", "20y", run(lambda x, d, y: (x, MA.inconsistent(d)))))
    e0, y0 = a.joint_eps, a.joint_years
    rows.append(("pixel only", f"eps={e0:g}", run(lambda x, d, y: (pgd(ens, x, d, y, e0, a.steps), d))))
    rows.append(("age only", f"+-{y0:g}y", run(lambda x, d, y: (x, MA.age_pgd(ens, x, d, y, y0, a.steps)))))
    rows.append(("joint pixel+age", f"eps={e0:g},+-{y0:g}y",
                 run(lambda x, d, y: MA.joint_pgd(ens, x, d, y, e0, y0, a.steps))))
    df = pd.DataFrame(rows, columns=["attack", "budget", "bal_acc"]).assign(run=name)
    log.info("[%s] classification under attack:\n%s", name, df.to_string(index=False))

    # minimal flipping budget (per scan)
    mb_age, mb_px = [], []
    for x, d, y, _ in batches(ds, dev):
        mb_age.append(MA.minimal_flip_budget(ens, x, d, y, a.age_grid,
                                             lambda b: (x, MA.age_pgd(ens, x, d, y, b, a.steps))))
        mb_px.append(MA.minimal_flip_budget(ens, x, d, y, a.eps_grid,
                                            lambda b: (pgd(ens, x, d, y, b, a.steps), d)))
    mb_age, mb_px = np.concatenate(mb_age), np.concatenate(mb_px)
    budget = {"age_years_median": float(np.median(mb_age)), "age_flip_rate": float(np.isfinite(mb_age).mean()),
              "pixel_eps_median": float(np.median(mb_px)), "pixel_flip_rate": float(np.isfinite(mb_px).mean())}
    log.info("[%s] minimal flipping budget: %s", name, budget)
    return df, budget


# ---------------------------------------------------------------- RQ1 / RQ-M2
def trajectory_section(wm, ds, dev, a, log):
    H = a.horizons
    res, amp, unc_clean, unc_adv = [], {h: [] for h in H}, [], []
    mono_viol, labels = [], []
    for x, d, y, _ in batches(ds, dev):
        labels.append(y.cpu())
        with torch.no_grad():
            clean = {h: TA.readouts(wm, x, TA.make_action(d, h)) for h in H}
        pa = torch.stack([clean[h]["pred_atrophy"] for h in H], 1)
        mono_viol.append(((pa[:, 0] < -a.mono_tol) | (pa[:, 1:] < pa[:, :-1] - a.mono_tol).any(1)).cpu())
        unc_clean.append(clean[H[-1]]["uncertainty"].cpu())
        rec = {"y": y.cpu()}
        for h in H:
            rec[f"clean_atrophy_{h}"] = clean[h]["pred_atrophy"].cpu()
            rec[f"clean_drift_{h}"] = clean[h]["drift"].cpu()
        # pixel hijack (untargeted) at the largest eps
        eps = max(a.eps)
        xa = TA.hijack(wm, x, d, H, eps, a.steps)
        with torch.no_grad():
            for h in H:
                adv = TA.readouts(wm, xa, TA.make_action(d, h))
                rec[f"hijack_mu_shift_{h}"] = (adv["mu"] - clean[h]["mu"]).norm(dim=1).cpu()
                rec[f"hijack_atrophy_change_{h}"] = (adv["pred_atrophy"] - clean[h]["pred_atrophy"]).cpu()
            unc_adv.append(TA.readouts(wm, xa, TA.make_action(d, H[-1]))["uncertainty"].cpu())
        for h, v in TA.amplification(wm, x, xa, d, H).items():
            amp[h].append(v)
        # targeted: fake decline (+1) at middle horizon, pixel vs action channel
        hm = H[len(H) // 2]
        for direction, tag in ((+1, "fake_decline"), (-1, "hide_decline")):
            xt = TA.targeted_atrophy(wm, x, d, hm, eps, a.steps, direction)
            with torch.no_grad():
                rec[f"pixel_{tag}"] = (TA.readouts(wm, xt, TA.make_action(d, hm))["pred_atrophy"]
                                       - clean[hm]["pred_atrophy"]).cpu()
            for yrs in a.dt_years:
                act = TA.action_attack(wm, x, d, hm, "dt", yrs, a.steps, "atrophy", direction)
                with torch.no_grad():
                    g0, mu, _ = wm.forecast(x, act)
                    rec[f"dt{yrs:g}y_{tag}"] = (wm.volume(g0) - wm.volume(mu) - clean[hm]["pred_atrophy"]).cpu()
            for yrs in a.age_years:
                act = TA.action_attack(wm, x, d, hm, "age", yrs, a.steps, "atrophy", direction)
                with torch.no_grad():
                    g0, mu, _ = wm.forecast(x, act)
                    rec[f"age{yrs:g}y_{tag}"] = (wm.volume(g0) - wm.volume(mu) - clean[hm]["pred_atrophy"]).cpu()
        res.append(rec)
    cat = {k: torch.cat([r[k] for r in res]).numpy() for k in res[0]}
    y = cat.pop("y")

    out = {"monotonicity_violation_rate": float(torch.cat(mono_viol).float().mean()),
           "amplification_median": {h: float(torch.cat(amp[h]).median()) for h in H},
           "uncertainty_detector_auroc": auroc(torch.cat(unc_clean).numpy(), torch.cat(unc_adv).numpy())}
    for h in H:
        out[f"clean_pred_atrophy_{h}y"] = {LABELS[c]: float(cat[f"clean_atrophy_{h}"][y == c].mean())
                                          for c in np.unique(y)}
        out[f"hijack_mu_shift_{h}y"] = float(cat[f"hijack_mu_shift_{h}"].mean())
        out[f"hijack_atrophy_change_{h}y"] = float(np.abs(cat[f"hijack_atrophy_change_{h}"]).mean())
    hm = H[len(H) // 2]
    gap = None
    if 0 in y and 2 in y:  # CN-to-AD gap in predicted atrophy: how much of it can an attack fake?
        gap = float(cat[f"clean_atrophy_{hm}"][y == 2].mean() - cat[f"clean_atrophy_{hm}"][y == 0].mean())
        out[f"clean_AD_minus_CN_atrophy_{hm}y"] = gap
        if gap <= 0:
            log.warning("model does not predict more atrophy for AD than CN (gap %.4f) - the "
                        "'fraction of CN-AD gap' column is not meaningful; train longer / check data", gap)
            gap = None
    channel = []
    for tag in ("fake_decline", "hide_decline"):
        keys = [f"pixel_{tag}"] + [f"dt{v:g}y_{tag}" for v in a.dt_years] + [f"age{v:g}y_{tag}" for v in a.age_years]
        for k in keys:
            m = float(cat[k].mean())
            channel.append({"goal": tag, "attack": k.replace(f"_{tag}", ""), "mean_change_pred_atrophy": m,
                            "fraction_of_CN_AD_gap": (m / gap) if gap else None})
    channel = pd.DataFrame(channel)
    log.info("trajectory summary: %s", json.dumps(out, indent=1, default=str))
    log.info("pixel vs action channel at %sy horizon:\n%s", hm, channel.round(4).to_string(index=False))
    return out, channel, amp


def forecast_error_section(wm, df, key, dev, a, log):
    """Real follow-up visits in the test set: forecast error clean vs dt-tampered vs pixel."""
    df = df.copy()
    df["acq_date"] = pd.to_datetime(df.acq_date, errors="coerce")
    pairs = []
    for _, g in df.dropna(subset=["acq_date"]).sort_values("acq_date").groupby("subject"):
        # one scan per session (repeats on the same day are not follow-ups), then consecutive sessions
        g = g.drop_duplicates("acq_date")
        idx = g.index.tolist()
        for i, j in zip(idx[:-1], idx[1:]):
            dt = (df.acq_date[j] - df.acq_date[i]).days / 365.25
            if dt > 0.06:  # > ~3 weeks
                pairs.append((i, j, dt))
    if not pairs:
        log.info("no longitudinal pairs in test set - forecast-error section skipped")
        return None
    ds = ScanDataset(df, None, key)
    errs = {"clean": [], "pixel_hijack": [], **{f"dt_tamper_{v:g}y": [] for v in a.dt_years}}
    for i, j, dt in pairs:
        si, sj = ds[i], ds[j]
        x, d = si["x"][None].to(dev), si["demo"][None].to(dev)
        with torch.no_grad():
            tgt, _ = wm.encode(sj["x"][None].to(dev), target=True)
            g_f = wm.target_space(tgt).mean(1)
            act = TA.make_action(d, dt)
            errs["clean"].append(float((wm.forecast(x, act)[1] - g_f).norm()))
        xa = TA.hijack(wm, x, d, [dt], max(a.eps), a.steps)
        with torch.no_grad():
            errs["pixel_hijack"].append(float((wm.forecast(xa, act)[1] - g_f).norm()))
        for v in a.dt_years:
            at = TA.action_attack(wm, x, d, dt, "dt", v, a.steps, "drift")
            with torch.no_grad():
                errs[f"dt_tamper_{v:g}y"].append(float((wm.forecast(x, at)[1] - g_f).norm()))
    out = {k: float(np.mean(v)) for k, v in errs.items()}
    out["n_pairs"] = len(pairs)
    log.info("forecast error to the real follow-up (lower = better): %s", out)
    return out


def brain_age_section(wm, ds, dev, a, log):
    """Defence: |brain age (image) - stated age| flags inconsistent / tampered metadata."""
    res = {"clean": []}
    attacks = {"inconsistent_20y": lambda d: MA.inconsistent(d)}
    for yrs in a.age_years:
        attacks[f"age_shift_{yrs:g}y"] = (lambda d, s=yrs: d + torch.tensor([s / 8.0, 0.0], device=d.device)
                                          * torch.where(torch.rand(d.shape[0], 1, device=d.device) < .5, -1.0, 1.0))
    res.update({k: [] for k in attacks})
    known, mae = [], []
    for x, d, y, b in batches(ds, dev):
        with torch.no_grad():
            tok, _ = wm.encode(x)
            ba = wm.brain_age(tok)
        k = b["age_known"].to(dev) > 0
        known.append(k.cpu())
        res["clean"].append(((ba - d[:, 0]).abs() * 8).cpu())
        mae.append(((ba - d[:, 0]).abs() * 8)[k].cpu())
        for n, fn in attacks.items():
            res[n].append(((ba - fn(d)[:, 0]).abs() * 8).cpu())
    res = {k: torch.cat(v).numpy() for k, v in res.items()}
    out = {"brain_age_MAE_years": float(torch.cat(mae).mean()) if len(mae) else None}
    for n in attacks:
        out[f"auroc_{n}"] = auroc(res["clean"], res[n])
    log.info("brain-age consistency defence: %s", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--world_model", required=True, help="pretrain checkpoint (runs/pretrain/last.pt)")
    ap.add_argument("--run", nargs=2, action="append", metavar=("NAME", "CKPT_GLOB"), default=[])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument("--eps_grid", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02, 0.04])
    ap.add_argument("--age_years", type=float, nargs="+", default=[3, 5, 10])
    ap.add_argument("--age_grid", type=float, nargs="+", default=[0.5, 1, 2, 3, 5, 7, 10, 15])
    ap.add_argument("--dt_years", type=float, nargs="+", default=[0.5, 1, 2])
    ap.add_argument("--joint_eps", type=float, default=0.005)
    ap.add_argument("--joint_years", type=float, default=2.0)
    ap.add_argument("--horizons", type=float, nargs="+", default=[1, 2, 3])
    ap.add_argument("--mono_tol", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--guard_n", type=int, default=40)
    ap.add_argument("--out", default="runs/paper1")
    ap.add_argument("--key_file", default=None)
    a = ap.parse_args()
    set_seed(0)
    log = get_logger("paper1", a.out)
    dev = get_device()
    key = None
    if a.key_file:
        from secure_io import load_key
        key = load_key(a.key_file)

    te = load_split(a.index, json.load(open(a.splits)), part="test")
    ds = ScanDataset(te, None, key)
    log.info("test set: %d scans / %d subjects", len(te), te.subject.nunique())
    report = {}

    ck = torch.load(a.world_model, map_location="cpu", weights_only=False)
    wm = BrainWorldModel(**ck["cfg"])
    missing = wm.load_state_dict(ck["model"], strict=False).missing_keys
    if missing:
        log.warning("world model checkpoint lacks %s - retrain pretraining for Paper 1", missing[:4])
    wm = wm.to(dev).eval()

    traj, channel, amp = trajectory_section(wm, ds, dev, a, log)
    report["RQ1_and_RQ-M2_trajectory"] = traj
    channel.to_csv(os.path.join(a.out, "action_vs_pixel_channel.csv"), index=False)
    report["RQ-M2_forecast_error_real_followups"] = forecast_error_section(wm, te, key, dev, a, log)
    report["RQ-M4_brain_age_defence"] = brain_age_section(wm, ds, dev, a, log)

    gkey = key or os.urandom(32)
    kinds = ["age_shift", "sex_flip", "date_shift", "date_and_age_shift", "missing", "inconsistent_age"]
    guard = catch_rates(te, gkey, kinds, n=a.guard_n)
    guard.to_csv(os.path.join(a.out, "metadata_guard_catch_rates.csv"), index=False)
    log.info("metadata guard catch rates (plausibility rules vs HMAC):\n%s", guard.round(3).to_string(index=False))
    report["RQ-M4_guard_catch_rates"] = guard.to_dict("records")

    tables = []
    for name, pattern in a.run:
        paths = sorted(glob.glob(pattern))
        if not paths:
            log.warning("no checkpoints for %s", name)
            continue
        ens = Ensemble([load_classifier(p, dev) for p in paths]).to(dev).eval()
        df, budget = classification_section(name, ens, ds, dev, a, log)
        tables.append(df)
        report.setdefault("RQ-M1_minimal_flip_budget", {})[name] = budget
    if tables:
        t = pd.concat(tables)
        t["condition"] = t.attack + " " + t.budget.where(t.budget != "-", "")
        piv = t.pivot_table(index="condition", columns="run", values="bal_acc", sort=False)
        piv.to_csv(os.path.join(a.out, "classification_under_attack.csv"))
        log.info("RQ-M1 / RQ-M3 balanced accuracy:\n%s", piv.round(3).to_string())
        report["RQ-M1_RQ-M3_bal_acc"] = piv.round(4).to_dict()

    json.dump(report, open(os.path.join(a.out, "paper1_report.json"), "w"), indent=2, default=str)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        H = a.horizons
        ax[0].plot(H, [traj["amplification_median"][h] for h in H], "o-")
        ax[0].axhline(1, ls=":", c="gray")
        ax[0].set_xlabel("forecast horizon (years)")
        ax[0].set_ylabel("error amplification A(h)")
        ax[0].set_title("RQ1: does attack error grow with horizon?")
        c = channel[channel.goal == "fake_decline"]
        ax[1].barh(c.attack, c.mean_change_pred_atrophy)
        ax[1].set_xlabel("increase in predicted atrophy")
        ax[1].set_title("RQ-M2: pixel vs action channel (fake decline)")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "paper1_figures.png"), dpi=110)
    except ImportError:
        pass
    log.info("report saved to %s", os.path.join(a.out, "paper1_report.json"))


if __name__ == "__main__":
    main()
