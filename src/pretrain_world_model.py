"""Stage 3 - Self-supervised world-model pretraining (no labels used).

Trains the encoder with 3D JEPA + the action-conditioned dynamics on
longitudinal transitions. Test-set subjects are excluded entirely.

  python src/pretrain_world_model.py --index data/proc/index_proc.csv \
      --splits data/splits.json --out runs/pretrain --epochs 150 --bs 4
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from common import get_device, get_logger, set_seed
from data import Augment3D, LongitudinalPairDataset, load_split
from models import BrainWorldModel, cosine_lr, effective_rank
from trajectory_attacks import trajectory_trades_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if it exists")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--mask_ratio", type=float, default=0.5)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--widths", default="32,64,128,256")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--key_file", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_fold", type=int, default=None,
                    help="fold-specific pretraining: also leave out this fold's validation subjects, so "
                         "that fold's classifier is validated on people the world model never saw")
    # Paper 1 additions
    ap.add_argument("--w_vol", type=float, default=0.5, help="brain-volume (BPF) readout loss")
    # anti-collapse (VICReg-style). Older runs used --w_var 0.04 --w_cov 0 --var_queue 0, which gave
    # dimensional collapse (effective rank ~1.1 of 256) on OASIS-2; VICReg weights invariance :
    # variance : covariance as 25 : 25 : 1, i.e. 1 : 1 : 0.04 relative to the JEPA loss
    ap.add_argument("--w_var", type=float, default=1.0, help="variance term (keep each dimension varying)")
    ap.add_argument("--w_cov", type=float, default=0.04,
                    help="covariance term (decorrelate dimensions; 0 = off, as in older runs)")
    ap.add_argument("--var_queue", type=int, default=64,
                    help="recent detached summaries added to the variance/covariance statistics (0 = batch only)")
    ap.add_argument("--w_mono", type=float, default=0.5, help="atrophy monotonicity loss (0 = off)")
    ap.add_argument("--w_age", type=float, default=0.1, help="brain-age head loss")
    ap.add_argument("--traj_eps", type=float, default=0.0,
                    help="Trajectory-TRADES pixel budget (0 = off), e.g. 0.01")
    ap.add_argument("--traj_steps", type=int, default=2)
    ap.add_argument("--w_traj", type=float, default=1.0)
    a = ap.parse_args()
    set_seed(a.seed)
    log = get_logger("pretrain", a.out)
    dev = get_device()

    key = None
    if a.key_file:
        from secure_io import load_key
        key = load_key(a.key_file)
    splits = json.load(open(a.splits))
    df = load_split(a.index, splits, part="dev")  # never touch test subjects
    if a.exclude_fold is not None:
        held = set(splits["folds"][a.exclude_fold]["val"])
        df = df[~df.subject.isin(held)].reset_index(drop=True)
        log.info("fold-specific pretraining: excluded %d validation subjects of fold %d", len(held), a.exclude_fold)
    ds = LongitudinalPairDataset(df, aug=Augment3D(), key=key)
    log.info("scans=%d  subjects=%d  transitions(dt>0)=%d  items=%d",
             len(df), df.subject.nunique(), ds.n_transitions, len(ds))
    if ds.n_transitions == 0:
        log.warning("no longitudinal pairs in your subset - dynamics only learns dt=0; "
                    "download follow-up visits for the same subjects to train it properly")
    dl = DataLoader(ds, a.bs, shuffle=True, num_workers=a.workers, drop_last=len(ds) > a.bs,
                    pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)

    widths = tuple(int(w) for w in a.widths.split(","))
    model = BrainWorldModel(a.embed_dim, widths).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    log.info("trainable params: %.2fM", sum(p.numel() for p in params) / 1e6)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = max(1, len(dl) // a.accum)
    total = a.epochs * steps_per_epoch
    step, best, start = 0, math.inf, 0
    last = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last):  # continue after a Colab/Kaggle disconnect
        ck = torch.load(last, map_location=dev, weights_only=False)
        if "opt" in ck:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            step, best, start = ck["step"], ck["best"], ck["epoch"]
            log.info("resumed from %s at epoch %d", last, start)
    for ep in range(start, a.epochs):
        model.train()
        t0, agg, n = time.time(), {}, 0
        opt.zero_grad(set_to_none=True)
        for it, b in enumerate(dl):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            x_adv = None
            if a.traj_eps > 0 and ep >= a.warmup_epochs:
                x_adv = trajectory_trades_example(model, b["x_t"], b["action"], a.traj_eps, a.traj_steps)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                loss, parts = model.forward_pretrain(
                    b["x_t"], b["x_f"], b["action"], b["is_transition"], b["bpf_t"], b["bpf_f"],
                    b["age_known"], a.mask_ratio, w_vol=a.w_vol, w_mono=a.w_mono, w_age=a.w_age,
                    x_adv=x_adv, w_traj=a.w_traj, w_var=a.w_var, w_cov=a.w_cov, var_queue=a.var_queue)
            scaler.scale(loss / a.accum).backward()
            if (it + 1) % a.accum == 0 or it + 1 == len(dl):
                for g in opt.param_groups:
                    g["lr"] = cosine_lr(step, total, a.lr, a.warmup_epochs * steps_per_epoch)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                m = 1 - (1 - a.ema) * (math.cos(math.pi * step / total) + 1) / 2
                model.ema_update(m)
                step += 1
            for k, v in {"loss": loss.item(), **parts}.items():
                agg[k] = agg.get(k, 0) + v
            n += 1
        agg = {k: v / n for k, v in agg.items()}
        q = getattr(model, "_queue_mean", None)
        if q is not None and q.shape[0] >= 16:  # collapse monitor on recent mean-pooled summaries
            agg["eff_rank"] = effective_rank(q)
        log.info("ep %3d | %s | %.0fs", ep + 1, " ".join(f"{k}={v:.4f}" for k, v in agg.items()),
                 time.time() - t0)
        ckpt = {"model": model.state_dict(), "cfg": model.cfg, "epoch": ep + 1, "args": vars(a)}
        if agg["loss"] < best:
            best = agg["loss"]
            torch.save(ckpt, os.path.join(a.out, "best.pt"))
        torch.save({**ckpt, "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "best": best},
                   last + ".tmp")
        os.replace(last + ".tmp", last)  # atomic: a disconnect mid-save never corrupts last.pt
    log.info("done - checkpoint: %s", os.path.join(a.out, "last.pt"))


if __name__ == "__main__":
    main()
