"""Stage 4 - CN / MCI / AD classification on top of the world model,
with optional adversarial training (robustness) and adversarial debiasing.

Run once per CV fold. Encoder is frozen for --freeze_epochs (linear probe
warm-up), then fine-tuned with a 10x lower LR than the head.

  # standard
  python src/finetune_classify.py --fold 0 --pretrained runs/pretrain/last.pt --out runs/cls/fold0

  # combined model (image + metadata adversarial training + debiasing) - the one to deploy
  python src/finetune_classify.py --fold 0 --pretrained runs/pretrain_p1/last.pt --out runs/p1_full/fold0 \
      --adv pgd --eps 0.01 --meta_adv 5 --meta_dropout 0.2 --debias scanner
  # adversarially robust (PGD-5, eps=0.01) + site-debiased
  python src/finetune_classify.py --fold 0 --pretrained runs/pretrain/last.pt --out runs/cls_adv/fold0 \
      --adv pgd --eps 0.01 --debias site
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from adversarial import (AttributeAdversary, adversarial_loss, build_attribute_labels, dann_lambda,
                         eps_schedule, pgd)
from metadata_attacks import joint_adversarial_loss, joint_pgd, meta_adversarial_batch
from common import LABEL2IDX, LABELS, get_device, get_logger, set_seed
from data import Augment3D, ScanDataset, load_split
from models import BrainWorldModel, WorldModelClassifier, cosine_lr


def build_model(ckpt_path, imagine, dev, cfg=None):
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        wm = BrainWorldModel(**ck["cfg"])
        if "model" in ck:  # pretrain checkpoint
            wm.load_state_dict(ck["model"])
    else:
        wm = BrainWorldModel(**(cfg or {}))
    return WorldModelClassifier(wm, len(LABELS), imagine).to(dev), wm.cfg


@torch.no_grad()
def predict(model, dl, dev, tta=True, attack=None):
    """Class probabilities. attack=(eps, steps) evaluates on PGD adversarial inputs instead;
    attack=(eps, steps, age_years) attacks pixels and age jointly."""
    model.eval()
    probs = []
    for b in dl:
        x, d = b["x"].to(dev), b["demo"].to(dev)
        if attack and len(attack) > 2 and attack[2] > 0:
            x, d = joint_pgd(model, x, d, b["y"].to(dev), attack[0], attack[2], attack[1])
        elif attack:
            x = pgd(model, x, d, b["y"].to(dev), attack[0], attack[1])
        p = F.softmax(model(x, d), 1)
        if tta:
            p = (p + F.softmax(model(x.flip(-1), d), 1)) / 2
        probs.append(p.cpu())
    return torch.cat(probs).numpy()


def metrics(y, p):
    pred = p.argmax(1)
    out = {"acc": float((pred == y).mean()), "bal_acc": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=range(len(LABELS)), zero_division=0))}
    present = np.unique(y)
    if len(present) == len(LABELS):
        out["macro_auc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    if {0, 2}.issubset(present):  # CN vs AD only, the classic ADNI benchmark
        m = np.isin(y, [0, 2])
        pa = p[m][:, 2] / (p[m][:, 0] + p[m][:, 2] + 1e-8)
        out["auc_cn_vs_ad"] = float(roc_auc_score(y[m] == 2, pa))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--pretrained", default=None, help="world-model checkpoint (omit = from scratch)")
    ap.add_argument("--out", default="runs/cls/fold0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--freeze_epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--enc_lr_mult", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--imagine", type=float, nargs="*", default=[1.0, 2.0],
                    help="future horizons (years) imagined by the dynamics; pass nothing to disable")
    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--key_file", default=None)
    ap.add_argument("--seed", type=int, default=42)
    # --- adversarial robustness
    ap.add_argument("--adv", default="none", choices=["none", "fgsm", "pgd", "trades"])
    ap.add_argument("--eps", type=float, default=0.01, help="L-inf budget in [0,1] intensity units")
    ap.add_argument("--pgd_steps", type=int, default=5)
    ap.add_argument("--adv_ratio", type=float, default=0.5, help="weight of adversarial vs clean CE")
    ap.add_argument("--trades_beta", type=float, default=6.0)
    ap.add_argument("--eps_ramp", type=int, default=10, help="epochs to ramp eps from 0 to --eps")
    ap.add_argument("--select", default="auto", choices=["auto", "clean", "robust", "mean"],
                    help="checkpoint selection metric (auto = mean when --adv is on, else clean)")
    # --- adversarial debiasing
    ap.add_argument("--debias", default="none", choices=["none", "site", "scanner", "sex"])
    ap.add_argument("--debias_lambda", type=float, default=0.3)
    # --- metadata robustness (Paper 1, RQ-M3 / RQ-M4)
    ap.add_argument("--no_meta", action="store_true", help="image-only model (ignore age/sex)")
    ap.add_argument("--meta_dropout", type=float, default=0.0, help="train-time prob. of hiding metadata")
    ap.add_argument("--meta_adv", type=float, default=0.0,
                    help="metadata adversarial training: worst-case age shift budget in years (0 = off)")
    a = ap.parse_args()
    set_seed(a.seed + a.fold)
    os.makedirs(a.out, exist_ok=True)
    log = get_logger(f"finetune_fold{a.fold}", a.out)
    dev = get_device()
    key = None
    if a.key_file:
        from secure_io import load_key
        key = load_key(a.key_file)

    splits = json.load(open(a.splits))
    tr = load_split(a.index, splits, a.fold, "train")
    va = load_split(a.index, splits, a.fold, "val")
    log.info("fold %d | train %d scans (%s) | val %d scans (%s)", a.fold, len(tr),
             tr.label.value_counts().to_dict(), len(va), va.label.value_counts().to_dict())

    y_tr = tr.label.map(LABEL2IDX).values
    counts = np.bincount(y_tr, minlength=len(LABELS)).astype(float)
    cls_w = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float32)
    ds_tr = ScanDataset(tr, Augment3D(), key)
    sampler = (WeightedRandomSampler(cls_w[y_tr].double(), len(y_tr), replacement=True)
               if a.balanced_sampler else None)
    dl_tr = DataLoader(ds_tr, a.bs, shuffle=sampler is None, sampler=sampler, num_workers=a.workers,
                       drop_last=len(ds_tr) > a.bs)
    dl_va = DataLoader(ScanDataset(va, None, key), a.bs, num_workers=a.workers)

    model, wm_cfg = build_model(a.pretrained, a.imagine, dev)
    model.use_meta, model.meta_dropout = not a.no_meta, a.meta_dropout
    groups = model.param_groups(a.lr, a.enc_lr_mult, a.wd)

    adversary, attr = None, None
    if a.debias != "none":
        attr, attr_classes = build_attribute_labels(tr, a.debias)
        if len(attr_classes) < 2:
            log.warning("only one %s group in training data - debiasing disabled", a.debias)
        else:
            adversary = AttributeAdversary(model.feat_dim, len(attr_classes)).to(dev)
            groups.append({"params": adversary.parameters(), "lr": a.lr, "weight_decay": a.wd})
            log.info("debiasing %s: %d groups %s", a.debias, len(attr_classes), attr_classes[:10])
            attr = torch.from_numpy(attr)

    opt = torch.optim.AdamW(groups)
    base_lrs = [g["lr"] for g in opt.param_groups]
    crit = torch.nn.CrossEntropyLoss(weight=None if a.balanced_sampler else cls_w.to(dev),
                                     label_smoothing=a.label_smoothing)
    select = a.select if a.select != "auto" else ("mean" if a.adv != "none" else "clean")
    # combined image + metadata adversarial training: --adv pgd|fgsm together with --meta_adv > 0
    joint = a.adv in ("pgd", "fgsm") and a.meta_adv > 0 and not a.no_meta
    log.info("adv=%s eps=%.4f steps=%d | debias=%s | select=%s | meta: use=%s dropout=%.2f adv=%.1fy | joint=%s",
             a.adv, a.eps, a.pgd_steps, a.debias, select, not a.no_meta, a.meta_dropout, a.meta_adv, joint)
    total, step = a.epochs * len(dl_tr), 0
    best, best_ep, y_va = -1, 0, va.label.map(LABEL2IDX).values
    adv_start = a.freeze_epochs if a.pretrained else 0

    for ep in range(a.epochs):
        frozen = ep < a.freeze_epochs and a.pretrained is not None
        for p in model.encoder.parameters():
            p.requires_grad = not frozen
        eps_now = eps_schedule(ep, a.eps, a.eps_ramp, adv_start)
        model.train()
        if adversary:
            adversary.train()
        tot, tot_adv, adv_acc = 0.0, 0.0, []
        for b in dl_tr:
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = cosine_lr(step, total, base, len(dl_tr) * 2)
            x, d, y = b["x"].to(dev), b["demo"].to(dev), b["y"].to(dev)
            meta_on = a.meta_adv > 0 and not frozen and model.use_meta
            if joint and meta_on and eps_now > 0:
                # combined model: pixels and age attacked together in one PGD
                loss, _, feats = joint_adversarial_loss(model, x, d, y, crit, eps_now, a.meta_adv,
                                                        a.pgd_steps, a.adv_ratio)
            else:
                if meta_on:
                    d_adv = meta_adversarial_batch(model, x, d, y, a.meta_adv)
                    pick = torch.rand(d.shape[0], 1, device=dev) < 0.5
                    d = torch.where(pick, d_adv, d)
                loss, _, feats = adversarial_loss(model, x, d, y, crit, a.adv, eps_now, a.pgd_steps,
                                                  a.adv_ratio, a.trades_beta)
            if adversary:
                adversary.lam = 0.0 if frozen else dann_lambda(step / total, a.debias_lambda)
                a_logits = adversary(feats)
                a_y = attr[b["idx"]].to(dev)
                l_adv = F.cross_entropy(a_logits, a_y)
                loss = loss + l_adv
                tot_adv += l_adv.item()
                adv_acc.append((a_logits.argmax(1) == a_y).float().mean().item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            step += 1

        p_va = predict(model, dl_va, dev)
        m = metrics(y_va, p_va)
        if a.adv != "none" or select != "clean":
            p_rob = predict(model, dl_va, dev, tta=False,
                            attack=(a.eps, a.pgd_steps, a.meta_adv if joint and model.use_meta else 0))
            m["robust_bal_acc"] = metrics(y_va, p_rob)["bal_acc"]
        score = {"clean": m["bal_acc"], "robust": m.get("robust_bal_acc", 0),
                 "mean": (m["bal_acc"] + m.get("robust_bal_acc", 0)) / 2}[select]
        score += 0.01 * m.get("macro_auc", 0)
        extra = f" | eps {eps_now:.4f}" if a.adv != "none" else ""
        if adversary:
            extra += f" | {a.debias}-adv loss {tot_adv / len(dl_tr):.3f} acc {np.mean(adv_acc):.2f}"
        log.info("ep %3d%s | loss %.4f%s | %s", ep + 1, " (enc frozen)" if frozen else "",
                 tot / max(1, len(dl_tr)), extra, " ".join(f"{k}={v:.3f}" for k, v in m.items()))
        if score > best:
            best, best_ep = score, ep + 1
            torch.save({"state": model.state_dict(), "cfg": wm_cfg, "imagine": a.imagine,
                        "val_metrics": m, "epoch": ep + 1, "adv": a.adv, "eps": a.eps,
                        "debias": a.debias, "use_meta": not a.no_meta, "meta_dropout": a.meta_dropout,
                        "meta_adv": a.meta_adv, "joint": joint}, os.path.join(a.out, "best.pt"))
            out = va[["image_id", "subject", "label"]].copy()
            for i, l in enumerate(LABELS):
                out[f"p_{l}"] = p_va[:, i]
            out.to_csv(os.path.join(a.out, "val_predictions.csv"), index=False)
        elif ep + 1 - best_ep >= a.patience and not frozen and eps_now >= a.eps:
            log.info("early stop (no val improvement for %d epochs)", a.patience)
            break
    log.info("best epoch %d -> %s", best_ep, os.path.join(a.out, "best.pt"))


if __name__ == "__main__":
    main()
