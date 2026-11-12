"""Bundle trained checkpoints for the web app, with SHA-256 hashes the app checks at start-up.

  python deploy/package_models.py --classifiers "runs/p1_meta/fold*/best.pt" \
      --world_model runs/pretrain_p1/last.pt --out models --version v1 \
      --trained_on "ADNI, 120 subjects, 5-fold"

Creates models/classifiers/fold<i>.pt, models/world_model.pt and models/manifest.json.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil

import torch


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifiers", required=True, help="glob of finetune best.pt files")
    ap.add_argument("--world_model", required=True, help="pretrain checkpoint (last.pt)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--version", default=datetime.date.today().isoformat())
    ap.add_argument("--trained_on", default="")
    ap.add_argument("--header_detector", default=None, help="optional header_detector.joblib (src/header_anomaly.py)")
    ap.add_argument("--target", type=int, default=None, help="input size; read from data if omitted")
    a = ap.parse_args()
    cls = sorted(glob.glob(a.classifiers))
    assert cls, f"no classifiers match {a.classifiers}"
    os.makedirs(os.path.join(a.out, "classifiers"), exist_ok=True)
    rel = []
    for i, p in enumerate(cls):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        keep = {k: ck[k] for k in ("state", "cfg", "imagine", "use_meta", "val_metrics", "adv", "eps", "debias") if k in ck}
        r = f"classifiers/fold{i}.pt"
        torch.save(keep, os.path.join(a.out, r))  # drops anything not needed for inference
        rel.append(r)
    wm = torch.load(a.world_model, map_location="cpu", weights_only=False)
    torch.save({"model": wm["model"], "cfg": wm["cfg"]}, os.path.join(a.out, "world_model.pt"))
    target = a.target or wm.get("args", {}).get("target") or 96
    files = rel + ["world_model.pt"]
    extra = {}
    if a.header_detector:
        shutil.copy(a.header_detector, os.path.join(a.out, "header_detector.joblib"))
        files.append("header_detector.joblib")
        extra["header_detector"] = "header_detector.joblib"
    man = {"version": a.version, "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "trained_on": a.trained_on,
           "target": int(target), "classifiers": rel, "world_model": "world_model.pt",
           **extra, "sha256": {f: sha256(os.path.join(a.out, f)) for f in files}}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"{len(rel)} classifiers + world model{' + header detector' if extra else ''} -> {a.out}/ (manifest.json with SHA-256)")


if __name__ == "__main__":
    main()
