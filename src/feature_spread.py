"""Feature-collapse check: do different scans get different image features?

For the same sample of TRAINING scans (one fold, no validation or test subjects), and for each encoder
(random init, plus any classifier or world-model checkpoints you pass), this measures the image vector
under several poolings of the 6x6x6 = 216 latent tokens:

  attn     the world model's attention pool (what the classifier uses by default)
  mean     average of tokens
  max      max of tokens
  meanmax  [mean, max]
  tokens   all 216 tokens flattened, no pooling (does the encoder separate scans at all?)

Columns:
  spread/RMS   mean per-feature SD across scans / feature RMS. Near 0.01 = collapse (every scan
               gets nearly the same vector); tens of percent = scans are distinguishable
  cosine       mean cosine similarity between different scans (near 1 = collapse)
  eff. rank    effective number of directions the features use (about 1 = dimensional collapse,
               even when spread/RMS looks fine)
  probe AUC    logistic regression on the vector, cross-validated by SUBJECT on this sample,
               CDR 0 vs CDR >= 0.5 (and CN vs AD): how much label signal the vector carries

Reading it:
  tokens spread high but attn spread ~0.01      -> the pooling throws the differences away
                                                    (use --pool meanmax and --head_norm batch)
  all poolings ~0.01, tokens too                -> the encoder itself does not separate scans
                                                    (check the input volumes / training)
  spread high but probe AUC ~0.5                -> features differ, but not by diagnosis

  python src/feature_spread.py --index data/proc/index_proc.csv --splits data/splits.json --fold 0 \
      --key_file data.key --n 96 \
      --ckpt scratch=runs/classifier-scratch-fold0/best.pt \
      --ckpt no_aug=runs/classifier-no-augmentation-fold0/best.pt \
      --ckpt world_model=runs/pretrain/last.pt --out runs/feature_spread
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from common import get_device, get_logger, set_seed
from data import ScanDataset, load_split
from models import BrainWorldModel, effective_rank

log = get_logger("spread")
POOLS = ("attn", "mean", "max", "meanmax", "tokens")


def sample_scans(df, n, seed):
    """Up to n scans, as balanced across labels as the fold allows."""
    rng = np.random.default_rng(seed)
    per = int(np.ceil(n / df.label.nunique()))
    parts = [g.iloc[rng.permutation(len(g))[:per]] for _, g in df.groupby("label")]
    out = pd.concat(parts)
    if len(out) < n:  # a small class: top up from the rest
        rest = df.drop(out.index)
        out = pd.concat([out, rest.iloc[rng.permutation(len(rest))[:n - len(out)]]])
    return out.iloc[:n].reset_index(drop=True)


def load_encoder(path, cfg_default):
    """BrainWorldModel whose encoder + attention pool come from a classifier or world-model checkpoint."""
    if path is None:
        return BrainWorldModel(**cfg_default), "random init"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    wm = BrainWorldModel(**ck["cfg"])
    if "model" in ck:  # world-model pretraining checkpoint
        missing, _ = wm.load_state_dict(ck["model"], strict=False)
        kind = "world model"
    else:              # classifier checkpoint: encoder.* and pool.* live in ck["state"]
        st = ck["state"]
        enc = {k[len("encoder."):]: v for k, v in st.items() if k.startswith("encoder.")}
        pool = {k[len("pool."):]: v for k, v in st.items() if k.startswith("pool.")}
        wm.encoder.load_state_dict(enc)
        wm.pool.load_state_dict(pool)
        kind = f"classifier ({ck.get('task', '3class')}, pool={ck.get('pool_mode', 'attn')})"
    return wm, kind


@torch.no_grad()
def token_batches(wm, dl, dev):
    wm.eval().to(dev)
    toks = []
    for b in dl:
        tok, _ = BrainWorldModel.to_tokens(wm.encoder(b["x"].to(dev)))
        toks.append(tok.float().cpu())
    return torch.cat(toks)


def pooled(wm, tok, mode):
    if mode == "attn":
        with torch.no_grad():
            return wm.pool.cpu().eval()(tok)
    if mode == "mean":
        return tok.mean(1)
    if mode == "max":
        return tok.amax(1)
    if mode == "meanmax":
        return torch.cat([tok.mean(1), tok.amax(1)], 1)
    return tok.flatten(1)


def spread(f):
    f = f.double()
    rms = f.pow(2).mean().sqrt().item()
    sd = f.std(0).mean().item()
    fn = torch.nn.functional.normalize(f, dim=1)
    cos = fn @ fn.T
    off = cos[~torch.eye(len(f), dtype=torch.bool)].mean().item()
    return rms, sd / max(rms, 1e-12), off


def probe_auc(f, y, groups, seed):
    """Subject-grouped cross-validated AUC of a linear probe (NaN if a class is too small)."""
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 4 or len(np.unique(groups)) < 5:
        return np.nan
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(f, y, groups):
        if len(np.unique(y[tr])) < 2:
            return np.nan
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        m.fit(f[tr], y[tr])
        p[te] = m.predict_proba(f[te])[:, 1]
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data/proc/index_proc.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--key_file", default=None)
    ap.add_argument("--n", type=int, default=96, help="training scans to sample")
    ap.add_argument("--ckpt", action="append", default=[], help="name=path (classifier best.pt or pretrain .pt)")
    ap.add_argument("--no_random", action="store_true", help="skip the random-init encoder")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/feature_spread")
    a = ap.parse_args()
    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = get_device()
    key = None
    if a.key_file:
        from secure_io import load_key
        key = load_key(a.key_file)

    splits = json.load(open(a.splits))
    tr = load_split(a.index, splits, a.fold, "train")
    df = sample_scans(tr, a.n, a.seed)
    log.info("fold %d training sample: %d scans / %d subjects %s", a.fold, len(df), df.subject.nunique(),
             df.label.value_counts().to_dict())
    dl = DataLoader(ScanDataset(df, None, key), a.bs, num_workers=0)
    y_imp = (df.label != "CN").values.astype(int)
    m_ad = df.label.isin(["CN", "AD"]).values
    y_ad = (df.label[m_ad] == "AD").values.astype(int)
    groups = df.subject.values

    encoders = [(n.split("=", 1)[0], n.split("=", 1)[1]) for n in a.ckpt]
    cfg_default = {}
    if encoders:
        cfg_default = torch.load(encoders[0][1], map_location="cpu", weights_only=False)["cfg"]
    if not a.no_random:
        encoders = [("random", None)] + encoders

    rows = []
    for name, path in encoders:
        torch.manual_seed(a.seed)
        wm, kind = load_encoder(path, cfg_default)
        tok = token_batches(wm, dl, dev)
        log.info("%s (%s): tokens %s", name, kind, tuple(tok.shape))
        for mode in POOLS:
            f = pooled(wm, tok, mode)
            rms, ratio, cos = spread(f)
            fn = f.numpy()
            rows.append({"encoder": name, "kind": kind, "pool": mode, "dim": f.shape[1],
                         "RMS": round(rms, 4), "spread/RMS": round(ratio, 4), "cosine": round(cos, 4),
                         "eff. rank": round(effective_rank(f), 2),
                         "probe AUC CDR0 vs >=0.5": round(probe_auc(fn, y_imp, groups, a.seed), 3),
                         "probe AUC CN vs AD": round(probe_auc(fn[m_ad], y_ad, groups[m_ad], a.seed), 3)})
        del wm, tok
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.out, "feature_spread.csv"), index=False)
    try:
        body = tab.to_markdown(index=False)
    except ImportError:
        body = "```\n" + tab.to_string(index=False) + "\n```"
    open(os.path.join(a.out, "feature_spread.md"), "w").write(
        f"# Feature spread, fold {a.fold}, {len(df)} training scans / {df.subject.nunique()} subjects\n\n"
        f"Labels in the sample: {df.label.value_counts().to_dict()}\n\n{body}\n\n"
        "spread/RMS near 0.01 and cosine near 1 = collapse. Compare 'tokens' with 'attn': if tokens differ "
        "but attn does not, the pooling discards the differences. Probe AUCs on a sample this small are "
        "rough (wide uncertainty); use them to rank poolings, not as results.\n")
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(tab.to_string(index=False))
    print(f"\nwrote {a.out}/feature_spread.md and .csv")


if __name__ == "__main__":
    main()
