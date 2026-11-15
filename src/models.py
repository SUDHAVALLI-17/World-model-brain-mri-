"""Brain-MRI world model.

  Encoder      3D ResNet -> grid of latent tokens (D/16)^3 x E
  JEPA         predicts EMA-target tokens of masked 3D blocks from the visible
               context (learns anatomy without labels; I-JEPA style, in 3D)
  Dynamics     action-conditioned transition model: given tokens of the scan at
               time t and an action a = [dt, age, sex], predicts the latent
               tokens of the same brain at t+dt, plus a Gaussian over the
               future global state (mean, log-variance = uncertainty)
  Classifier   CN / MCI / AD head on [current state, imagined future states]
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class BasicBlock3D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = gn(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = gn(cout)
        self.sc = (nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), gn(cout))
                   if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        return F.silu(self.n2(self.c2(h)) + self.sc(x))


class ResEncoder3D(nn.Module):
    """Total stride 16: 96^3 input -> 6^3 = 216 tokens."""

    def __init__(self, widths=(32, 64, 128, 256), blocks=(1, 2, 2, 2), embed_dim=256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 5, 2, 2, bias=False), gn(widths[0]), nn.SiLU())
        layers, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(BasicBlock3D(cin, w, 2 if (j == 0 and i > 0) else 1))
                cin = w
        self.stages = nn.Sequential(*layers)
        self.proj = nn.Conv3d(cin, embed_dim, 1)

    def forward(self, x):
        return self.proj(self.stages(self.stem(x)))  # (B, E, d, h, w)


def sincos_pos_3d(grid, dim, device):
    """Fixed 3D sine-cosine positional embedding, (N, dim)."""
    d3 = (dim // 6) * 2
    out = []
    for g in grid:
        pos = torch.arange(g, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (torch.arange(d3 // 2, device=device) / (d3 / 2)))
        out.append(torch.cat([torch.sin(pos[:, None] * omega), torch.cos(pos[:, None] * omega)], 1))
    zz, yy, xx = torch.meshgrid(*[torch.arange(g, device=device) for g in grid], indexing="ij")
    pe = torch.cat([out[0][zz.flatten()], out[1][yy.flatten()], out[2][xx.flatten()]], 1)
    return F.pad(pe, (0, dim - pe.shape[1]))


def transformer(dim, depth, heads=8, drop=0.1):
    layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, drop, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class AttentionPool(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tok):
        out, _ = self.attn(self.q.expand(tok.shape[0], -1, -1), tok, tok)
        return self.norm(out[:, 0])


class JEPAPredictor(nn.Module):
    def __init__(self, dim, pdim=192, depth=4):
        super().__init__()
        self.inp = nn.Linear(dim, pdim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pdim))
        self.blocks = transformer(pdim, depth, heads=6)
        self.out = nn.Sequential(nn.LayerNorm(pdim), nn.Linear(pdim, dim))
        self.pdim = pdim
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, ctx, mask, grid):
        h = self.inp(ctx)
        h = torch.where(mask[..., None], self.mask_token.expand_as(h), h)
        h = h + sincos_pos_3d(grid, self.pdim, h.device)
        return self.out(self.blocks(h))


class ActionDynamics(nn.Module):
    """z_{t+dt} ~ f(z_t, a): FiLM + action token + transformer, residual prediction."""

    def __init__(self, dim, action_dim=3, depth=3):
        super().__init__()
        self.act = nn.Sequential(nn.Linear(action_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.film = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.blocks = transformer(dim, depth)
        self.delta = nn.Linear(dim, dim)
        self.pool = AttentionPool(dim)
        self.dist = nn.Linear(dim, 2 * dim)

    def forward(self, tok, action, grid):
        a = self.act(action)
        g, b = self.film(a).chunk(2, -1)
        h = tok * (1 + g[:, None]) + b[:, None] + sincos_pos_3d(grid, tok.shape[-1], tok.device)
        h = self.blocks(torch.cat([a[:, None], h], 1))[:, 1:]
        tok_next = tok + self.delta(h)
        mu, logvar = self.dist(self.pool(tok_next)).chunk(2, -1)
        return tok_next, mu, logvar.clamp(-6, 4)


def sample_block_mask(B, grid, ratio=0.5, device="cpu"):
    """Mask several random 3D blocks of tokens until ~ratio of tokens are hidden."""
    d, h, w = grid
    m = torch.zeros(B, d, h, w, dtype=torch.bool)
    for b in range(B):
        for _ in range(50):
            if m[b].float().mean() >= ratio:
                break
            sz = [torch.randint(max(1, g // 4), max(2, g // 2 + 1), (1,)).item() for g in grid]
            st = [torch.randint(0, g - s + 1, (1,)).item() for g, s in zip(grid, sz)]
            m[b, st[0]:st[0] + sz[0], st[1]:st[1] + sz[1], st[2]:st[2] + sz[2]] = True
        if m[b].all():
            m[b].view(-1)[torch.randint(0, m[b].numel(), (1,))] = False
    return m.to(device)


class BrainWorldModel(nn.Module):
    """Encoder + JEPA predictor + action-conditioned dynamics, plus two readout heads:

    * vol_head : brain-parenchymal-fraction (BPF, a clinical atrophy measure) from a
                 global latent in the dynamics' target space. Because it reads the same
                 space the dynamics predicts, v(mu_{t+dt}) is the *predicted future atrophy*.
    * age_head : brain age from the pooled latent. Used to check stated age (metadata)
                 against what the image looks like (metadata-inconsistency defence).
    """

    def __init__(self, embed_dim=256, widths=(32, 64, 128, 256), blocks=(1, 2, 2, 2),
                 pred_depth=4, dyn_depth=3, action_dim=3):
        super().__init__()
        self.cfg = dict(embed_dim=embed_dim, widths=widths, blocks=blocks, pred_depth=pred_depth,
                        dyn_depth=dyn_depth, action_dim=action_dim)
        self.encoder = ResEncoder3D(widths, blocks, embed_dim)
        self.pool = AttentionPool(embed_dim)
        self.predictor = JEPAPredictor(embed_dim, depth=pred_depth)
        self.dynamics = ActionDynamics(embed_dim, action_dim, dyn_depth)
        self.vol_head = nn.Linear(embed_dim, 1)
        self.age_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @staticmethod
    def to_tokens(fmap):
        return fmap.flatten(2).transpose(1, 2), tuple(fmap.shape[2:])

    def encode(self, x, target=False):
        enc = self.target_encoder if target else self.encoder
        return self.to_tokens(enc(x))

    @torch.no_grad()
    def ema_update(self, m):
        for pt, po in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(m).add_(po.detach(), alpha=1 - m)

    @staticmethod
    def target_space(tok):
        return F.layer_norm(tok, tok.shape[-1:])

    def volume(self, g):
        return self.vol_head(g).squeeze(-1)

    def brain_age(self, tok):
        """Predicted age in normalised units ((years - 73) / 8)."""
        return self.age_head(self.pool(tok)).squeeze(-1)

    def forecast(self, x, action, tok=None, grid=None):
        """Current global state g0, predicted future (mu, logvar) for one action."""
        if tok is None:
            tok, grid = self.encode(x)
        g0 = self.target_space(tok).mean(1)
        _, mu, logvar = self.dynamics(tok, action, grid)
        return g0, mu, logvar

    def forward_pretrain(self, x_t, x_f, action, is_transition, bpf_t=None, bpf_f=None,
                         age_known=None, mask_ratio=0.5, w_jepa=1.0, w_dyn=1.0, w_nll=0.1,
                         w_var=0.04, w_vol=0.5, w_mono=0.5, w_age=0.1, x_adv=None, w_traj=1.0):
        B = x_t.shape[0]
        with torch.no_grad():
            tgt_t, grid = self.encode(x_t, target=True)
            tgt_f, _ = self.encode(x_f, target=True)
            tgt_t, tgt_f = self.target_space(tgt_t), self.target_space(tgt_f)

        # --- JEPA: hide 3D blocks of the input, predict their target latents
        mask = sample_block_mask(B, grid, mask_ratio, x_t.device)
        vox_mask = F.interpolate(mask[:, None].float(), size=x_t.shape[2:], mode="nearest")
        ctx, _ = self.encode(x_t * (1 - vox_mask))
        mflat = mask.flatten(1)
        pred = self.predictor(ctx, mflat, grid)
        l_jepa = F.smooth_l1_loss(pred[mflat], tgt_t[mflat])

        # --- Dynamics: act on the full current state, predict the future state
        tok_t, _ = self.encode(x_t)
        tok_next, mu, logvar = self.dynamics(tok_t, action, grid)
        w = 0.2 + 0.8 * is_transition  # self-transitions (dt=0) count less
        l_dyn = (F.smooth_l1_loss(tok_next, tgt_f, reduction="none").mean((1, 2)) * w).sum() / w.sum()
        g_t, g_f = tgt_t.mean(1), tgt_f.mean(1)
        nll = 0.5 * (logvar + (g_f - mu) ** 2 / logvar.exp()).mean(1)
        l_nll = (nll * w).sum() / w.sum()

        # --- anti-collapse: keep per-dimension variance of the pooled state
        z = self.pool(tok_t)
        l_var = F.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean() if B >= 4 else z.new_zeros(())
        parts = {"jepa": l_jepa.item(), "dyn": l_dyn.item(), "nll": l_nll.item(), "var": l_var.item()}
        loss = w_jepa * l_jepa + w_dyn * l_dyn + w_nll * l_nll + w_var * l_var

        # --- volume readout: BPF of the current scan, and of the real follow-up from mu
        if bpf_t is not None and w_vol > 0:
            l_vol = F.smooth_l1_loss(self.volume(g_t), bpf_t)
            l_vol_f = (F.smooth_l1_loss(self.volume(mu), bpf_f, reduction="none") * is_transition).sum() \
                / is_transition.sum().clamp(min=1)
            loss = loss + w_vol * (l_vol + l_vol_f)
            parts["vol"] = (l_vol + l_vol_f).item()

        # --- atrophy monotonicity: predicted brain volume must not grow with time.
        #     v(mu(dt1)) <= v(g_t) and v(mu(dt1 + dt2)) <= v(mu(dt1)) for random horizons
        if w_mono > 0:
            dt1 = torch.rand(B, 1, device=x_t.device) * 1.5 + 0.25
            dt2 = dt1 + torch.rand(B, 1, device=x_t.device) * 1.5 + 0.25
            a1 = torch.cat([dt1 / 2.0, action[:, 1:]], 1)
            a2 = torch.cat([dt2 / 2.0, action[:, 1:]], 1)
            _, mu1, _ = self.dynamics(tok_t, a1, grid)
            _, mu2, _ = self.dynamics(tok_t, a2, grid)
            v0 = self.volume(self.target_space(tok_t).mean(1))
            v1, v2 = self.volume(mu1), self.volume(mu2)
            l_mono = (F.relu(v1 - v0) + F.relu(v2 - v1)).mean()
            loss = loss + w_mono * l_mono
            parts["mono"] = l_mono.item()

        # --- brain-age head (only where age is known)
        if age_known is not None and w_age > 0 and age_known.sum() > 0:
            pa = self.brain_age(tok_t)
            l_age = (F.smooth_l1_loss(pa, action[:, 1], reduction="none") * age_known).sum() / age_known.sum()
            loss = loss + w_age * l_age
            parts["age"] = l_age.item()

        # --- Trajectory-TRADES: predicted future of an attacked scan must match the clean one
        if x_adv is not None and w_traj > 0:
            tok_a, _ = self.encode(x_adv)
            _, mu_a, logvar_a = self.dynamics(tok_a, action, grid)
            l_traj = F.mse_loss(mu_a, mu.detach()) + 0.1 * F.mse_loss(logvar_a, logvar.detach())
            loss = loss + w_traj * l_traj
            parts["traj"] = l_traj.item()
        return loss, parts

    @torch.no_grad()
    def rollout(self, x, action_seq):
        """Imagine a trajectory: apply actions one after another in latent space."""
        tok, grid = self.encode(x)
        traj = []
        for a in action_seq:
            tok, mu, logvar = self.dynamics(tok, a, grid)
            traj.append((mu, logvar.exp()))
        return traj


class WorldModelClassifier(nn.Module):
    """Classify from the current latent state + imagined future states.

    Features = [pool(z_t), mu(t+dt_1) - g_t, mu(t+dt_2) - g_t, ...] where g_t is the
    current global state in the dynamics' target space. The deltas describe how
    the world model expects this brain to change - informative for MCI vs AD.
    """

    def __init__(self, wm: BrainWorldModel, n_classes=3, imagine_dts=(1.0, 2.0), dropout=0.3,
                 freeze_dynamics=True, use_meta=True, meta_dropout=0.0, n_extra=0):
        super().__init__()
        # demo = [age, sex, *extra]: age/sex drive the dynamics; extra scalar features (e.g. brain
        # volume / eTIV) go to the classification head only and are never hidden or attacked
        self.n_extra = n_extra
        self.encoder, self.pool, self.dynamics = wm.encoder, wm.pool, wm.dynamics
        self.age_head = wm.age_head
        self.use_meta = use_meta          # False = image-only model (RQ-M3 baseline)
        self.meta_dropout = meta_dropout  # train-time probability of hiding metadata
        self.imagine_dts = list(imagine_dts)
        if freeze_dynamics:
            for p in self.dynamics.parameters():
                p.requires_grad = False
        E = wm.cfg["embed_dim"]
        fdim = E * (1 + len(self.imagine_dts)) + 2 + n_extra
        self.head = nn.Sequential(nn.LayerNorm(fdim), nn.Dropout(dropout), nn.Linear(fdim, 256),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(256, n_classes))
        self._fmap = None

    @property
    def feat_dim(self):
        return self.head[0].normalized_shape[0] - 2 - self.n_extra

    @staticmethod
    def missing_meta(demo):
        """'Unknown' metadata: population-mean age, sex = 0.5."""
        d = demo.clone()
        d[:, 0], d[:, 1] = 0.0, 0.5
        return d

    def brain_age(self, x):
        tok, _ = BrainWorldModel.to_tokens(self.encoder(x))
        return self.age_head(self.pool(tok)).squeeze(-1)

    def forward(self, x, demo, return_feats=False):
        if not self.use_meta:
            demo = self.missing_meta(demo)
        elif self.training and self.meta_dropout > 0:
            drop = torch.rand(demo.shape[0], 1, device=demo.device) < self.meta_dropout
            demo = torch.where(drop, self.missing_meta(demo), demo)
        fmap = self.encoder(x)
        if fmap.requires_grad:
            fmap.retain_grad()
        self._fmap = fmap  # kept for Grad-CAM
        tok, grid = BrainWorldModel.to_tokens(fmap)
        feats = [self.pool(tok)]
        if self.imagine_dts:
            g_now = BrainWorldModel.target_space(tok).mean(1)
            for dt in self.imagine_dts:
                a = torch.cat([torch.full_like(demo[:, :1], dt / 2.0), demo[:, :2]], 1)
                _, mu, _ = self.dynamics(tok, a, grid)
                feats.append(mu - g_now)
        img_feats = torch.cat(feats, 1)  # image-derived features (what the debias adversary sees)
        logits = self.head(torch.cat([img_feats, demo], 1))
        return (logits, img_feats) if return_feats else logits

    def param_groups(self, lr, enc_mult=0.1, wd=0.05):
        enc = [p for n, p in self.named_parameters()
               if not n.startswith(("head", "adversary")) and p.requires_grad]
        return [{"params": enc, "lr": lr * enc_mult, "weight_decay": wd},
                {"params": self.head.parameters(), "lr": lr, "weight_decay": wd}]


def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
