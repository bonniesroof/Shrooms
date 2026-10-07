"""Static graph JEPA (PyTorch): predict the latents of masked patches from the visible ones.

One snapshot at a time, no time step. A block of patches is hidden; the
context encoder sees the rest, and a predictor fills in what the *target
encoder* says the hidden patches look like. The loss lives in latent space:
nothing is reconstructed.

    snapshot ──► target encoder (EMA, no grad) ──► layer-normed latents of masked patches ┐
       │                                                                   smooth-L1 │
       └─ mask ─► context encoder ──► predictor (mask tokens at hidden patches) ──────┘

Anti-collapse hygiene: the target encoder is an exponential moving average of
the context encoder and gets no gradient (stop-grad), and an optional VICReg
variance hinge keeps every latent dimension spread out across patches.

Building blocks come from gnn.py: the same typed message-passing layer and
adjacency (adjacent, downslope, hyphal). Terrain (elevation) is treated as
known everywhere, like a position; a hidden patch never contributes hyphal
edges, so its fungal state can't leak through the graph.
"""

import copy
import math

import torch
import torch.nn.functional as fn
from torch import nn

from worldmodel.gnn import Layer, adjacency, mlp
from worldmodel.graph import adjacent_pairs


def grid_mask(rows: int, cols: int) -> torch.Tensor:
    pairs = torch.as_tensor(adjacent_pairs(rows, cols))
    mask = torch.zeros(rows * cols, rows * cols)
    mask[pairs[:, 0], pairs[:, 1]] = 1
    mask[pairs[:, 1], pairs[:, 0]] = 1
    return mask


class Encoder(nn.Module):
    """Patch features -> one latent per patch. Hidden patches get a learned mask embedding."""

    def __init__(self, n_in: int, n_glob: int, hidden: int, layers: int, n_nodes: int):
        super().__init__()
        self.embed = mlp(n_in, hidden, hidden)
        self.embed_glob = mlp(n_glob, hidden, hidden)
        self.pos = nn.Parameter(torch.randn(n_nodes, hidden) * 0.02)  # stable patch IDs
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        self.layers = nn.ModuleList(Layer(hidden) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, nodes, glob, adj, hidden=None):
        h = self.embed(nodes)
        if hidden is not None:
            h = torch.where(hidden[..., None], self.mask_token.expand_as(h), h)
        h = h + self.pos
        gl = self.embed_glob(glob)[:, None, :]
        for layer in self.layers:
            h = layer(h, adj, gl)
        return self.norm(h)


class Predictor(nn.Module):
    """Context latents (hidden patches swapped for a mask token) -> predicted target latents."""

    def __init__(self, hidden: int, layers: int, n_nodes: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(n_nodes, hidden) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        self.layers = nn.ModuleList(Layer(hidden) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))

    def forward(self, ctx, glob_latent, adj, hidden):
        h = torch.where(hidden[..., None], self.mask_token.expand_as(ctx), ctx) + self.pos
        for layer in self.layers:
            h = layer(h, adj, glob_latent)
        return self.out(h)


class StaticJEPA(nn.Module):
    def __init__(self, n_in: int, n_glob: int, hidden: int = 64, layers: int = 3,
                 pred_layers: int = 2, rows: int = 8, cols: int = 8):  # fmt: skip
        super().__init__()
        self.config = {"n_in": n_in, "n_glob": n_glob, "hidden": hidden, "layers": layers,
                       "pred_layers": pred_layers, "rows": rows, "cols": cols}  # fmt: skip
        n = rows * cols
        self.register_buffer("mask", grid_mask(rows, cols))
        self.context = Encoder(n_in, n_glob, hidden, layers, n)
        self.target = copy.deepcopy(self.context)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(hidden, pred_layers, n)

    def adjacency(self, elev, on, hidden=None):
        if hidden is not None:
            on = on * (~hidden).float()  # a hidden patch shows no hyphal edges
        return adjacency(self.mask, elev, on)

    @torch.no_grad()
    def encode(self, nodes, glob, on, elev) -> torch.Tensor:
        """(B, N, hidden) target-encoder latents of full snapshots: what the probes read."""
        return self.target(nodes, glob, self.adjacency(elev, on))

    def predict(self, nodes, glob, on, elev, hidden) -> tuple[torch.Tensor, torch.Tensor]:
        """Context latents and predicted latents for every patch, given which are hidden."""
        adj = self.adjacency(elev, on, hidden)
        ctx = self.context(nodes, glob, adj, hidden)
        gl = self.context.embed_glob(glob)[:, None, :]
        return ctx, self.predictor(ctx, gl, adj, hidden)

    def loss(self, nodes, glob, on, elev, hidden, var_weight: float = 1.0) -> dict:
        with torch.no_grad():
            tgt = self.target(nodes, glob, self.adjacency(elev, on))
            tgt = fn.layer_norm(tgt, tgt.shape[-1:])
        ctx, pred = self.predict(nodes, glob, on, elev, hidden)
        jepa = fn.smooth_l1_loss(pred[hidden], tgt[hidden])
        var = variance_hinge(ctx[~hidden])
        return {"loss": jepa + var_weight * var, "jepa": jepa, "var": var}

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for pt, pc in zip(self.target.parameters(), self.context.parameters(), strict=True):
            pt.mul_(momentum).add_(pc.detach(), alpha=1 - momentum)


def variance_hinge(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance term: mean over dims of relu(gamma - std across rows)."""
    std = torch.sqrt(z.reshape(-1, z.shape[-1]).var(dim=0) + eps)
    return fn.relu(gamma - std).mean()


def block_mask(batch: int, rows: int, cols: int, ratio: float, gen: torch.Generator,
               min_side: int = 2, max_side: int = 3) -> torch.Tensor:  # fmt: skip
    """(B, N) bool: random square blocks of patches hidden until `ratio` of them are.

    Blocks (I-JEPA style) rather than scattered patches, so a hidden patch can't
    just be interpolated from its four neighbours.
    """
    n = rows * cols
    want = max(1, round(ratio * n))
    out = torch.zeros(batch, rows, cols, dtype=torch.bool)
    for b in range(batch):
        while int(out[b].sum()) < want:
            side = int(torch.randint(min_side, max_side + 1, (1,), generator=gen))
            r = int(torch.randint(0, rows - side + 1, (1,), generator=gen))
            c = int(torch.randint(0, cols - side + 1, (1,), generator=gen))
            out[b, r : r + side, c : c + side] = True
    return out.reshape(batch, n)


def momentum_at(step: int, total: int, start: float = 0.996, end: float = 1.0) -> float:
    """EMA momentum, cosine-ramped from `start` to `end` over training (BYOL / I-JEPA)."""
    frac = min(step / max(total, 1), 1.0)
    return end - (end - start) * (math.cos(math.pi * frac) + 1) / 2


# --- temporal JEPA: predict the latents of t + k from t -------------------------------------


class TemporalPredictor(nn.Module):
    """Latents at t and their change since t - 7 d, plus a horizon embedding -> latents at t + k."""

    def __init__(self, hidden: int, layers: int, n_nodes: int, n_horizons: int):
        super().__init__()
        self.inp = mlp(2 * hidden, hidden, hidden)
        self.horizon = nn.Embedding(n_horizons, hidden)
        self.pos = nn.Parameter(torch.randn(n_nodes, hidden) * 0.02)
        self.layers = nn.ModuleList(Layer(hidden) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))

    def forward(self, z_now, z_past, glob_latent, adj, k):
        hk = self.horizon(k)[:, None, :]
        h = self.inp(torch.cat([z_now, z_now - z_past], dim=-1)) + self.pos + hk
        gl = glob_latent + hk
        for layer in self.layers:
            h = layer(h, adj, gl)
        return self.out(h)


class TemporalJEPA(nn.Module):
    """Context encoder on the snapshot at t (fully visible) -> predicted target latents at t + k.

        x_t, x_{t-7} ──► context encoder ──► predictor(k) ──► ẑ_{t+k} ┐
        x_{t+k} ────────► target encoder (EMA, no grad) ──► z_{t+k} ───┘ smooth-L1

    One predictor serves every horizon, told which by a learned embedding.
    """

    def __init__(self, n_in: int, n_glob: int, hidden: int = 64, layers: int = 3,
                 pred_layers: int = 2, n_horizons: int = 3, rows: int = 8,
                 cols: int = 8):  # fmt: skip
        super().__init__()
        self.config = {"n_in": n_in, "n_glob": n_glob, "hidden": hidden, "layers": layers,
                       "pred_layers": pred_layers, "n_horizons": n_horizons, "rows": rows,
                       "cols": cols}  # fmt: skip
        n = rows * cols
        self.register_buffer("mask", grid_mask(rows, cols))
        self.context = Encoder(n_in, n_glob, hidden, layers, n)
        self.target = copy.deepcopy(self.context)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = TemporalPredictor(hidden, pred_layers, n, n_horizons)

    def init_encoders(self, static_state: dict) -> None:
        """Start both encoders from a trained StaticJEPA's, each from its namesake."""
        for name in ("context", "target"):
            sub = {k[len(name) + 1 :]: v.float() for k, v in static_state.items()
                   if k.startswith(name + ".")}  # fmt: skip
            getattr(self, name).load_state_dict(sub)

    def _encode(self, enc, s):
        return enc(s["nodes"], s["glob"], adjacency(self.mask, s["elev"], s["on"]))

    def predict(self, now: dict, past: dict, k: torch.Tensor):
        """(B, N, D) context latents at t, and (B, N, D) predicted latents at t + k[b]."""
        z_now, z_past = self._encode(self.context, now), self._encode(self.context, past)
        gl = self.context.embed_glob(now["glob"])[:, None, :]
        adj = adjacency(self.mask, now["elev"], now["on"])
        return z_now, self.predictor(z_now, z_past, gl, adj, k)

    def loss(self, now: dict, past: dict, futures: list[dict], var_weight: float = 0.1) -> dict:
        """futures[i] is the snapshot at t + horizon i. All horizons in one stacked batch."""
        h = len(futures)
        b = now["nodes"].shape[0]
        with torch.no_grad():
            fut = {key: torch.cat([f[key] for f in futures]) for key in futures[0]}
            tgt = self._encode(self.target, fut)
            tgt = fn.layer_norm(tgt, tgt.shape[-1:])
        z_now, z_past = self._encode(self.context, now), self._encode(self.context, past)
        gl = self.context.embed_glob(now["glob"])[:, None, :]
        adj = adjacency(self.mask, now["elev"], now["on"])
        k = torch.arange(h).repeat_interleave(b)
        rep = lambda t: t.repeat(h, *([1] * (t.dim() - 1)))  # noqa: E731
        pred = self.predictor(rep(z_now), rep(z_past), rep(gl), rep(adj), k)
        per_h = fn.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=(1, 2)).view(h, b).mean(1)
        jepa = per_h.mean()
        var = variance_hinge(z_now)
        return {"loss": jepa + var_weight * var, "jepa": jepa, "var": var, "per_horizon": per_h}

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for pt, pc in zip(self.target.parameters(), self.context.parameters(), strict=True):
            pt.mul_(momentum).add_(pc.detach(), alpha=1 - momentum)
