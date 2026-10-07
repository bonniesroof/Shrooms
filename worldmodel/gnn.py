"""Supervised GNN forecaster (PyTorch): typed message passing on the patch graph.

    encode patches and the atmosphere node
    3 x  [ messages along adjacent / downslope / hyphal edges  ->  residual update ]
    node head: biomass, contamination, moisture, mortality  x  24 h, 7 d, 30 d
    edge head: fungal link probability per adjacent pair     x  24 h, 7 d, 30 d

Adjacency is built inside the model from elevation (downslope) and network
membership (hyphal), so it adapts to each world and moment. Plain dense
matrices: at 64 nodes that is faster and clearer than a sparse GNN library.
forecast.py mirrors this forward pass in numpy for torch-free inference.
"""

import torch
from torch import nn

from worldmodel.graph import adjacent_pairs
from worldmodel.targets import HORIZONS_DAYS, NODE_HEADS


def mlp(n_in: int, n_hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_out))


def adjacency(mask, elev, on):
    """(B, 3, N, N) row-normalized adjacency, target <- source: adjacent, downslope, hyphal."""
    b, n = elev.shape
    adj = mask.expand(b, n, n)
    drop = (elev[:, None, :] - elev[:, :, None]).clamp(min=0)  # [t, s] = elev_s - elev_t
    down = adj * drop
    hyph = adj * on[:, :, None] * on[:, None, :]
    mats = torch.stack([adj, down, hyph], dim=1)
    deg = mats.sum(dim=-1, keepdim=True)
    return torch.where(deg > 0, mats / deg.clamp(min=1e-12), torch.zeros_like(mats))


class Layer(nn.Module):
    def __init__(self, hidden: int, n_types: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.msg = nn.Linear(hidden, hidden * n_types, bias=False)
        self.update = mlp(hidden * (n_types + 2), hidden, hidden)
        self.n_types = n_types

    def forward(self, h, adj, gl):
        b, n, d = h.shape
        z = self.norm(h)
        m = self.msg(z).view(b, n, self.n_types, d)
        msgs = [adj[:, k] @ m[:, :, k] for k in range(self.n_types)]
        return h + self.update(torch.cat([z, *msgs, gl.expand(b, n, d)], dim=-1))


class Forecaster(nn.Module):
    def __init__(self, n_in: int, n_glob: int, hidden: int = 64, layers: int = 3,
                 rows: int = 8, cols: int = 8):  # fmt: skip
        super().__init__()
        self.config = {"n_in": n_in, "n_glob": n_glob, "hidden": hidden, "layers": layers,
                       "rows": rows, "cols": cols}  # fmt: skip
        pairs = torch.as_tensor(adjacent_pairs(rows, cols))
        n = rows * cols
        mask = torch.zeros(n, n)
        mask[pairs[:, 0], pairs[:, 1]] = 1
        mask[pairs[:, 1], pairs[:, 0]] = 1
        self.register_buffer("mask", mask)
        self.register_buffer("pairs", pairs)
        self.encode = mlp(n_in, hidden, hidden)
        self.encode_glob = mlp(n_glob, hidden, hidden)
        self.layers = nn.ModuleList(Layer(hidden) for _ in range(layers))
        self.node_head = mlp(hidden, hidden, len(NODE_HEADS) * len(HORIZONS_DAYS))
        self.link_head = mlp(2 * hidden + 1, hidden, len(HORIZONS_DAYS))
        self.link_prior = nn.Parameter(torch.tensor(4.0))  # links persist unless told otherwise

    def forward(self, nodes, glob, on, elev, link_now):
        """Returns node outputs (B, N, heads, horizons) and link logits (B, E, horizons).

        Node outputs are scaled deltas for continuous heads and a logit for mortality.
        """
        adj = adjacency(self.mask, elev, on)
        h = self.encode(nodes)
        gl = self.encode_glob(glob)[:, None, :]
        for layer in self.layers:
            h = layer(h, adj, gl)
        b, n, _ = h.shape
        node = self.node_head(h).view(b, n, len(NODE_HEADS), len(HORIZONS_DAYS))
        hi, hj = h[:, self.pairs[:, 0]], h[:, self.pairs[:, 1]]
        sign = (2 * link_now - 1)[..., None]
        link = self.link_head(torch.cat([hi + hj, (hi - hj).abs(), sign], dim=-1))
        return node, link + self.link_prior * sign
