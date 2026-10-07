"""Linear probes and collapse statistics for frozen representations (numpy only).

A probe is ridge regression from a frozen per-patch representation to a
current-state quantity. The ridge strength is picked on validation worlds and
R² is reported on test worlds, so a probe never sees the worlds it is scored on.

Probe targets (current state, same definitions as targets.state_heads):
    biomass        log(1 + patch plant C)
    contamination  log(1 + patch contaminant)
    moisture       root-zone relative moisture
    network        whether the patch is on the fungal network (0/1)
    links          whether an adjacent pair is a fungal link (0/1); probed from the
                   pair's representations summed, the symmetric readout of the link head

Collapse statistics on a (rows, dims) matrix of embeddings:
    std            mean over dimensions of the per-dimension standard deviation
    effective rank exp(entropy of the normalized singular values) (RankMe); 1 = collapsed
"""

import numpy as np

from worldmodel.graph import adjacent_pairs, links, on_network
from worldmodel.targets import state_heads

NODE_PROBES = ("biomass", "contamination", "moisture", "network")
PROBES = (*NODE_PROBES, "links")
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def probe_targets(x: np.ndarray, rows: int = 8, cols: int = 8) -> dict[str, np.ndarray]:
    """Raw snapshots (S, N, F) -> {probe: (S, N) or (S, E) for links}."""
    out = dict(state_heads(x))
    out["network"] = on_network(x).astype(np.float64)
    out["links"] = links(x, rows, cols).astype(np.float64)
    return out


def pair_features(z: np.ndarray, rows: int = 8, cols: int = 8) -> np.ndarray:
    """(S, N, D) per-patch representations -> (S, E, D) per-pair, summed (order-free)."""
    pairs = adjacent_pairs(rows, cols)
    return z[:, pairs[:, 0]] + z[:, pairs[:, 1]]


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class Ridge:
    """Closed-form ridge with an unpenalized intercept, on standardized inputs."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        x = x.astype(np.float64)
        self.mu, sd = x.mean(0), x.std(0)
        self.sd = np.where(sd > 1e-8, sd, 1.0)
        xs = (x - self.mu) / self.sd
        self.y0 = float(y.mean())
        a = xs.T @ xs + self.alpha * len(xs) * np.eye(xs.shape[1])
        self.w = np.linalg.solve(a, xs.T @ (y - self.y0))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mu) / self.sd) @ self.w + self.y0


def fit_probe(train: tuple, val: tuple, test: tuple, alphas=ALPHAS) -> dict:
    """Each split is (features (rows, D), target (rows,)). Alpha chosen on val R²."""
    best = None
    for a in alphas:
        model = Ridge(a).fit(*train)
        score = r2(val[1], model.predict(val[0]))
        if best is None or score > best[0]:
            best = (score, a, model)
    score, alpha, model = best
    return {"val_r2": score, "test_r2": r2(test[1], model.predict(test[0])),
            "train_r2": r2(train[1], model.predict(train[0])), "alpha": alpha}  # fmt: skip


def run_probes(feats: dict, targets: dict, rows: int = 8, cols: int = 8,
               rows_mask: dict | None = None) -> dict:  # fmt: skip
    """Probe every target from one representation.

    feats:     {split: (S, N, D)} frozen per-patch representations
    targets:   {split: probe_targets(...)}
    rows_mask: optional {split: (S, N) bool}; node probes only use those patches
               (e.g. the hidden ones in a masked probe). The links probe is skipped then.
    """
    out = {}
    for name in PROBES:
        if name == "links" and rows_mask is not None:
            continue
        splits = []
        for split in ("train", "val", "test"):
            z, y = feats[split], targets[split][name]
            if name == "links":
                z = pair_features(z, rows, cols)
            if rows_mask is not None:
                keep = rows_mask[split]
                splits.append((z[keep], y[keep]))
            else:
                splits.append((z.reshape(-1, z.shape[-1]), y.reshape(-1)))
        out[name] = fit_probe(*splits)
    out["mean_test_r2"] = float(np.mean([v["test_r2"] for v in out.values()]))
    return out


def collapse_stats(z: np.ndarray) -> dict:
    """Embedding spread and effective rank of (rows, D) embeddings (any leading shape)."""
    z = z.reshape(-1, z.shape[-1]).astype(np.float64)
    zc = z - z.mean(0)
    s = np.linalg.svd(zc, compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return {"std": float(z.std(0).mean()), "min_std": float(z.std(0).min()),
            "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
            "dims": int(z.shape[1])}  # fmt: skip
