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


# --- forecast probes: frozen representations -> the Phase 4 forecast heads -------------------


class LinearProbe:
    """Linear map on standardized inputs. Ridge for squared error, IRLS for absolute error.

    The Gram matrix is computed once, so trying several ridge strengths is cheap.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = x.astype(np.float64)
        self.mu, sd = x.mean(0), x.std(0)
        self.sd = np.where(sd > 1e-8, sd, 1.0)
        self.xs = np.concatenate([(x - self.mu) / self.sd, np.ones((len(x), 1))], 1)
        self.y = y.astype(np.float64)
        self.gram = self.xs.T @ self.xs
        self.xty = self.xs.T @ self.y

    def _reg(self, alpha: float) -> np.ndarray:
        reg = alpha * len(self.xs) * np.eye(self.xs.shape[1])
        reg[-1, -1] = 0.0  # intercept unpenalized
        return reg

    def ridge(self, alpha: float) -> np.ndarray:
        return np.linalg.solve(self.gram + self._reg(alpha), self.xty)

    def lad(self, alpha: float, iters: int = 12, eps: float = 1e-6) -> np.ndarray:
        """Least absolute deviations (+ ridge penalty) by iteratively reweighted least squares."""
        w = self.ridge(alpha)
        scale = max(float(np.median(np.abs(self.y))), 1e-9)
        for _ in range(iters):
            r = np.abs(self.y - self.xs @ w)
            q = 1.0 / np.maximum(r, eps * scale)
            q /= q.mean()
            a = (self.xs * q[:, None]).T @ self.xs
            w = np.linalg.solve(a + self._reg(alpha), self.xs.T @ (q * self.y))
        return w

    def predict(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        return ((x - self.mu) / self.sd) @ w[:-1] + w[-1]


FORECAST_ALPHAS = (1e-3, 1e-1, 10.0)


def forecast_probes(feats: dict, samples: dict, rows: int = 8, cols: int = 8,
                    alphas=FORECAST_ALPHAS, max_rows: int = 60_000,
                    seed: int = 0) -> dict[str, np.ndarray]:  # fmt: skip
    """Linear probes onto the Phase 4 targets (targets.build_samples), one per head and horizon.

    feats:   {split: (S, H, N, D)} per-horizon representations, or a tuple of them to
             concatenate; H may be 1 when a representation doesn't depend on the horizon
    samples: {split: build_samples(...)}, with y_<head> and p_<head> of shape (S, N or E, H)

    Links probes predict the change from today's links (persistence plus a linear
    correction); the other heads' persistence is zero change, except mortality,
    whose persistence (last window's rate) is not an input, so it is fit directly.
    Probes are fit for the reported metric: least absolute deviations for the MAE
    heads, least squares for links (Brier). The ridge strength is chosen per head
    and horizon on the validation split, and each probe trains on at most
    `max_rows` random training rows (patches or pairs), which keeps it to seconds.
    Returns test-split predictions {head: (S, N or E, H)} in target units, ready
    for targets.scores; probabilities are clipped to [0, 1].
    """
    from worldmodel.targets import HORIZONS_DAYS, NODE_HEADS

    out = {}
    for name in (*NODE_HEADS, "links"):
        preds = []
        for k in range(len(HORIZONS_DAYS)):
            xs, ys, ps = {}, {}, {}
            for split in ("train", "val", "test"):
                y = samples[split][f"y_{name}"][..., k]
                sel = np.arange(y.size)
                if split == "train" and y.size > max_rows:
                    sel = np.random.default_rng(seed).permutation(y.size)[:max_rows]
                xs[split] = _gather(feats[split], k, sel, y.shape[1], name, rows, cols)
                ys[split] = y.reshape(-1)[sel].astype(np.float64)
                p = samples[split][f"p_{name}"][..., k].reshape(-1)[sel].astype(np.float64)
                ps[split] = p if name == "links" else np.zeros_like(p)
            probe = LinearProbe(xs["train"], ys["train"] - ps["train"])
            fit = probe.ridge if name == "links" else probe.lad
            best = None
            for a in alphas:
                w = fit(a)
                pv = _clip(name, ps["val"] + probe.predict(w, xs["val"]))
                d = pv - ys["val"]
                err = np.mean(d**2) if name == "links" else np.mean(np.abs(d))
                if best is None or err < best[0]:
                    best = (err, w)
            pt = _clip(name, ps["test"] + probe.predict(best[1], xs["test"]))
            preds.append(pt.reshape(samples["test"][f"y_{name}"][..., k].shape))
        out[name] = np.stack(preds, axis=-1)
    return out


def _gather(parts, k: int, sel: np.ndarray, width: int, name: str, rows: int,
            cols: int) -> np.ndarray:  # fmt: skip
    """Rows `sel` of the flattened (snapshot, node or pair) features for horizon k.

    `parts` is one (S, H, N, D) array or a tuple of them, concatenated along D after
    gathering, so a combined representation is never materialized in full.
    """
    parts = parts if isinstance(parts, tuple) else (parts,)
    snap, j = np.divmod(sel, width)
    out = []
    for z in parts:
        zk = z[:, min(k, z.shape[1] - 1)]
        if name == "links":
            pairs = adjacent_pairs(rows, cols)
            out.append(zk[snap, pairs[j, 0]] + zk[snap, pairs[j, 1]])
        else:
            out.append(zk[snap, j])
    return np.concatenate(out, axis=-1).astype(np.float64)


def _clip(name: str, p: np.ndarray) -> np.ndarray:
    return np.clip(p, 0.0, 1.0) if name in ("mortality", "links") else p
