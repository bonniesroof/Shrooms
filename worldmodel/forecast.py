"""Torch-free inference for the trained forecaster, used by the game server and CI.

Loads worldmodel/forecaster.npz (weights, normalization, scales) and runs the
same forward pass as gnn.Forecaster in numpy. A test checks the two agree.
"""

import json
from pathlib import Path

import numpy as np

from worldmodel.graph import F, adjacent_pairs, links
from worldmodel.targets import CONTINUOUS, HORIZONS_DAYS, NODE_HEADS, TREND_DAYS, node_inputs

DEFAULT_PATH = Path(__file__).resolve().parent / "forecaster.npz"


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyForecaster:
    def __init__(self, path: Path = DEFAULT_PATH):
        with np.load(path, allow_pickle=False) as z:
            self.w = {k: z[k].astype(np.float64) for k in z.files if k != "meta"}
            self.meta = json.loads(str(z["meta"]))
        c = self.meta["config"]
        self.rows, self.cols, self.layers = c["rows"], c["cols"], c["layers"]
        self.pairs = adjacent_pairs(self.rows, self.cols)
        n = self.rows * self.cols
        self.mask = np.zeros((n, n))
        self.mask[self.pairs[:, 0], self.pairs[:, 1]] = 1
        self.mask[self.pairs[:, 1], self.pairs[:, 0]] = 1

    # --- layers mirroring torch ---------------------------------------------
    def _lin(self, name, x):
        y = x @ self.w[f"{name}.weight"].T
        b = self.w.get(f"{name}.bias")
        return y + b if b is not None else y

    def _mlp(self, name, x):
        return self._lin(f"{name}.2", silu(self._lin(f"{name}.0", x)))

    def _norm(self, name, x, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * self.w[f"{name}.weight"] + self.w[f"{name}.bias"]

    def _adjacency(self, elev, on):
        b, n = elev.shape
        adj = np.broadcast_to(self.mask, (b, n, n))
        drop = np.maximum(elev[:, None, :] - elev[:, :, None], 0)
        mats = np.stack([adj, adj * drop, adj * on[:, :, None] * on[:, None, :]], axis=1)
        deg = mats.sum(-1, keepdims=True)
        return np.divide(mats, deg, out=np.zeros_like(mats), where=deg > 0)

    def raw(self, nodes, glob, on, elev, link_now):
        """The forward pass on normalized inputs. Same outputs as gnn.Forecaster."""
        adj = self._adjacency(elev, on.astype(np.float64))
        h = self._mlp("encode", nodes)
        gl = self._mlp("encode_glob", glob)[:, None, :]
        b, n, d = h.shape
        for i in range(self.layers):
            p = f"layers.{i}"
            z = self._norm(f"{p}.norm", h)
            m = self._lin(f"{p}.msg", z).reshape(b, n, 3, d)
            msgs = [adj[:, k] @ m[:, :, k] for k in range(3)]
            h = h + self._mlp(
                f"{p}.update", np.concatenate([z, *msgs, np.broadcast_to(gl, (b, n, d))], -1)
            )
        node = self._mlp("node_head", h).reshape(b, n, len(NODE_HEADS), len(HORIZONS_DAYS))
        hi, hj = h[:, self.pairs[:, 0]], h[:, self.pairs[:, 1]]
        sign = (2 * link_now - 1)[..., None]
        link = self._mlp("link_head", np.concatenate([hi + hj, np.abs(hi - hj), sign], -1))
        return node, link + self.w["link_prior"] * sign

    # --- from raw features to forecasts --------------------------------------
    def prepare(self, x_now, x_week_ago, g):
        """Batch arrays (leading axis) of raw features -> normalized model inputs."""
        m = self.meta
        nodes = (node_inputs(x_now, x_week_ago) - m["node_mean"]) / m["node_std"]
        glob = (g - m["glob_mean"]) / m["glob_std"]
        on = x_now[..., F["mycorrhiza_c"]] >= 2.0
        link_now = links(x_now, self.rows, self.cols).astype(np.float64)
        return nodes, glob, on, x_now[..., F["elevation"]], link_now

    def decode(self, node, link) -> dict[str, np.ndarray]:
        """Model outputs -> head predictions in target units, horizons last."""
        out = {}
        scales = self.meta["target_scale"]
        for k, name in enumerate(NODE_HEADS):
            v = node[:, :, k, :]
            out[name] = sigmoid(v) if name == "mortality" else v * np.array(scales[name])
        out["links"] = sigmoid(link)
        return out

    def predict(self, x_now, x_week_ago, g) -> dict[str, np.ndarray]:
        return self.decode(*self.raw(*self.prepare(x_now, x_week_ago, g)))

    def forecast_one(self, x_now: np.ndarray, x_week_ago: np.ndarray, g: np.ndarray) -> dict:
        """One snapshot -> {head: {horizon_days: per-patch (or per-pair) values}}."""
        pred = self.predict(x_now[None], x_week_ago[None], g[None])
        return {name: {h: pred[name][0, :, k] for k, h in enumerate(HORIZONS_DAYS)}
                for name in (*NODE_HEADS, "links")}  # fmt: skip


__all__ = ["NumpyForecaster", "CONTINUOUS", "TREND_DAYS"]
