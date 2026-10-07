"""Forecast targets, model inputs, and persistence baselines.

Heads, each at three horizons (HORIZONS_DAYS):
    biomass        change in log(1 + patch plant C)
    contamination  change in log(1 + patch contaminant)
    moisture       change in root-zone relative moisture
    mortality      share of vegetated cells (> 25 g C) losing >= 30% of plant C
    links          whether each adjacent patch pair is a fungal link

Persistence baselines: no change (biomass, contamination, moisture), the
mortality rate over the previous window of the same length, and today's links.

Inputs per patch: transformed features now, plus their change over the last
7 days; the atmosphere node carries season and recent weather.
"""

import numpy as np

from worldmodel.graph import PATCH_FEATURES, F, links

HORIZONS_DAYS = (1, 7, 30)
NODE_HEADS = ("biomass", "contamination", "moisture", "mortality")
CONTINUOUS = ("biomass", "contamination", "moisture")
TREND_DAYS = 7
VEGETATED_C = 25.0
DIE_OFF = 0.7  # a cell "dies back" if it keeps less than 70% of its plant C
LOG_FEATURES = {"plant_c", "litter_c", "som_c", "bacteria_c", "saprotrophs_c", "mycorrhiza_c",
                "insects_c", "mineral_n", "mineral_p", "contaminant"}  # fmt: skip


def transform(x: np.ndarray) -> np.ndarray:
    """Raw patch features -> model scale (log1p for masses; C:N / 100)."""
    out = x.astype(np.float64).copy()
    for name in PATCH_FEATURES:
        i = F[name]
        if name in LOG_FEATURES:
            out[..., i] = np.log1p(np.maximum(out[..., i], 0.0))
        elif name == "plant_cn":
            out[..., i] = out[..., i] / 100.0
    return out


def node_inputs(x_now: np.ndarray, x_week_ago: np.ndarray) -> np.ndarray:
    """(..., N, 2F): transformed features and their 7-day change."""
    a, b = transform(x_now), transform(x_week_ago)
    return np.concatenate([a, a - b], axis=-1)


def state_heads(x: np.ndarray) -> dict[str, np.ndarray]:
    t = transform(x)
    return {"biomass": t[..., F["plant_c"]], "contamination": t[..., F["contaminant"]],
            "moisture": x[..., F["moisture"]]}  # fmt: skip


def mortality(plant_from: np.ndarray, plant_to: np.ndarray, size: int = 8) -> np.ndarray:
    """(N,) share of vegetated cells per patch that died back between two cell fields."""
    veg = plant_from > VEGETATED_C
    died = veg & (plant_to < DIE_OFF * plant_from)
    h, w = plant_from.shape
    shape = (h // size, size, w // size, size)
    n_veg = veg.reshape(shape).sum(axis=(1, 3))
    n_died = died.reshape(shape).sum(axis=(1, 3))
    return np.where(n_veg > 0, n_died / np.maximum(n_veg, 1), 0.0).ravel()


def valid_days(n_days: int) -> np.ndarray:
    """Snapshot indices with a full trend window behind and every horizon ahead."""
    back = max(max(HORIZONS_DAYS), TREND_DAYS)
    return np.arange(back, n_days - max(HORIZONS_DAYS))


def build_samples(run: dict) -> dict[str, np.ndarray]:
    """All samples from one run: inputs, targets, and persistence baselines."""
    x, g, plant = run["x"], run["g"], run["plant"]
    rows, cols = (int(v) for v in run["grid"])
    days = valid_days(len(x))
    heads_now = state_heads(x)
    out: dict[str, list] = {k: [] for k in ("nodes", "glob", "on", "elev", "link_now", "day")}
    for name in NODE_HEADS:
        out[f"y_{name}"], out[f"p_{name}"] = [], []
    out["y_links"], out["p_links"] = [], []
    link_all = links(x, rows, cols)
    for t in days:
        out["nodes"].append(node_inputs(x[t], x[t - TREND_DAYS]))
        out["glob"].append(g[t])
        out["on"].append(x[t, :, F["mycorrhiza_c"]] >= 2.0)
        out["elev"].append(x[t, :, F["elevation"]])
        out["link_now"].append(link_all[t])
        out["day"].append(t)
        for name in CONTINUOUS:
            now = heads_now[name][t]
            out[f"y_{name}"].append([heads_now[name][t + h] - now for h in HORIZONS_DAYS])
            out[f"p_{name}"].append([np.zeros_like(now) for _ in HORIZONS_DAYS])
        out["y_mortality"].append([mortality(plant[t], plant[t + h]) for h in HORIZONS_DAYS])
        out["p_mortality"].append([mortality(plant[t - h], plant[t]) for h in HORIZONS_DAYS])
        out["y_links"].append([link_all[t + h] for h in HORIZONS_DAYS])
        out["p_links"].append([link_all[t] for _ in HORIZONS_DAYS])
    arrays = {k: np.array(v) for k, v in out.items()}
    for k in list(arrays):  # (S, H, N) -> (S, N, H): horizons last
        if (k.startswith("y_") or k.startswith("p_")) and arrays[k].ndim == 3:
            arrays[k] = arrays[k].transpose(0, 2, 1)
    return arrays


def scores(pred: dict[str, np.ndarray], data: dict[str, np.ndarray]) -> dict:
    """Per head and horizon: model error, persistence error, and skill (1 - model/persistence).

    MAE for continuous heads and mortality; Brier score for link probabilities.
    """
    out = {}
    for name in (*NODE_HEADS, "links"):
        y, p, m = data[f"y_{name}"].astype(float), data[f"p_{name}"].astype(float), pred[name]
        for k, h in enumerate(HORIZONS_DAYS):
            if name == "links":
                err_m = np.mean((m[..., k] - y[..., k]) ** 2)
                err_p = np.mean((p[..., k] - y[..., k]) ** 2)
            else:
                err_m = np.mean(np.abs(m[..., k] - y[..., k]))
                err_p = np.mean(np.abs(p[..., k] - y[..., k]))
            skill = 1 - err_m / err_p if err_p > 0 else 0.0
            out[f"{name}@{h}d"] = {"model": float(err_m), "persistence": float(err_p),
                                   "skill": float(skill)}  # fmt: skip
    for h in HORIZONS_DAYS:
        out[f"mean_skill@{h}d"] = float(
            np.mean(
                [v["skill"] for k, v in out.items() if k.endswith(f"@{h}d") and "mean" not in k]
            )
        )
    return out
