"""Forcing between two snapshots: the interventions applied and the weather that fell (numpy).

The action-conditioned JEPA is told what happens between t and t + k:

    interventions  per patch, one channel per kind (FORCING_KINDS), summed over the
                   intents applied in the window, then log1p:
                       spill                       contaminant added, g
                       inoculate, amend, seed      carbon added, g C
                       disturb, excavate           mean fraction of the patch hit
                       irrigate                    mean water added, mm
    weather        mean temperature, mean daily rain and mean shortwave over the window

A snapshot at day d (dataset index d) is taken at tick ticks[d] and includes every
intent with tick < ticks[d], so the window (t, t + k] holds the intents with
ticks[t] <= tick < ticks[t + k], and the weather of days t + 1 .. t + k.
Cumulative sums over days make every window a difference of two rows.
"""

import json

import numpy as np
from pydantic import TypeAdapter

from sim.intents import Intent, blob, peak_blob

FORCING_KINDS = ("spill", "disturb", "excavate", "inoculate", "amend", "seed", "irrigate")
WEATHER = ("temp_c", "rain_mm_per_day", "shortwave")
_ADAPTER = TypeAdapter(Intent)


def parse_intents(blob_json: str | np.ndarray) -> list:
    return [_ADAPTER.validate_python(d) for d in json.loads(str(blob_json))]


def footprint(intent, rows: int = 8, cols: int = 8, size: int = 8) -> np.ndarray:
    """(N, len(FORCING_KINDS)) raw per-patch amounts of one intent."""
    shape = (rows * size, cols * size)
    out = np.zeros((rows * cols, len(FORCING_KINDS)))
    k = intent.kind
    if k in ("disturb", "excavate"):
        cells = np.zeros(shape)
        cells[intent.y0 : intent.y1, intent.x0 : intent.x1] = intent.fraction
        field, how = cells, "mean"
    elif k == "irrigate":
        field, how = intent.water_mm * peak_blob(shape, intent.x, intent.y, intent.radius), "mean"
    elif k == "spill":
        field, how = intent.mass_g * blob(shape, intent.x, intent.y, intent.radius), "sum"
    elif k in ("inoculate", "amend", "seed"):
        field, how = intent.mass_c_g * blob(shape, intent.x, intent.y, intent.radius), "sum"
    else:
        return out  # kinds the research dataset never draws
    patches = field.reshape(rows, size, cols, size)
    agg = patches.sum(axis=(1, 3)) if how == "sum" else patches.mean(axis=(1, 3))
    out[:, FORCING_KINDS.index(k)] = agg.ravel()
    return out


def cumulative_forcing(intents: list, ticks: np.ndarray, rows: int = 8,
                       cols: int = 8) -> np.ndarray:  # fmt: skip
    """(days, N, C): row d sums the footprints of every intent applied before snapshot d
    (tick < ticks[d]); the window between snapshots t and t + k is row t + k minus row t."""
    per_day = np.zeros((len(ticks), rows * cols, len(FORCING_KINDS)))
    for intent in intents:
        d = int(np.searchsorted(ticks, intent.tick, side="right"))  # first snapshot after it
        if d < len(ticks):
            per_day[d] += footprint(intent, rows, cols)
    return np.cumsum(per_day, axis=0)


def window(cum: np.ndarray, t, k: int) -> np.ndarray:
    """Forcing between snapshots t and t + k (t may be an index array)."""
    return cum[np.asarray(t) + k] - cum[np.asarray(t)]


def weather_cumsum(w: np.ndarray) -> np.ndarray:
    """(days, 3) daily (mean temp, rain total, mean shortwave) -> cumulative sums by day."""
    return np.cumsum(w, axis=0)


def weather_window(wcum: np.ndarray, t, k: int) -> np.ndarray:
    """Means over days t + 1 .. t + k: temperature, daily rain, shortwave."""
    return (wcum[np.asarray(t) + k] - wcum[np.asarray(t)]) / k


def transform_forcing(f: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(f, 0.0)) / scale


def forcing_scale(f: np.ndarray) -> np.ndarray:
    """Per-channel scale: the std of log1p amounts where the channel is active (1 if never)."""
    z = np.log1p(np.maximum(f.reshape(-1, f.shape[-1]), 0.0))
    out = np.ones(z.shape[-1])
    for c in range(z.shape[-1]):
        on = z[:, c][z[:, c] > 1e-9]
        if len(on) > 1 and on.std() > 1e-9:
            out[c] = float(np.sqrt(np.mean(on**2)))
    return out
