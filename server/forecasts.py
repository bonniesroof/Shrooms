"""In-game forecasts: the Phase 4 GNN, run once per sim-day on the live world.

The model needs today's graph and the one from a week ago (for trends), so the
session keeps the last 8 daily snapshots. Inference is numpy-only (no torch).
Each forecast is shipped with the model's test skill against persistence, so
the HUD can say how far to trust it.
"""

import json
import logging
from collections import deque

import numpy as np

from sim import TICKS_PER_DAY
from sim.engine import Simulation
from worldmodel.forecast import DEFAULT_PATH, NumpyForecaster
from worldmodel.graph import adjacent_pairs, links, snapshot
from worldmodel.targets import HORIZONS_DAYS, TREND_DAYS

log = logging.getLogger("shrooms.forecasts")
METRICS_PATH = DEFAULT_PATH.with_name(DEFAULT_PATH.stem + "_metrics.json")


class LiveForecasts:
    def __init__(self) -> None:
        self.model: NumpyForecaster | None = None
        self.reason = ""
        self.skill: dict = {}
        self.history: deque = deque(maxlen=TREND_DAYS + 1)
        self.latest: dict | None = None
        try:
            self.model = NumpyForecaster()
            metrics = json.loads(METRICS_PATH.read_text())["test"]
            self.skill = {
                k: round(v["skill"], 2) for k, v in metrics.items() if isinstance(v, dict)
            }
        except (OSError, KeyError, ValueError) as err:
            self.reason = f"no trained forecaster ({err})"
            log.warning("forecasts disabled: %s", self.reason)

    def daily(self, sim: Simulation) -> None:
        if self.model is None:
            return
        snap = snapshot(sim)
        if (snap.rows, snap.cols) != (self.model.rows, self.model.cols):
            self.reason = f"model expects a {self.model.rows}x{self.model.cols} patch grid"
            return
        self.history.append(snap)
        if len(self.history) <= TREND_DAYS:
            self.reason = f"warming up: forecasts start after {TREND_DAYS} days"
            return
        week_ago = self.history[0]
        out = self.model.forecast_one(snap.x, week_ago.x, snap.g)
        r3 = lambda a: np.round(a, 3).tolist()  # noqa: E731
        self.latest = {
            "made_day": sim.state.tick // TICKS_PER_DAY,
            "horizons": list(HORIZONS_DAYS),
            "heads": {head: {str(h): r3(v) for h, v in per.items()} for head, per in out.items()},
            "links_now": links(snap.x, snap.rows, snap.cols).astype(int).tolist(),
            "pairs": adjacent_pairs(snap.rows, snap.cols).tolist(),
            "grid": [snap.rows, snap.cols],
            "skill": self.skill,
            # Typical size of each change (std over training data), for color scales.
            "typical": {
                head: dict(zip([str(h) for h in HORIZONS_DAYS], v, strict=True))
                for head, v in self.model.meta["target_scale"].items()
            },
        }
        self.reason = ""

    def view(self) -> dict:
        if self.latest is None:
            return {"available": False, "reason": self.reason or "no forecast yet"}
        return {"available": True, **self.latest}
