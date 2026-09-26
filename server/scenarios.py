"""Scenarios: a starting world, setup intents, objectives and a clock.

A scenario is fully described by its params and setup intents, which go into
the replay record like any other intents, so a scenario run replays exactly.
"""

from dataclasses import dataclass, field

import numpy as np

from sim import TICKS_PER_DAY
from sim.engine import Simulation
from sim.intents import AnyIntent, Excavate, Spill
from sim.params import ContaminationParams, SimParams
from sim.patches import patch_means
from sim.plants import canopy_cover


@dataclass(frozen=True)
class Objective:
    key: str
    label: str
    target: float
    higher_is_better: bool

    def progress(self, value: float, start: float) -> float:
        """0 at the starting value, 1 when the target is reached."""
        if self.higher_is_better:
            span = self.target - start
            return 1.0 if value >= self.target else max(0.0, (value - start) / span) if span else 0
        span = start - self.target
        return 1.0 if value <= self.target else max(0.0, (start - value) / span) if span else 0


@dataclass
class Brownfield:
    """An old industrial pad in the middle of a meadow: scraped bare and soaked in
    a persistent organic contaminant. Bring it back to life before the grant runs out."""

    name: str = "brownfield"
    title: str = "Brownfield remediation"
    site: tuple[int, int, int, int] = (16, 16, 48, 48)  # x0, y0, x1, y1 (exclusive)
    deadline_days: int = 3 * 365
    start_budget: float = 150.0
    grant: float = 30.0
    grant_every_days: int = 30
    spills: tuple[tuple[float, float, float, float], ...] = (
        (24.0, 25.0, 5.0, 3500.0),  # x, y, radius, grams
        (38.0, 22.0, 4.0, 2500.0),
        (30.0, 39.0, 6.0, 4000.0),
        (41.0, 41.0, 3.5, 2000.0),
    )
    objectives: tuple[Objective, ...] = (
        Objective("contaminant_left", "Contaminant left on site", 0.25, False),
        Objective("site_cover", "Plant cover on site", 0.6, True),
        Objective("network_patches", "Site patches on the fungal network", 10, True),
    )
    start: dict[str, float] = field(default_factory=dict)

    def params(self) -> SimParams:
        # All contamination comes from the site spills, not random hotspots. It is
        # diesel-range hydrocarbon: far more biodegradable than the default PAH-like
        # contaminant, so microbes (and the player's help) can make real progress.
        return SimParams(contamination=ContaminationParams(hotspots=0, degradation_per_day=0.035))

    def setup_intents(self) -> list[AnyIntent]:
        """Strip the topsoil (plants, litter, SOM, microbes, fungi) in 16x16 blocks,
        then lay down the legacy contamination."""
        x0, y0, x1, y1 = self.site
        out: list[AnyIntent] = []
        for by in range(y0, y1, 16):
            for bx in range(x0, x1, 16):
                out.append(Excavate(
                    tick=0, agent="scenario", x0=bx, y0=by, x1=min(bx + 16, x1),
                    y1=min(by + 16, y1), fraction=0.9,
                    rationale="scenario: the old industrial pad was stripped to subsoil",
                ))  # fmt: skip
        for x, y, r, g in self.spills:
            out.append(Spill(tick=0, agent="scenario", x=x, y=y, radius=r, mass_g=g,
                             rationale="scenario: decades of drum leaks"))  # fmt: skip
        return out

    def _region(self) -> tuple[slice, slice]:
        x0, y0, x1, y1 = self.site
        return slice(y0, y1), slice(x0, x1)

    def measure(self, sim: Simulation) -> dict[str, float]:
        s, r = sim.state, self._region()
        contaminant = float(s.contaminant[r].sum())
        cover = float(canopy_cover(s.plant.c[r], sim.params.plants).mean())
        size = sim.params.network.patch_size
        fungal = patch_means(s.mycorrhiza.c[r], size)
        return {
            "contaminant_g": contaminant,
            "contaminant_left": contaminant / self.start["contaminant_g"] if self.start else 1.0,
            "site_cover": cover,
            "network_patches": float((fungal >= sim.params.network.min_network_c).sum()),
        }

    def evaluate(self, sim: Simulation) -> dict:
        """Objective progress, and whether the game is won or lost."""
        m = self.measure(sim)
        if not self.start and sim.state.tick >= 1:  # baseline once setup has applied
            self.start = {**m, "contaminant_left": 1.0}
            m["contaminant_left"] = 1.0
        rows = []
        for o in self.objectives:
            start = self.start.get(o.key, m[o.key])
            rows.append({
                "key": o.key, "label": o.label, "value": m[o.key], "target": o.target,
                "progress": o.progress(m[o.key], start),
                "done": (m[o.key] >= o.target) if o.higher_is_better else (m[o.key] <= o.target),
            })  # fmt: skip
        day = sim.state.tick // TICKS_PER_DAY
        won = all(r["done"] for r in rows)
        lost = not won and day >= self.deadline_days
        return {
            "objectives": rows,
            "progress": float(np.mean([r["progress"] for r in rows])),
            "day": day,
            "days_left": max(0, self.deadline_days - day),
            "status": "won" if won else "lost" if lost else "playing",
        }


SCENARIOS = {"brownfield": Brownfield}
