"""Scripted players, for balancing the scenario and testing it end to end.

`idle` never acts. `greedy` is a sensible but unsophisticated gardener: every two
weeks it spends what it can on the worst spot for each objective. If `greedy`
cannot win, the scenario is too hard; if `idle` wins, it is too easy.
"""

import numpy as np

from server.session import TOOLS, GameSession
from sim import TICKS_PER_DAY
from sim.patches import patch_means


def _worst(field: np.ndarray, site, size: int = 8, highest: bool = True) -> tuple[float, float]:
    """Centre of the site patch with the highest (or lowest) mean value."""
    x0, y0, x1, y1 = site
    means = patch_means(field[y0:y1, x0:x1], size)
    r, c = np.unravel_index(np.argmax(means) if highest else np.argmin(means), means.shape)
    return x0 + c * size + size / 2, y0 + r * size + size / 2


def idle(session: GameSession) -> None:
    return None


def greedy(session: GameSession) -> None:
    """Spend on whichever objective is furthest behind, where it will help most."""
    s, site = session.sim.state, session.scenario.site
    x0, y0, x1, y1 = site
    size = session.sim.params.network.patch_size
    clean = patch_means(s.contaminant[y0:y1, x0:x1], size) < 3.0  # fungi can live here
    fungal = patch_means(s.mycorrhiza.c[y0:y1, x0:x1], size)
    lagging = sorted(session.status["objectives"], key=lambda o: o["progress"])
    for objective in lagging:
        key = objective["key"]
        if objective["done"]:
            continue
        if key == "contaminant_left":
            plan = [("inoculate_bacteria", _worst(s.contaminant, site)),
                    ("compost", _worst(s.contaminant, site))]  # fmt: skip
        elif key == "network_patches":
            gaps = np.where(clean, fungal, np.inf)  # thinnest network on clean ground
            if not np.isfinite(gaps).any():
                continue
            r, c = np.unravel_index(np.argmin(gaps), gaps.shape)
            spot = (x0 + c * size + size / 2, y0 + r * size + size / 2)
            plan = [("inoculate_mycorrhiza", spot), ("seed", spot)]
        else:
            plan = [("seed", _worst(s.plant.c, site, highest=False))]
        for name, (x, y) in plan:
            if session.budget >= TOOLS[name].cost:
                session.use_tool(name, x, y)


def play(session: GameSession, player, every_days: int = 14, max_days: int | None = None) -> dict:
    """Run a scenario to the end (or max_days), letting `player` act on its schedule."""
    limit = max_days if max_days is not None else session.scenario.deadline_days + 1
    day = 0
    while session.status["status"] == "playing" and day < limit:
        if day % every_days == 0:
            player(session)
        session.advance(TICKS_PER_DAY)
        day += 1
    return session.status
