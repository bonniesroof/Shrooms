"""Thin debug viewer: colored cells and pool time series, saved as PNGs.

Deliberately ugly and useful. The real PixiJS client arrives in Phase 3.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sim import TICKS_PER_DAY  # noqa: E402
from sim.engine import Simulation  # noqa: E402

# (title, colormap, getter)
PANELS = (
    ("Plant C (g)", "Greens", lambda s: s.plant.c),
    ("Plant C:N", "YlOrRd", lambda s: s.plant.c / np.maximum(s.plant.n, 1e-9)),
    ("Insects C (g)", "Purples", lambda s: s.insects.c),
    ("Litter C (g)", "YlOrBr", lambda s: s.litter.c),
    ("Bacteria C (g)", "Oranges", lambda s: s.bacteria.c),
    ("Saprotrophs C (g)", "copper", lambda s: s.saprotrophs.c),
    ("Mycorrhiza C (g)", "PuBu", lambda s: s.mycorrhiza.c),
    ("SOM C (g)", "pink", lambda s: s.som.c),
    ("Root-zone moisture", "Blues", lambda s: s.moisture()),
    ("Mineral N (g)", "BuGn", lambda s: s.mineral_n),
    ("Contaminant (g)", "Reds", lambda s: s.contaminant),
    ("Elevation (m)", "terrain", lambda s: s.elevation),
)


def grid_figure(sim: Simulation) -> plt.Figure:
    fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), constrained_layout=True)
    for ax, (title, cmap, get) in zip(axes.flat, PANELS, strict=True):
        im = ax.imshow(get(sim.state), cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    day = sim.state.tick / TICKS_PER_DAY
    fig.suptitle(f"Shrooms seed {sim.seed} - tick {sim.state.tick} (day {day:.0f})")
    return fig


def _daily(x: np.ndarray, how=np.mean) -> np.ndarray:
    full = len(x) // TICKS_PER_DAY * TICKS_PER_DAY
    return how(x[:full].reshape(-1, TICKS_PER_DAY), axis=1)


def timeseries_figure(sim: Simulation) -> plt.Figure:
    h = {k: np.asarray(v) for k, v in sim.history.items()}
    n = sim.state.elevation.size
    d = np.arange(len(h["temp_c"]) // TICKS_PER_DAY)
    per_cell = lambda k: _daily(h[k]) / n  # noqa: E731

    fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(d, _daily(h["temp_c"]), color="tab:red")
    ax.set_ylabel("Temp (C)")
    twin = ax.twinx()
    twin.bar(d, _daily(h["rain_mm"], np.sum), color="tab:blue", alpha=0.4, width=1.0)
    twin.set_ylabel("Rain (mm/day)")

    for k, color in (("plant_c", "tab:green"), ("litter_c", "goldenrod")):
        axes[1].plot(d, per_cell(k), label=k.removesuffix("_c"), color=color)
    axes[1].set_ylabel("C (g/cell)")
    axes[1].legend(loc="upper left")

    for k, color in (("bacteria_c", "tab:orange"), ("saprotrophs_c", "saddlebrown"),
                     ("mycorrhiza_c", "tab:blue"), ("insects_c", "tab:purple")):  # fmt: skip
        axes[2].plot(d, per_cell(k), label=k.removesuffix("_c"), color=color)
    axes[2].set_ylabel("Guild C (g/cell)")
    axes[2].legend(loc="upper left")

    axes[3].plot(d, per_cell("som_c"), color="rosybrown")
    axes[3].set_ylabel("SOM C (g/cell)")

    axes[4].plot(d, _daily(h["trade_c"], np.sum) / n, color="tab:blue", label="C paid to fungi")
    axes[4].set_ylabel("Market (g C/cell/day)")
    twin = axes[4].twinx()
    twin.plot(d, _daily(h["trade_n"], np.sum) / n * 1000, color="tab:green", label="N delivered")
    twin.set_ylabel("N delivered (mg/cell/day)")
    axes[4].legend(loc="upper left")

    axes[5].plot(d, _daily(h["contaminant"]), color="tab:red")
    axes[5].set_ylabel("Contaminant (g total)")
    axes[5].set_xlabel("Day")
    fig.suptitle(f"Shrooms seed {sim.seed} - pools over time")
    return fig


def save_views(sim: Simulation, out_dir: str | Path, stem: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, make in (("grid", grid_figure), ("timeseries", timeseries_figure)):
        path = out / f"{stem}_{suffix}.png"
        fig = make(sim)
        fig.savefig(path, dpi=80)
        plt.close(fig)
        paths.append(path)
    return paths
