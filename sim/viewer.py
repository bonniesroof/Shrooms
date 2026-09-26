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

PANELS = (
    ("plant_c", "Plant C (g/cell)", "Greens"),
    ("soil_c", "Soil C (g/cell)", "copper"),
    ("soil_water", "Soil water (mm)", "Blues"),
    ("elevation", "Elevation (0-1)", "terrain"),
)


def grid_figure(sim: Simulation) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, (name, title, cmap) in zip(axes.flat, PANELS, strict=True):
        im = ax.imshow(getattr(sim.state, name), cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    day = sim.state.tick / TICKS_PER_DAY
    fig.suptitle(f"Shrooms seed {sim.seed} - tick {sim.state.tick} (day {day:.0f})")
    return fig


def timeseries_figure(sim: Simulation) -> plt.Figure:
    h = {k: np.asarray(v) for k, v in sim.history.items()}
    n_cells = sim.state.plant_c.size
    days = np.arange(len(h["plant_c"])) / TICKS_PER_DAY

    def daily(x: np.ndarray, how=np.mean) -> np.ndarray:
        full = len(x) // TICKS_PER_DAY * TICKS_PER_DAY
        return how(x[:full].reshape(-1, TICKS_PER_DAY), axis=1)

    d = np.arange(len(h["plant_c"]) // TICKS_PER_DAY)
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    axes[0].plot(d, daily(h["temp_c"]), color="tab:red")
    axes[0].set_ylabel("Temp (C, daily mean)")
    ax_rain = axes[0].twinx()
    ax_rain.bar(d, daily(h["rain_mm"], np.sum), color="tab:blue", alpha=0.4, width=1.0)
    ax_rain.set_ylabel("Rain (mm/day)")
    axes[1].plot(days, h["plant_c"] / n_cells, label="plant", color="tab:green")
    axes[1].plot(days, h["soil_c"] / n_cells, label="soil", color="saddlebrown")
    axes[1].set_ylabel("C (g/cell)")
    axes[1].legend(loc="upper left")
    axes[2].plot(days, h["soil_water"] / n_cells, color="tab:blue")
    axes[2].set_ylabel("Soil water (mm)")
    axes[3].plot(days, h["canopy_cover"], color="olive")
    axes[3].set_ylabel("Canopy cover")
    axes[3].set_xlabel("Day")
    fig.suptitle(f"Shrooms seed {sim.seed} - pools over time")
    return fig


def save_views(sim: Simulation, out_dir: str | Path, stem: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, make in (("grid", grid_figure), ("timeseries", timeseries_figure)):
        path = out / f"{stem}_{suffix}.png"
        fig = make(sim)
        fig.savefig(path, dpi=90)
        plt.close(fig)
        paths.append(path)
    return paths
