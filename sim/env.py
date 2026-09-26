"""Per-tick context shared by every guild: environment responses and C booking."""

from dataclasses import dataclass

import numpy as np

from sim import TICKS_PER_DAY
from sim.ledger import Ledger
from sim.weather import Weather

DT = 1.0 / TICKS_PER_DAY  # days per tick


def q10(temp_c: float, q: float) -> float:
    return q ** ((temp_c - 20.0) / 10.0)


def moisture_response(moisture: np.ndarray, optimum: float, width: float = 0.3) -> np.ndarray:
    """Bell-shaped response to relative moisture, 1 at `optimum`."""
    return np.exp(-(((moisture - optimum) / width) ** 2))


@dataclass
class TickContext:
    weather: Weather
    moisture: np.ndarray  # root-zone relative moisture 0..1
    toxicity: np.ndarray  # 1 = clean, -> 0 as contaminant rises
    ledger: Ledger
    atmosphere_delta: float = 0.0

    def respire(self, flow: str, amount: np.ndarray) -> None:
        """Carbon leaving the land to the atmosphere."""
        total = float(amount.sum())
        self.ledger.outflow("c", flow, total)
        self.atmosphere_delta += total

    def fix_carbon(self, flow: str, amount: np.ndarray) -> None:
        """Carbon entering the land from the atmosphere."""
        total = float(amount.sum())
        self.ledger.inflow("c", flow, total)
        self.atmosphere_delta -= total

    def move(self, substance: str, flow: str, amount: np.ndarray) -> None:
        self.ledger.move(substance, flow, float(amount.sum()))
