"""Mass-balance ledger.

Every carbon and water flow is booked here by name. The budget check is:

    carbon: total now == total at t0            (closed system: atmosphere included)
    water:  total now == total at t0 + in - out  (open system: rain in; ET, drainage out)

Any mismatch beyond float rounding raises MassBalanceError.
"""

from collections import defaultdict
from dataclasses import dataclass, field

CARBON_FLOWS = ("gpp", "autotrophic_resp", "litterfall", "disturbance", "heterotrophic_resp")
WATER_IN = ("rain",)
WATER_OUT = ("evapotranspiration", "drainage")

RTOL = 1e-10


class MassBalanceError(AssertionError):
    pass


@dataclass
class Ledger:
    carbon_t0: float
    water_t0: float
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def book(self, flow: str, amount: float) -> None:
        if amount < 0:
            raise MassBalanceError(f"negative flow booked: {flow}={amount}")
        self.totals[flow] += amount

    def expected_water(self) -> float:
        inflow = sum(self.totals[k] for k in WATER_IN)
        outflow = sum(self.totals[k] for k in WATER_OUT)
        return self.water_t0 + inflow - outflow

    def check(self, carbon_now: float, water_now: float, tick: int) -> None:
        c_err = carbon_now - self.carbon_t0
        if abs(c_err) > RTOL * self.carbon_t0:
            raise MassBalanceError(f"tick {tick}: unexplained carbon {c_err:+.6g} g")
        w_expected = self.expected_water()
        w_err = water_now - w_expected
        if abs(w_err) > RTOL * max(self.water_t0, w_expected, 1.0):
            raise MassBalanceError(f"tick {tick}: unexplained water {w_err:+.6g} mm")
