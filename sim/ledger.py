"""Mass-balance ledger for every conserved substance.

Each substance's land budget must close every tick:

    total now == total at t0 + booked inflows - booked outflows

    c           in: gpp, fixation-free; out: every respiration flux (to atmosphere)
    n           in: deposition, fixation;  out: leaching, denitrification
    p           in: weathering;            out: leaching
    water       in: rain;                  out: evapotranspiration, drainage
    contaminant in: spills;                out: degradation, leaching

Carbon is also checked as a closed system with the atmosphere included, so a
flux booked to the land but not debited from the atmosphere is caught too.
Internal flows (litterfall, herbivory, trade...) are booked for reporting only.
"""

from collections import defaultdict
from dataclasses import dataclass, field

SUBSTANCES = ("c", "n", "p", "water", "contaminant")
RTOL = 1e-10


class MassBalanceError(AssertionError):
    pass


def _key(substance: str, flow: str) -> str:
    return f"{substance}:{flow}"


@dataclass
class Ledger:
    t0: dict[str, float]  # land total per substance at t0
    atmosphere_c0: float
    inflows: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    outflows: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    internal: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def _checked(self, substance: str, flow: str, amount: float) -> str:
        if substance not in SUBSTANCES:
            raise KeyError(substance)
        if not amount >= 0:  # also rejects NaN
            raise MassBalanceError(f"bad flow booked: {substance}:{flow}={amount}")
        return _key(substance, flow)

    def inflow(self, substance: str, flow: str, amount: float) -> None:
        self.inflows[self._checked(substance, flow, amount)] += amount

    def outflow(self, substance: str, flow: str, amount: float) -> None:
        self.outflows[self._checked(substance, flow, amount)] += amount

    def move(self, substance: str, flow: str, amount: float) -> None:
        self.internal[self._checked(substance, flow, amount)] += amount

    def flows(self, substance: str) -> dict[str, float]:
        """All booked flows for one substance, inflows positive and outflows negative."""
        out = {}
        for k, v in self.inflows.items():
            if k.startswith(substance + ":"):
                out[k.split(":", 1)[1]] = v
        for k, v in self.outflows.items():
            if k.startswith(substance + ":"):
                out[k.split(":", 1)[1]] = -v
        return out

    def expected(self, substance: str) -> float:
        return self.t0[substance] + sum(self.flows(substance).values())

    def check(self, totals: dict[str, float], atmosphere_c: float, tick: int) -> None:
        for s in SUBSTANCES:
            want, got = self.expected(s), totals[s]
            if abs(got - want) > RTOL * max(abs(self.t0[s]), abs(want), 1.0):
                raise MassBalanceError(f"tick {tick}: unexplained {s} {got - want:+.6g}")
        closed = self.atmosphere_c0 + self.t0["c"]
        if abs(atmosphere_c + totals["c"] - closed) > RTOL * closed:
            raise MassBalanceError(
                f"tick {tick}: unexplained c (closed) {atmosphere_c + totals['c'] - closed:+.6g}"
            )
