"""Event log: notable things that happened, derived deterministically from the run.

Events are *derived* data. They aren't part of the replay record, because a
replay regenerates exactly the same events. The Narrator reads them to write
the field journal, and the keystone agent sees the recent ones in its
observation.

Detection runs once per sim-day on daily aggregates; intents are logged by
the engine as they are applied or rejected.
"""

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

from sim import DAYS_PER_YEAR, TICKS_PER_DAY
from sim.world import POOLS

log = logging.getLogger("shrooms.events")

GUILDS = ("plant", "bacteria", "saprotrophs", "mycorrhiza", "insects")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def date_label(tick: int) -> str:
    day = tick // TICKS_PER_DAY
    year, doy = divmod(day, DAYS_PER_YEAR)
    month = min(int(doy / (DAYS_PER_YEAR / 12)), 11)
    return f"Y{year + 1} {MONTHS[month]} (day {doy + 1})"


@dataclass(frozen=True)
class Event:
    tick: int
    kind: str
    message: str
    data: dict = field(default_factory=dict)

    @property
    def date(self) -> str:
        return date_label(self.tick)

    def as_dict(self) -> dict:
        return {**asdict(self), "date": self.date}


class EventLog:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, tick: int, kind: str, message: str, **data) -> Event:
        clean = {k: (round(float(v), 4) if isinstance(v, float | np.floating) else v)
                 for k, v in data.items()}  # fmt: skip
        event = Event(tick, kind, message, clean)
        self.events.append(event)
        log.info("[%s] %s: %s", event.date, kind, message)
        return event

    def since(self, tick: int) -> list[Event]:
        return [e for e in self.events if e.tick >= tick]


class EventDetector:
    """Daily checks with hysteresis, so each episode is reported once."""

    def __init__(self, initial_contaminant: float, n_cells: int):
        self.n = n_cells
        self.contam0 = initial_contaminant
        self.milestones = [0.9, 0.75, 0.5, 0.25]
        self.frost_years: set[int] = set()
        self.in_drought = False
        self.heat_days = 0
        self.in_heatwave = False
        self.market_open = False
        self.quiet_days = 0
        self.weekly: dict[str, float] = {}

    def daily(self, sim, log_: EventLog) -> None:
        s, h = sim.state, sim.history
        tick = s.tick
        if tick % TICKS_PER_DAY or len(h["temp_c"]) < TICKS_PER_DAY:
            return
        day = tick // TICKS_PER_DAY
        year, doy = divmod(day - 1, DAYS_PER_YEAR)
        last = slice(-TICKS_PER_DAY, None)
        temps = h["temp_c"][last]
        tmin, tmax = min(temps), max(temps)
        rain = sum(h["rain_mm"][last])
        trade = sum(h["trade_c"][last]) / self.n
        moisture = float(s.moisture().mean())

        if rain > 20:
            log_.emit(tick, "storm", f"{rain:.0f} mm of rain in a day", rain_mm=rain)
        if doy > 200 and tmin < 0 and year not in self.frost_years:
            self.frost_years.add(year)
            log_.emit(tick, "first_frost", f"first autumn frost, low of {tmin:.1f} C",
                      tmin_c=tmin)  # fmt: skip

        self.heat_days = self.heat_days + 1 if tmax > 28 else 0
        if self.heat_days >= 3 and not self.in_heatwave:
            self.in_heatwave = True
            log_.emit(tick, "heatwave", f"three days above 28 C (high {tmax:.1f} C)", tmax_c=tmax)
        elif self.heat_days == 0:
            self.in_heatwave = False

        if not self.in_drought and moisture < 0.40:
            self.in_drought = True
            log_.emit(tick, "drought_start", f"root-zone moisture down to {moisture:.2f}",
                      moisture=moisture)  # fmt: skip
        elif self.in_drought and moisture > 0.50:
            self.in_drought = False
            log_.emit(tick, "drought_end", f"root-zone moisture recovered to {moisture:.2f}",
                      moisture=moisture)  # fmt: skip

        if not self.market_open and trade > 0.05:
            self.market_open, self.quiet_days = True, 0
            log_.emit(tick, "market_open", f"fungal-plant trade resumes ({trade:.2f} g C/cell/day)",
                      trade_c=trade)  # fmt: skip
        elif self.market_open:
            self.quiet_days = self.quiet_days + 1 if trade < 0.02 else 0
            if self.quiet_days >= 7:
                self.market_open = False
                log_.emit(tick, "market_close", "fungal-plant trade has gone quiet for a week")

        total = float(s.contaminant.sum())
        while self.milestones and self.contam0 > 0 and total < self.milestones[0] * self.contam0:
            m = self.milestones.pop(0)
            log_.emit(tick, "cleanup_milestone",
                      f"contaminant below {m:.0%} of its starting mass ({total:.0f} g left)",
                      fraction=m, remaining_g=total)  # fmt: skip

        if day % 7 == 0:
            now = {g: s.pool(g).c.mean() for g in GUILDS}
            for g, v in now.items():
                prev = self.weekly.get(g)
                if prev and prev > 1e-6:
                    change = v / prev - 1
                    if change > 0.4:
                        log_.emit(tick, "boom", f"{g} up {change:.0%} in a week",
                                  guild=g, change=change, mean_c=v)  # fmt: skip
                    elif change < -0.3:
                        log_.emit(tick, "crash", f"{g} down {-change:.0%} in a week",
                                  guild=g, change=change, mean_c=v)  # fmt: skip
            self.weekly = now


__all__ = ["Event", "EventLog", "EventDetector", "date_label", "POOLS"]
