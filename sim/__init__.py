"""Shrooms deterministic simulation core.

Phase 0: grid, tick loop, seeded RNG streams, weather, soil water, and plants
with explicit carbon accounting. See ROADMAP.md.
"""

SIM_VERSION = "0.2.0"
TICKS_PER_DAY = 24
DAYS_PER_YEAR = 365
TICKS_PER_YEAR = TICKS_PER_DAY * DAYS_PER_YEAR  # 8760; 1 tick = 1 sim-hour

__all__ = ["SIM_VERSION", "TICKS_PER_DAY", "DAYS_PER_YEAR", "TICKS_PER_YEAR"]
