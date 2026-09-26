"""Weather generator: one uniform weather reading for the whole plot per tick.

Temperature = seasonal cycle + diurnal cycle + AR(1) anomaly.
Shortwave follows day length and season, and is cut by cloud when raining.
Rain is a two-state Markov chain with exponentially distributed hourly amounts.
"""

import math
from dataclasses import dataclass

import numpy as np

from sim import DAYS_PER_YEAR, TICKS_PER_DAY
from sim.params import WeatherParams


@dataclass(frozen=True)
class Weather:
    day_of_year: int
    hour: int
    temp_c: float
    shortwave: float  # W/m^2
    rain_mm: float


def step_weather(
    tick: int,
    temp_anomaly_c: float,
    raining: bool,
    p: WeatherParams,
    rng: np.random.Generator,
) -> tuple[Weather, float, bool]:
    """Return (weather for this tick, new temp anomaly, new raining flag)."""
    day = (tick // TICKS_PER_DAY) % DAYS_PER_YEAR
    hour = tick % TICKS_PER_DAY
    year_frac = day / DAYS_PER_YEAR

    # Draw in a fixed order so the stream never shifts between code paths.
    noise, rain_switch, rain_amount = rng.standard_normal(), rng.random(), rng.exponential()

    anomaly = p.temp_noise_ar * temp_anomaly_c + p.temp_noise_sd_c * noise
    seasonal = p.seasonal_amp_c * math.sin(2 * math.pi * (year_frac - 110 / DAYS_PER_YEAR))
    diurnal = p.diurnal_amp_c * math.sin(2 * math.pi * (hour - 9) / TICKS_PER_DAY)
    temp = p.mean_temp_c + seasonal + diurnal + anomaly

    raining = (rain_switch >= p.p_rain_stop) if raining else (rain_switch < p.p_rain_start)
    rain_mm = p.mean_rain_mm * rain_amount if raining else 0.0

    season = math.sin(2 * math.pi * (year_frac - 80 / DAYS_PER_YEAR))  # +1 at June solstice
    day_length = 12.0 + 3.0 * season
    sunrise = 12.0 - day_length / 2.0
    since_sunrise = hour + 0.5 - sunrise
    if 0.0 < since_sunrise < day_length:
        peak = p.peak_shortwave * (0.65 + 0.35 * season)
        shortwave = peak * math.sin(math.pi * since_sunrise / day_length)
    else:
        shortwave = 0.0
    if raining:
        shortwave *= p.rain_cloud_factor

    return Weather(day, hour, temp, shortwave, rain_mm), anomaly, raining
