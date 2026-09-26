"""Simulation parameters.

Every tunable number lives here so a run is fully described by
(seed, sim version, params, intents). Units: carbon in g C per cell,
water in mm (= litres on a 1 m^2 cell), temperature in deg C,
shortwave radiation in W/m^2, rates per tick (1 tick = 1 hour).
"""

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WorldParams(_Frozen):
    width: int = Field(64, ge=4)
    height: int = Field(64, ge=4)
    terrain_smoothing_passes: int = Field(6, ge=0)


class WeatherParams(_Frozen):
    mean_temp_c: float = 11.0
    seasonal_amp_c: float = 12.0
    diurnal_amp_c: float = 5.0
    temp_noise_sd_c: float = 0.4
    temp_noise_ar: float = Field(0.97, ge=0.0, lt=1.0)
    peak_shortwave: float = 900.0
    p_rain_start: float = Field(0.02, ge=0.0, le=1.0)
    p_rain_stop: float = Field(0.2, ge=0.0, le=1.0)
    mean_rain_mm: float = Field(1.1, gt=0.0)
    rain_cloud_factor: float = Field(0.35, ge=0.0, le=1.0)


class WaterParams(_Frozen):
    capacity_mm: float = Field(150.0, gt=0.0)
    valley_capacity_bonus: float = Field(0.6, ge=0.0)  # low cells hold up to +60%
    initial_fraction: float = Field(0.6, ge=0.0, le=1.0)
    percolation_rate: float = Field(0.0004, ge=0.0, le=1.0)
    pet_per_wm2: float = Field(0.0006, ge=0.0)  # mm/h per W/m^2 at 20 C
    bare_soil_evap_fraction: float = Field(0.3, ge=0.0, le=1.0)


class PlantParams(_Frozen):
    initial_mean_c: float = Field(300.0, ge=0.0)
    initial_bare_fraction: float = Field(0.15, ge=0.0, le=1.0)
    light_use_efficiency: float = Field(0.0022, ge=0.0)  # g C per (W/m^2 * h)
    cover_k: float = Field(0.006, gt=0.0)  # canopy cover = 1 - exp(-k * C)
    max_c: float = Field(2500.0, gt=0.0)  # crowding limit per cell
    temp_opt_c: float = 22.0
    temp_width_c: float = Field(11.0, gt=0.0)
    water_stress_fraction: float = Field(0.5, gt=0.0, le=1.0)
    respiration_rate_20c: float = Field(0.00012, ge=0.0)
    q10: float = Field(2.0, gt=0.0)
    turnover_rate: float = Field(0.00006, ge=0.0, le=1.0)
    dispersal_rate: float = Field(0.0002, ge=0.0, le=0.25)
    disturbance_prob: float = Field(2e-6, ge=0.0, le=1.0)  # per cell per tick
    disturbance_severity: float = Field(0.9, ge=0.0, le=1.0)


class SoilParams(_Frozen):
    initial_c: float = Field(1500.0, ge=0.0)
    decomposition_rate_20c: float = Field(0.000035, ge=0.0, le=1.0)
    q10: float = Field(2.2, gt=0.0)


class SimParams(_Frozen):
    world: WorldParams = WorldParams()
    weather: WeatherParams = WeatherParams()
    water: WaterParams = WaterParams()
    plants: PlantParams = PlantParams()
    soil: SoilParams = SoilParams()
    atmosphere_initial_c: float = Field(1.0e9, gt=0.0)
