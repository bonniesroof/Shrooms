"""Simulation parameters.

Every tunable number lives here so a run is fully described by
(seed, sim version, params, intents).

Units: mass in g per cell (1 cell = 1 m^2) for C, N and P; contaminant in g;
water in mm (= litres per cell); temperature in deg C; shortwave in W/m^2.
Biological rates are per day (`*_per_day`) and converted to per tick in code.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WorldParams(_Frozen):
    width: int = Field(64, ge=4)
    height: int = Field(64, ge=4)
    terrain_smoothing_passes: int = Field(6, ge=0)
    relief: float = Field(4.0, ge=0.0)  # metres between lowest and highest cell


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


class HydrologyParams(_Frozen):
    """Three soil layers (top to bottom), lateral flow downslope, deep drainage."""

    layer_capacity_mm: tuple[float, float, float] = (40.0, 80.0, 120.0)
    root_fraction: tuple[float, float, float] = (0.6, 0.3, 0.1)
    initial_fraction: float = Field(0.6, ge=0.0, le=1.0)
    percolation_per_day: float = Field(0.25, ge=0.0, le=24.0)  # of water above half capacity
    deep_drainage_per_day: float = Field(0.02, ge=0.0, le=24.0)
    lateral_per_day: float = Field(0.4, ge=0.0, le=24.0)  # subsurface flow per m of drop
    pet_per_wm2: float = Field(0.0006, ge=0.0)  # mm/h per W/m^2 at 20 C
    bare_soil_evap_fraction: float = Field(0.3, ge=0.0, le=1.0)


class PlantParams(_Frozen):
    initial_mean_c: float = Field(300.0, ge=0.0)
    initial_bare_fraction: float = Field(0.15, ge=0.0, le=1.0)
    light_use_efficiency: float = Field(0.0022, ge=0.0)  # g C per (W/m^2 * h)
    cover_k: float = Field(0.006, gt=0.0)
    max_c: float = Field(2500.0, gt=0.0)
    temp_opt_c: float = 22.0
    temp_width_c: float = Field(11.0, gt=0.0)
    water_stress_fraction: float = Field(0.5, gt=0.0, le=1.0)
    respiration_per_day: float = Field(0.0029, ge=0.0)  # at 20 C
    q10: float = Field(2.0, gt=0.0)
    turnover_per_day: float = Field(0.00144, ge=0.0, le=1.0)
    resorption: float = Field(0.5, ge=0.0, lt=1.0)  # N and P pulled back before litterfall
    target_cn: float = Field(30.0, gt=0.0)
    target_cp: float = Field(300.0, gt=0.0)
    root_uptake_n_per_day: float = Field(0.004, ge=0.0)  # g N per g plant C at saturation
    root_uptake_p_per_day: float = Field(0.0004, ge=0.0)
    root_half_sat_n: float = Field(1.0, gt=0.0)  # g mineral N per cell
    root_half_sat_p: float = Field(0.5, gt=0.0)
    dispersal_per_day: float = Field(0.0048, ge=0.0, le=1.0)
    disturbance_prob: float = Field(2e-6, ge=0.0, le=1.0)  # per cell per tick
    disturbance_severity: float = Field(0.9, ge=0.0, le=1.0)


# Initial pools sit near the seasonal equilibrium found by multi-year runs, so a
# 1-year run measures the dynamics rather than a spin-up transient.


class MicrobeParams(_Frozen):
    """One decomposer or symbiont guild with fixed biomass stoichiometry."""

    initial_c: float = Field(ge=0.0)
    cn: float = Field(gt=0.0)
    cp: float = Field(gt=0.0)
    uptake_per_day: float = Field(ge=0.0)  # max substrate C taken per g biomass C
    half_sat_c: float = Field(gt=0.0)
    cue: float = Field(gt=0.0, lt=1.0)  # carbon use efficiency
    moisture_opt: float = Field(gt=0.0, le=1.0)
    death_per_day: float = Field(ge=0.0)
    crowding_c: float = Field(gt=0.0)  # density at which death doubles


class DecomposerParams(_Frozen):
    bacteria: MicrobeParams = MicrobeParams(
        initial_c=10.0, cn=5.0, cp=40.0, uptake_per_day=0.5, half_sat_c=300.0,
        cue=0.35, moisture_opt=0.7, death_per_day=0.01, crowding_c=150.0,
    )  # fmt: skip
    saprotrophs: MicrobeParams = MicrobeParams(
        initial_c=15.0, cn=10.0, cp=80.0, uptake_per_day=0.3, half_sat_c=300.0,
        cue=0.45, moisture_opt=0.45, death_per_day=0.006, crowding_c=150.0,
    )  # fmt: skip
    q10: float = Field(2.2, gt=0.0)
    litter_cn_sapro_pref: float = Field(40.0, gt=0.0)  # litter C:N where saprotrophs win half
    som_turnover_per_day: float = Field(0.0003, ge=0.0)  # at 20 C, full activity
    initial_litter_c: float = Field(100.0, ge=0.0)
    initial_som_c: float = Field(1500.0, ge=0.0)
    som_cn: float = Field(12.0, gt=0.0)
    som_cp: float = Field(100.0, gt=0.0)
    fixation_per_day: float = Field(0.0004, ge=0.0)  # g N per g bacterial C when N-starved
    fixation_c_cost: float = Field(8.0, ge=0.0)  # g C respired per g N fixed
    denitrification_per_day: float = Field(0.02, ge=0.0)  # of mineral N when saturated


class MycorrhizaParams(_Frozen):
    """Model A: a biological market. Plants pay carbon only for nutrients delivered."""

    model: Literal["A"] = "A"
    guild: MicrobeParams = MicrobeParams(
        initial_c=15.0, cn=12.0, cp=100.0, uptake_per_day=0.0, half_sat_c=1.0,
        cue=0.5, moisture_opt=0.55, death_per_day=0.006, crowding_c=120.0,
    )  # fmt: skip
    # Fungal enzymes mine N and P straight out of soil organic matter, which roots can't.
    mine_n_per_day: float = Field(0.004, ge=0.0)  # g N per g fungal C at saturation
    mine_p_per_day: float = Field(0.0008, ge=0.0)
    mine_half_sat_n: float = Field(50.0, gt=0.0)  # g SOM N per cell
    mine_half_sat_p: float = Field(8.0, gt=0.0)
    forage_n_per_day: float = Field(0.03, ge=0.0)  # g N per g fungal C at saturation
    forage_p_per_day: float = Field(0.006, ge=0.0)
    half_sat_n: float = Field(0.3, gt=0.0)  # fungi forage far better than roots
    half_sat_p: float = Field(0.05, gt=0.0)
    max_c_offer_fraction: float = Field(0.3, ge=0.0, le=1.0)  # of GPP, at full nutrient demand
    price_n: float = Field(12.0, gt=0.0)  # g C paid per g N delivered, at a balanced market
    price_p: float = Field(80.0, gt=0.0)
    price_elasticity: float = Field(1.0, ge=0.0)  # how hard scarcity of stock raises price
    reserve_fraction: float = Field(0.3, ge=0.0, lt=1.0)  # of the store fungi won't sell


class InsectParams(_Frozen):
    initial_c: float = Field(0.5, ge=0.0)
    cn: float = Field(5.0, gt=0.0)
    cp: float = Field(50.0, gt=0.0)
    feeding_per_day: float = Field(0.6, ge=0.0)  # plant C eaten per g insect C at saturation
    half_sat_plant_c: float = Field(300.0, gt=0.0)
    assimilation: float = Field(0.35, gt=0.0, lt=1.0)  # of eaten C; rest leaves as frass
    nutrient_assimilation: float = Field(0.8, gt=0.0, lt=1.0)  # of eaten N and P
    growth_efficiency: float = Field(0.4, gt=0.0, lt=1.0)  # of assimilated C
    min_feeding_temp_c: float = 6.0
    death_per_day: float = Field(0.008, ge=0.0)
    diapause_death_per_day: float = Field(0.003, ge=0.0)  # replaces death below feeding temp
    crowding_c: float = Field(0.4, gt=0.0)
    dispersal_per_day: float = Field(0.2, ge=0.0, le=1.0)


class ContaminationParams(_Frozen):
    """A generic persistent organic contaminant, tracked as its own conserved substance."""

    hotspots: int = Field(2, ge=0)
    hotspot_mass_g: float = Field(40.0, ge=0.0)  # peak per cell
    hotspot_radius: float = Field(4.0, gt=0.0)  # cells
    ec50_g: float = Field(10.0, gt=0.0)  # dose halving plant and microbial activity
    sorbed_fraction: float = Field(0.995, ge=0.0, le=1.0)  # immobile share
    degradation_per_day: float = Field(0.003, ge=0.0)  # per g bacterial C at saturation
    degradation_half_sat_g: float = Field(20.0, gt=0.0)
    sapro_degradation_share: float = Field(0.5, ge=0.0)  # fungi degrade at this x bacteria


class NutrientParams(_Frozen):
    initial_mineral_n: float = Field(4.0, ge=0.0)
    initial_mineral_p: float = Field(0.8, ge=0.0)
    n_deposition_per_day: float = Field(0.002, ge=0.0)  # g N per cell per day
    p_weathering_per_day: float = Field(0.0003, ge=0.0)
    mobile_n_fraction: float = Field(0.5, ge=0.0, le=1.0)  # dissolved share of mineral N
    mobile_p_fraction: float = Field(0.02, ge=0.0, le=1.0)  # P binds tightly to soil


class NetworkParams(_Frozen):
    """Limits on the mycelial-network keystone agent's intents (see sim/validator.py)."""

    patch_size: int = Field(8, ge=1)
    min_network_c: float = Field(2.0, ge=0.0)  # mean fungal C/cell for a patch to be "on" the net
    max_hops: int = Field(3, ge=1)
    shuttle_cost_c_per_g_n: float = Field(2.0, ge=0.0)  # respired per g N per hop
    shuttle_cost_c_per_g_p: float = Field(10.0, ge=0.0)
    max_cost_fraction: float = Field(0.2, gt=0.0, le=1.0)  # of source fungal C per intent
    max_relocate_fraction: float = Field(0.2, gt=0.0, le=1.0)
    relocate_cost_fraction: float = Field(0.05, ge=0.0, lt=1.0)  # of moved C, respired
    trade_bias_min: float = Field(0.5, gt=0.0)
    trade_bias_max: float = Field(2.0, gt=0.0)
    trade_bias_max_step: float = Field(0.25, gt=0.0)
    max_intents_per_tick: int = Field(4, ge=1)  # per agent
    cooldown_ticks: int = Field(336, ge=0)  # same kind, same target; > the agent's clock


class SimParams(_Frozen):
    world: WorldParams = WorldParams()
    weather: WeatherParams = WeatherParams()
    hydrology: HydrologyParams = HydrologyParams()
    plants: PlantParams = PlantParams()
    decomposers: DecomposerParams = DecomposerParams()
    mycorrhiza: MycorrhizaParams = MycorrhizaParams()
    insects: InsectParams = InsectParams()
    contamination: ContaminationParams = ContaminationParams()
    nutrients: NutrientParams = NutrientParams()
    network: NetworkParams = NetworkParams()
    atmosphere_initial_c: float = Field(1.0e9, gt=0.0)
