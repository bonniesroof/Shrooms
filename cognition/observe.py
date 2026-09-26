"""Observations: compact, JSON-safe views of the sim for agents.

A 7B local model can't read 64 patches x 10 fields well, so the mycelium sees
ranked short lists: its neediest plant partners, its richest nutrient stores,
contaminated ground, active cooldowns, recent events, and how its last
decisions turned out. All numbers are plain floats rounded for readability.
"""

from collections import Counter

import numpy as np

from sim import TICKS_PER_DAY
from sim.engine import Simulation
from sim.events import date_label
from sim.intents import sellable
from sim.patches import all_ids, patch_means, patch_sums
from sim.validator import _target


def _r(x: float, digits: int = 2) -> float | int:
    return int(round(float(x))) if digits == 0 else round(float(x), digits)


def patch_table(sim: Simulation) -> dict[str, dict]:
    s, p = sim.state, sim.params
    size = p.network.patch_size
    pp = p.plants
    plant_c, plant_n, plant_p = (patch_sums(getattr(s.plant, e), size) for e in "cnp")
    safe_c = np.maximum(plant_c, 1e-9)
    demand_n = np.clip(1 - plant_n * pp.target_cn / safe_c, 0, 1)
    demand_p = np.clip(1 - plant_p * pp.target_cp / safe_c, 0, 1)
    fields = {
        "plant_c": patch_means(s.plant.c, size),
        "plant_cn": plant_c / np.maximum(plant_n, 1e-9),
        "n_demand": demand_n,
        "p_demand": demand_p,
        "fungal_c": patch_means(s.mycorrhiza.c, size),
        "sellable_n_g": patch_sums(sellable(s, "n", p), size),
        "sellable_p_g": patch_sums(sellable(s, "p", p), size),
        "moisture": patch_means(s.moisture(), size),
        "contaminant": patch_means(s.contaminant, size),
        "trade_bias": patch_means(s.trade_bias, size),
    }
    table = {}
    for pid in all_ids(s.shape, size):
        r, c = (int(v) for v in pid[1:].split("c"))
        row = {k: _r(v[r, c], 3 if k.startswith("sellable_p") else 2) for k, v in fields.items()}
        row["on_network"] = bool(fields["fungal_c"][r, c] >= p.network.min_network_c)
        table[pid] = row
    return table


def neighbours(pid: str, rows: int, cols: int) -> list[str]:
    r, c = (int(v) for v in pid[1:].split("c"))
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if 0 <= r + dr < rows and 0 <= c + dc < cols:
            out.append(f"r{r + dr}c{c + dc}")
    return out


def observe_mycelium(sim: Simulation, outcomes: list[dict], latency_ticks: int) -> dict:
    s, p, h = sim.state, sim.params, sim.history
    net = p.network
    table = patch_table(sim)
    rows = cols = s.shape[0] // net.patch_size
    on = {k: v for k, v in table.items() if v["on_network"]}

    def with_id(pid: str, keys: tuple[str, ...]) -> dict:
        return {"patch": pid, **{k: table[pid][k] for k in keys}}

    need = sorted(on, key=lambda k: -(on[k]["n_demand"] + on[k]["p_demand"]) * on[k]["plant_c"])
    needy = [
        with_id(
            k,
            (
                "plant_c",
                "plant_cn",
                "n_demand",
                "p_demand",
                "fungal_c",
                "sellable_n_g",
                "trade_bias",
            ),
        )
        for k in need[:6]
    ]
    # What each donor can afford to ship one hop: the validator caps a shipment's
    # carbon cost at max_cost_fraction of the source patch's fungal carbon.
    cells = net.patch_size**2
    budget = {k: net.max_cost_fraction * on[k]["fungal_c"] * cells for k in on}
    donors = [
        {
            **with_id(k, ("sellable_n_g", "sellable_p_g", "fungal_c", "n_demand")),
            "affordable_n_g_per_hop": _r(budget[k] / net.shuttle_cost_c_per_g_n, 2),
            "affordable_p_g_per_hop": _r(budget[k] / net.shuttle_cost_c_per_g_p, 3),
        }
        for k in sorted(on, key=lambda k: -on[k]["sellable_n_g"])[:5]
    ]
    dirty = []
    for k in sorted(table, key=lambda k: -table[k]["contaminant"])[:3]:
        if table[k]["contaminant"] < 1.0:
            break
        nb = min(neighbours(k, rows, cols), key=lambda n: table[n]["contaminant"])
        dirty.append(
            {
                **with_id(k, ("contaminant", "fungal_c", "plant_c")),
                "cleanest_neighbour": nb,
                "neighbour_contaminant": table[nb]["contaminant"],
            }
        )
    biased = [
        with_id(k, ("trade_bias", "n_demand", "p_demand"))
        for k in table
        if abs(table[k]["trade_bias"] - 1.0) > 1e-6
    ]
    act_tick = s.tick + latency_ticks
    cooling = sorted(
        {
            f"{i.kind}->{_target(i)}"
            for i in sim.accepted
            if i.agent == "mycelium" and _target(i) and 0 <= act_tick - i.tick < net.cooldown_ticks
        }
    )
    week = slice(-7 * TICKS_PER_DAY, None)
    trade_week = float(np.sum(h["trade_c"][week])) / s.plant.c.size if h["trade_c"] else 0.0
    return {
        "tick": s.tick,
        "date": date_label(s.tick),
        "weather_last_7_days": {
            "mean_temp_c": _r(np.mean(h["temp_c"][week])) if h["temp_c"] else None,
            "rain_mm": _r(np.sum(h["rain_mm"][week]), 1) if h["rain_mm"] else 0.0,
        },
        "market": {
            "active": trade_week > 0.1,
            "c_paid_last_7_days_g_per_m2": _r(trade_week, 3),
        },
        "network": {
            "patches_on_network": len(on),
            "patches_total": len(table),
            "mean_fungal_c": _r(s.mycorrhiza.c.mean()),
        },
        "needy_partners": needy,
        "nutrient_stores": donors,
        "contaminated": dirty,
        "biased_patches": biased,
        "cooling_down": cooling,
        "recent_events": [f"{e.date}: {e.message}" for e in sim.events.events[-6:]],
        "last_outcomes": outcomes,
        "limits": {
            "max_intents": net.max_intents_per_tick,
            "max_hops": net.max_hops,
            "relocate_max_fraction": net.max_relocate_fraction,
            "trade_bias_range": [net.trade_bias_min, net.trade_bias_max],
            "trade_bias_max_step": net.trade_bias_max_step,
            "shuttle_max": "90% of source sellable store, and at most "
            "affordable_<element>_g_per_hop / hops",
        },
    }


def narrator_facts(sim: Simulation, since_tick: int) -> dict:
    s, h = sim.state, sim.history
    period = sim.events.since(since_tick)
    n = s.plant.c.size
    span = slice(since_tick, s.tick)
    temps = h["temp_c"][span] or [0.0]
    return {
        "period": f"{date_label(since_tick)} to {date_label(max(s.tick - 1, since_tick))}",
        "temp_c": {
            "min": _r(min(temps), 1),
            "max": _r(max(temps), 1),
            "mean": _r(np.mean(temps), 1),
        },
        "rain_mm": _r(sum(h["rain_mm"][span]), 0),
        "per_m2_now": {
            "plant_c_g": _r(s.plant.c.mean(), 0),
            "insects_c_g": _r(s.insects.c.mean(), 2),
            "fungal_c_g": _r(s.mycorrhiza.c.mean(), 1),
            "bacteria_c_g": _r(s.bacteria.c.mean(), 1),
            "litter_c_g": _r(s.litter.c.mean(), 0),
        },
        "fungal_trade_c_g_per_m2": _r(sum(h["trade_c"][span]) / n, 1),
        "contaminant_kg": _r(s.contaminant.sum() / 1000, 2),
        "events": [f"{e.date}: {e.message}" for e in period if not e.kind.startswith("intent_")][
            :10
        ],
        "network_actions": dict(
            Counter(e.data.get("intent") for e in period if e.kind == "intent_applied")
        ),
    }
