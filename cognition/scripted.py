"""Deterministic stand-ins for the LLM, used for tests, CI and as the last fallback.

They read the same observation the model gets (passed as `context`) and return
text in the same format a model would, so the whole agent graph, including
parsing and validation, runs identically. Deliberately simple heuristics:
their job is to be predictable, not clever.
"""

import json
import re

from sim.patches import PATCH_ID


def _hops(a: str, b: str) -> int:
    (ra, ca), (rb, cb) = (tuple(int(x) for x in PATCH_ID.match(p).groups()) for p in (a, b))
    return abs(ra - rb) + abs(ca - cb)


PATCH_IN_TEXT = re.compile(r"\br\d+c\d+\b")


def _heed_whisper(obs: dict) -> tuple[dict, str]:
    """Move any needy patch the gardener named to the front of the queue."""
    text = obs.get("whisper")
    if not text:
        return obs, ""
    named = [m.group(0) for m in PATCH_IN_TEXT.finditer(text.lower())]
    needy = obs["needy_partners"]
    hit = [n for n in needy if n["patch"] in named]
    if hit:
        rest = [n for n in needy if n["patch"] not in named]
        obs = {**obs, "needy_partners": hit + rest}
        return obs, (f'The gardener whispered "{text}"; {hit[0]["patch"]} is indeed '
                     "hungry, so I start there. ")  # fmt: skip
    if named:
        return obs, (f'The gardener whispered "{text}", but {named[0]} is not short of '
                     "nutrients on my network, so I won't act there. ")  # fmt: skip
    return obs, f'The gardener whispered "{text}"; I note it. '


def mycelium(role: str, messages: list[dict], context: dict) -> str:
    obs, heard = _heed_whisper(context["observation"])
    if context["step"] == "deliberate":
        return heard + _deliberate(obs)
    return _propose(obs, context)


def _deliberate(obs: dict) -> str:
    if not obs["market"]["active"]:
        return (
            "The plants are not trading; nothing pays now. I will hold my stores and "
            "only pull hyphae out of contaminated ground."
        )
    if not obs["needy_partners"]:
        return "No partner is short of nutrients; I will hold my stores."
    top = obs["needy_partners"][0]
    return (
        f"Partners in {top['patch']} are the hungriest (C:N {top['plant_cn']}). "
        "I will route stored nutrients to them from the nearest rich patch and "
        "lower my price there, and pull hyphae away from contaminated ground."
    )


def _propose(obs: dict, context: dict) -> str:
    blocked = set(context.get("blocked", [])) | set(obs["cooling_down"])
    limits = obs["limits"]
    intents: list[dict] = []
    trading = obs["market"]["active"]  # no point investing while no plant can pay

    def add(intent: dict) -> None:
        target = intent.get("to_patch") or intent.get("patch")
        if len(intents) < limits["max_intents"] and f"{intent['kind']}->{target}" not in blocked:
            intents.append(intent)

    for need in obs["needy_partners"][:2] if trading else []:
        for element, demand_key, store_key in (
            ("n", "n_demand", "sellable_n_g"),
            ("p", "p_demand", "sellable_p_g"),
        ):
            if need.get(demand_key, need.get("n_demand", 0)) < 0.05:
                continue
            donors = [
                d
                for d in obs["nutrient_stores"]
                if d["patch"] != need["patch"]
                and d[store_key] > 0.05
                and _hops(d["patch"], need["patch"]) <= limits["max_hops"]
            ]
            if donors:
                d = max(donors, key=lambda d: d[store_key] / _hops(d["patch"], need["patch"]))
                distance = _hops(d["patch"], need["patch"])
                affordable = d[f"affordable_{element}_g_per_hop"] / distance
                amount = round(min(0.4 * d[store_key], 0.9 * affordable), 3)
                add(
                    {
                        "kind": "shuttle_nutrients",
                        "from_patch": d["patch"],
                        "to_patch": need["patch"],
                        "element": element,
                        "amount_g": amount,
                        "rationale": f"{need['patch']} partners need {element.upper()}; "
                        f"{d['patch']} has spare",
                    }
                )
                break
    if trading and obs["needy_partners"]:
        top = obs["needy_partners"][0]
        if top["trade_bias"] > limits["trade_bias_range"][0] + 1e-9:
            new = max(top["trade_bias"] - 0.1, limits["trade_bias_range"][0])
            add(
                {
                    "kind": "set_trade_bias",
                    "patch": top["patch"],
                    "bias": round(new, 2),
                    "rationale": "sell cheaply to a struggling partner",
                }
            )
    needy_ids = {n["patch"] for n in obs["needy_partners"]}
    for b in obs["biased_patches"]:
        if b["patch"] not in needy_ids and b["trade_bias"] < 1.0:
            add(
                {
                    "kind": "set_trade_bias",
                    "patch": b["patch"],
                    "bias": round(min(b["trade_bias"] + 0.1, 1.0), 2),
                    "rationale": "partner has recovered; return to a fair price",
                }
            )
    for dirty in obs["contaminated"]:
        if dirty["fungal_c"] > 2.0 and dirty["contaminant"] > 5.0:
            add(
                {
                    "kind": "relocate_hyphae",
                    "from_patch": dirty["patch"],
                    "to_patch": dirty["cleanest_neighbour"],
                    "fraction": 0.1,
                    "rationale": "withdraw hyphae from contaminated soil",
                }
            )
    return json.dumps({"intents": intents})


def narrator(role: str, messages: list[dict], context: dict) -> str:
    f = context["facts"]
    now = f["per_m2_now"]
    lines = [
        f"Temperatures ran from {f['temp_c']['min']} to {f['temp_c']['max']} C "
        f"with {f['rain_mm']} mm of rain."
    ]
    events = [e.split(": ", 1)[1] for e in f["events"]]
    if events:
        lines.append("Noted: " + "; ".join(events[:5]) + ".")
    actions = f.get("network_actions") or {}
    if actions:
        done = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in actions.items())
        lines.append(f"The mycelial network acted: {done}.")
    lines.append(
        f"Plants now hold about {now['plant_c_g']} g C per square metre, the fungal "
        f"network {now['fungal_c_g']} g, and plants paid the fungi "
        f"{f['fungal_trade_c_g_per_m2']} g C per square metre in trade."
    )
    return " ".join(lines)


POLICIES = {"mycelium": mycelium, "narrator": narrator}
