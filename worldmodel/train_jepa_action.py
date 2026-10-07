"""Train the action-conditioned JEPA and test it on spill-vs-no-spill counterfactuals.

    uv run python -m worldmodel.dataset --mode forcing --seeds 0-27   # ~11 min on 2 cores
    uv run python -m worldmodel.counterfactual                        # ~3 min on 2 cores
    uv run python -m worldmodel.train_jepa_action                     # ~5 min on 2 cores

The temporal JEPA (step 2) predicts the latents at t + k from the snapshot at t.
Here the predictor is also told the forcing over (t, t + k]: every intervention
applied in the window, per patch and per kind, and the window's mean weather
(forcing.py). It starts from the step-2 checkpoint; the new input paths start at
zero, so before training it is exactly the step-2 model. Data: the forcing-mode
dataset (data/phase5), an intervention every 10 days with probability 0.6,
with the applied intents recorded. Same world split as train.py.

Two evaluations, both on test worlds 24-27:

forecast   linear probes from frozen representations onto the Phase 4 heads
           (probes.fit_forecast_probes; skill vs persistence, as the GNN reports):
               action       this model's predicted latents, given the real forcing
               temporal     the step-2 model's predicted latents (no forcing)
               raw          the GNN's raw inputs
               raw+forcing  raw inputs plus the same forcing features, linearly
               constant     intercept only (the die-back floor)
               gnn          the shipped Phase 4 GNN, run on these test samples
counterfactual  for each pair (counterfactual.py), predict both branches, decode
           them with the same frozen probes, and compare the predicted effect
           (spill minus no spill) with the sim's true effect, per patch:
               effect skill     1 - MAE(predicted - true effect) / MAE(true effect);
                                0 for any model that can't tell the branches apart
               direction        share of clearly affected patches where the sign matches
               magnitude        predicted / true total effect over the map
               location         spatial correlation of predicted and true effect maps,
                                and whether the most affected patch is found

Writes worldmodel/jepa_action.pt and worldmodel/jepa_action_metrics.json.
Metrics come from the full-precision model; the checkpoint is stored in fp16.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from worldmodel.collapse import CollapseMonitor
from worldmodel.counterfactual import DEFAULT_PATH as CF_PATH
from worldmodel.counterfactual import load as load_pairs
from worldmodel.dataset import OUT_FORCING
from worldmodel.forcing import (
    FORCING_KINDS,
    WEATHER,
    footprint,
    forcing_scale,
    parse_intents,
    transform_forcing,
    weather_window,
    window,
)
from worldmodel.forecast import NumpyForecaster
from worldmodel.graph import F, links
from worldmodel.jepa import TemporalJEPA
from worldmodel.probes import (
    ALPHAS,
    LinearProbe,
    apply_forecast_probes,
    collapse_stats,
    fit_forecast_probes,
    gather,
    r2,
)
from worldmodel.targets import (
    HORIZONS_DAYS,
    mortality,
    node_inputs,
    scores,
    state_heads,
)
from worldmodel.train import SPLITS
from worldmodel.train_jepa_temporal import DEFAULT_PATH as TEMPORAL_PATH
from worldmodel.train_jepa_temporal import (
    HEADS,
    Worlds,
    conditioned,
    predicted,
    raw_inputs,
    samples_for,
    snapshot_tensors,
    train,
    with_ex_mortality,
)
from worldmodel.train_jepa_temporal import load_model as load_temporal

DEFAULT_PATH = Path(__file__).resolve().parent / "jepa_action.pt"
CF_HEADS = ("contamination", "biomass", "mortality")
CF_HORIZONS = (7, 30)


def forcing_normalizers(worlds: Worlds) -> dict:
    """Forcing scale per kind and weather mean/std, from training windows at every horizon."""
    r = worlds.t.numpy()
    f = np.concatenate([window(worlds.cum, r, h) for h in HORIZONS_DAYS])
    w = np.concatenate([weather_window(worlds.wcum, r, h) for h in HORIZONS_DAYS])
    sd = w.std(0)
    return {"forcing_kinds": list(FORCING_KINDS), "weather": list(WEATHER),
            "forcing_scale": forcing_scale(f).tolist(), "weather_mean": w.mean(0).tolist(),
            "weather_std": np.where(sd > 1e-8, sd, 1.0).tolist()}  # fmt: skip


def forcing_features(worlds: Worlds) -> np.ndarray:
    """(S, H, N, C + W) normalized forcing per patch, with the window weather broadcast."""
    out = []
    for h in HORIZONS_DAYS:
        a = worlds.acts(worlds.t, h)
        w = a["weather"][:, None, :].expand(-1, a["forcing"].shape[1], -1)
        out.append(torch.cat([a["forcing"], w], -1).numpy())
    return np.stack(out, axis=1).astype(np.float32)


def gnn_predictions(samples: dict, batch: int = 256) -> dict[str, np.ndarray]:
    """The shipped Phase 4 GNN (numpy twin) on prepared samples."""
    model = NumpyForecaster()
    m = model.meta
    parts = []
    for i in range(0, len(samples["day"]), batch):
        sl = slice(i, i + batch)
        nodes = (samples["nodes"][sl] - m["node_mean"]) / m["node_std"]
        glob = (samples["glob"][sl] - m["glob_mean"]) / m["glob_std"]
        parts.append(model.decode(*model.raw(nodes, glob, samples["on"][sl], samples["elev"][sl],
                                             samples["link_now"][sl].astype(float))))  # fmt: skip
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


LEVEL_FEATURE = {"contamination": "contaminant", "biomass": "plant_c"}


def levels(samples: dict, head: str, k: int) -> np.ndarray:
    """(S, N) the head's state at t + k in model units: state at t plus the change."""
    return samples["nodes"][..., F[LEVEL_FEATURE[head]]] + samples[f"y_{head}"][..., k]


def fit_level_probes(feats: dict, samples: dict, max_rows: int = 60_000,
                     seed: int = 0) -> dict:  # fmt: skip
    """Least-squares probes from a representation to the state at t + k (contamination and
    biomass), ridge strength by validation MSE.

    A second decoder for counterfactuals. The forecast probes are LAD fits of mostly
    zero changes, which barely respond to rare events like spills; a level probe
    reads what the representation says the state will be.
    """
    fitted = {}
    for head in LEVEL_FEATURE:
        for h in CF_HORIZONS:
            k = HORIZONS_DAYS.index(h)
            xs, ys = {}, {}
            for split in ("train", "val"):
                y = levels(samples[split], head, k)
                sel = np.arange(y.size)
                if split == "train" and y.size > max_rows:
                    sel = np.random.default_rng(seed).permutation(y.size)[:max_rows]
                xs[split] = gather(feats[split], k, sel, y.shape[1], head, 8, 8)
                ys[split] = y.reshape(-1)[sel]
            probe = LinearProbe(xs["train"], ys["train"])
            best = None
            for a in ALPHAS:
                w = probe.ridge(a)
                err = np.mean((probe.predict(w, xs["val"]) - ys["val"]) ** 2)
                if best is None or err < best[0]:
                    best = (err, w)
            probe.xs = probe.y = None
            fitted[(head, k)] = (probe, best[1])
    return fitted


def apply_level_probes(fitted: dict, feats, n_snap: int, n_nodes: int) -> dict:
    out = {}
    for (head, k), (probe, w) in fitted.items():
        x = gather(feats, k, np.arange(n_snap * n_nodes), n_nodes, head, 8, 8)
        out[(head, k)] = probe.predict(w, x).reshape(n_snap, n_nodes)
    return out


def evaluate_forecasts(action: TemporalJEPA, temporal: TemporalJEPA,
                       worlds: dict[str, Worlds], strides: dict) -> tuple[dict, dict]:  # fmt: skip
    """Forecast-probe skill on the test worlds. Returns (metrics, fitted probes per rep)."""
    samples = {s: samples_for(w, strides[s]) for s, w in worlds.items()}
    for s, w in worlds.items():
        assert len(samples[s]["day"]) == len(w.t)
    raw = {s: raw_inputs(samples[s]).astype(np.float32)[:, None] for s in worlds}
    reps = {
        "action": {s: predicted(action, w, w.t) for s, w in worlds.items()},
        "temporal": {s: predicted(temporal, w, w.t) for s, w in worlds.items()},
        "raw": raw,
        "raw+forcing": {s: (raw[s], forcing_features(w)) for s, w in worlds.items()},
        "constant": {
            s: np.zeros((len(w.t), 1, w.n_nodes, 1), np.float32) for s, w in worlds.items()
        },  # fmt: skip
    }
    out = {"test_samples": int(len(samples["test"]["day"])), "probes": {}, "level_r2": {}}
    fitted = {}
    shape = samples["test"]["y_biomass"].shape[:2]
    for name, feats in reps.items():
        fitted[name] = fit_forecast_probes(feats, samples)
        if name in ("action", "temporal", "raw+forcing"):
            lv = fit_level_probes(feats, samples)
            fitted[name + "/level"] = lv
            pt = apply_level_probes(lv, feats["test"], *shape)
            out["level_r2"][name] = {
                f"{hd}@{HORIZONS_DAYS[k]}d": float(r2(levels(samples["test"], hd, k), v))
                for (hd, k), v in pt.items()}  # fmt: skip
        pred = apply_forecast_probes(fitted[name], feats["test"], samples["test"])
        out["probes"][name] = with_ex_mortality(scores(pred, samples["test"]))
    out["probes"]["gnn"] = with_ex_mortality(scores(gnn_predictions(samples["test"]),
                                                    samples["test"]))  # fmt: skip
    out["collapse"] = {f"pred@{h}d": collapse_stats(reps["action"]["test"][:, k])
                       for k, h in enumerate(HORIZONS_DAYS)}  # fmt: skip
    return out, fitted


# --- counterfactuals ---------------------------------------------------------------------------


def pair_inputs(pairs: dict, meta: dict) -> tuple[dict, dict]:
    """Snapshot tensors at t and t - 7 for every pair."""
    return (snapshot_tensors(pairs["x_now"], pairs["g_now"], meta),
            snapshot_tensors(pairs["x_past"], pairs["g_past"], meta))  # fmt: skip


def pair_acts(pairs: dict, fmeta: dict, spill: bool) -> list[dict]:
    """Per horizon, the forcing of one branch: the spill (or nothing), and the shared weather."""
    n, n_nodes = len(pairs["day"]), pairs["x_now"].shape[1]
    raw = np.zeros((n, n_nodes, len(FORCING_KINDS)))
    if spill:
        raw = np.stack([footprint(i) for i in parse_intents("[" + ",".join(pairs["spill"]) + "]")])
    f = torch.as_tensor(transform_forcing(raw, np.array(fmeta["forcing_scale"])),
                        dtype=torch.float32)  # fmt: skip
    wcum = np.cumsum(pairs["w"], axis=1)  # (P, 30, 3): days t + 1 .. t + 30
    acts = []
    for h in HORIZONS_DAYS:
        w = (wcum[:, h - 1] / h - np.array(fmeta["weather_mean"])) / np.array(fmeta["weather_std"])
        acts.append({"forcing": f, "weather": torch.as_tensor(w, dtype=torch.float32)})
    return acts


def branch_latents(model: TemporalJEPA, now: dict, past: dict, acts: list[dict]) -> np.ndarray:
    """(P, H, N, D) predicted latents for one branch."""
    model.eval()
    with torch.no_grad():
        out = []
        for k in range(len(HORIZONS_DAYS)):
            kk = torch.full((len(now["nodes"]),), k, dtype=torch.long)
            act = acts[k] if conditioned(model) else None
            out.append(model.predict(now, past, kk, act)[1])
    return torch.stack(out, dim=1).numpy()


def branch_samples(pairs: dict, branch: str) -> dict[str, np.ndarray]:
    """True targets (as targets.build_samples defines them) for one branch, plus probe inputs."""
    x_now, fut = pairs["x_now"], pairs[f"x_{branch}"]  # fut: (P, H, N, F)
    now = state_heads(x_now)
    later = [state_heads(fut[:, k]) for k in range(len(HORIZONS_DAYS))]
    out = {f"y_{n}": np.stack([later[k][n] - now[n] for k in range(len(HORIZONS_DAYS))], -1)
           for n in ("biomass", "contamination", "moisture")}  # fmt: skip
    plant_now, plant_fut = pairs["plant_now"], pairs[f"plant_{branch}"]
    out["y_mortality"] = np.stack([
        np.stack([mortality(plant_now[i], plant_fut[i, k]) for i in range(len(x_now))])
        for k in range(len(HORIZONS_DAYS))], -1)  # fmt: skip
    rows = cols = int(np.sqrt(x_now.shape[1]))
    link_now = links(x_now, rows, cols).astype(np.float64)
    out["y_links"] = np.stack([links(fut[:, k], rows, cols) for k in range(len(HORIZONS_DAYS))],
                              -1).astype(np.float64)  # fmt: skip
    out["p_links"] = np.repeat(link_now[..., None], len(HORIZONS_DAYS), -1)
    out["nodes"] = node_inputs(x_now, pairs["x_past"])
    out["glob"] = pairs["g_now"]
    return out


def effect_metrics(pred: np.ndarray, true: np.ndarray, floor: float) -> dict:
    """Predicted vs true effect maps (P, N). `floor` defines a clearly affected patch."""
    err = np.abs(pred - true).mean()
    base = np.abs(true).mean()
    hit = np.abs(true) > floor
    corr = []
    for p, t in zip(pred, true, strict=True):
        if t.std() > 1e-12 and p.std() > 1e-12:
            corr.append(float(np.corrcoef(p, t)[0, 1]))
        else:
            corr.append(0.0)
    peak_true = np.abs(true).argmax(1)
    peak_pred = np.abs(pred).argmax(1)
    has_pred = np.abs(pred).max(1) > 0
    return {
        "effect_skill": float(1 - err / base) if base > 0 else 0.0,
        "true_mean_abs_effect": float(base), "pred_mean_abs_effect": float(np.abs(pred).mean()),
        "affected_patches": int(hit.sum()),
        "direction_agreement": float((np.sign(pred[hit]) == np.sign(true[hit])).mean())
        if hit.any() else None,
        "magnitude_ratio": float(pred.sum() / true.sum()) if abs(true.sum()) > 0 else None,
        "spatial_corr": float(np.mean(corr)),
        "peak_patch_found": float(np.mean((peak_pred == peak_true) & has_pred)),
    }  # fmt: skip


FLOORS = {"contamination": 0.01, "biomass": 0.001, "mortality": 0.01}


def evaluate_counterfactuals(models: dict, fitted: dict, pairs: dict, meta: dict,
                             fmeta: dict) -> dict:  # fmt: skip
    """Per representation, head and horizon: how well the predicted spill effect matches."""
    now, past = pair_inputs(pairs, meta)
    truth = {b: branch_samples(pairs, b) for b in ("base", "spill")}
    out = {"pairs": int(len(pairs["day"])), "true": {}}
    for head in CF_HEADS:
        for h in CF_HORIZONS:
            k = HORIZONS_DAYS.index(h)
            eff = truth["spill"][f"y_{head}"][..., k] - truth["base"][f"y_{head}"][..., k]
            out["true"][f"{head}@{h}d"] = {
                "mean_abs_effect": float(np.abs(eff).mean()), "max_effect": float(eff.max()),
                "min_effect": float(eff.min()),
                "affected_patches": int((np.abs(eff) > FLOORS[head]).sum())}  # fmt: skip
    for name, model in models.items():
        preds = {}
        for b, spill in (("base", False), ("spill", True)):
            if name == "raw+forcing":
                acts = pair_acts(pairs, fmeta, spill)
                raw = raw_inputs(truth[b]).astype(np.float32)[:, None]
                f = np.stack([torch.cat([a["forcing"], a["weather"][:, None, :].expand(
                    -1, a["forcing"].shape[1], -1)], -1).numpy() for a in acts], 1)  # fmt: skip
                feats = (raw, f.astype(np.float32))
            else:
                feats = branch_latents(model, now, past, pair_acts(pairs, fmeta, spill))
            preds[b] = apply_forecast_probes(fitted[name], feats, truth[b])
            shape = truth[b]["y_biomass"].shape[:2]
            for (head, k), v in apply_level_probes(fitted[name + "/level"], feats, *shape).items():
                preds[b][f"level/{head}/{k}"] = v  # levels at t + k; t is shared by both branches
        res = {}
        for head in CF_HEADS:
            for h in CF_HORIZONS:
                k = HORIZONS_DAYS.index(h)
                true = truth["spill"][f"y_{head}"][..., k] - truth["base"][f"y_{head}"][..., k]
                pred = preds["spill"][head][..., k] - preds["base"][head][..., k]
                res[f"{head}@{h}d"] = effect_metrics(pred, true, FLOORS[head])
                y = truth["spill"][f"y_{head}"][..., k]
                res[f"{head}@{h}d"]["spill_branch_mae"] = float(
                    np.abs(preds["spill"][head][..., k] - y).mean()
                )
                lk = f"level/{head}/{k}"
                if lk in preds["spill"]:
                    res[f"{head}@{h}d (level probe)"] = effect_metrics(
                        preds["spill"][lk] - preds["base"][lk], true, FLOORS[head]
                    )
        out[name] = res
    return out


def summary(ev: dict, cf: dict) -> str:
    names = list(ev["probes"])
    lines = [f"{'skill vs persistence':<30}" + "".join(f"{m:>12}" for m in names)]
    for key in (*[f"{n}@{h}d" for n in HEADS for h in HORIZONS_DAYS],
                *[f"mean_skill@{h}d" for h in HORIZONS_DAYS],
                *[f"mean_skill_ex_mortality@{h}d" for h in HORIZONS_DAYS]):  # fmt: skip
        vals = [ev["probes"][m][key] for m in names]
        vals = [v if isinstance(v, float) else v["skill"] for v in vals]
        lines.append(f"  {key:<28}" + "".join(f"{v:12.3f}" for v in vals))
    lines.append(f"\ncounterfactual spill effect ({cf['pairs']} pairs, test worlds)")
    reps = [m for m in cf if m not in ("pairs", "true")]
    for key, t in cf["true"].items():
        lines.append(f"  {key:<18} true mean |effect| {t['mean_abs_effect']:.5f}, "
                     f"{t['affected_patches']} affected patches")  # fmt: skip
        rows = [(m, kk) for kk in (key, key + " (level probe)") for m in reps if kk in cf[m]]
        for m, kk in rows:
            r = cf[m][kk]
            m = m if kk == key else m + " (lvl)"
            d = r["direction_agreement"]
            mr = r["magnitude_ratio"]
            lines.append(
                f"    {m:<18} skill {r['effect_skill']:+.3f}  dir "
                + (f"{d:.2f}" if d is not None else " n/a")
                + "  mag "
                + (f"{mr:+.2f}" if mr is not None else " n/a")
                + f"  corr {r['spatial_corr']:+.2f}  peak {r['peak_patch_found']:.2f}"
                + (
                    f"  spill-branch MAE {r['spill_branch_mae']:.5f}"
                    if "spill_branch_mae" in r
                    else ""
                )
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=OUT_FORCING)
    ap.add_argument("--pairs", type=Path, default=CF_PATH)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--var-weight", type=float, default=0.1)
    ap.add_argument("--train-stride", type=int, default=2, help="every k-th day for training")
    ap.add_argument("--probe-stride", type=int, default=2, help="every k-th day, probe train set")
    ap.add_argument("--init", type=Path, default=TEMPORAL_PATH, help="step-2 checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()

    temporal, meta = load_temporal(args.init)
    probe_worlds = Worlds(SPLITS["train"], args.data, meta, args.train_stride, forcing_meta={})
    fmeta = forcing_normalizers(probe_worlds)
    train_worlds = probe_worlds
    train_worlds.forcing_meta = fmeta
    print(f"training samples: {len(train_worlds.t)} from {len(SPLITS['train'])} worlds; "
          f"forcing scale {np.round(fmeta['forcing_scale'], 2).tolist()}", flush=True)  # fmt: skip

    model = TemporalJEPA(**temporal.config, n_forcing=len(FORCING_KINDS), n_weather=len(WEATHER))
    missing, unexpected = model.load_state_dict(temporal.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(("predictor.act.", "predictor.weather."))
                                  for k in missing), (missing, unexpected)  # fmt: skip
    monitor = CollapseMonitor()
    curve = train(model, train_worlds, epochs=args.epochs, batch=args.batch, lr=args.lr,
                  monitor=monitor,
                  var_weight=args.var_weight, seed=args.seed,
                  log=lambda s: print(s, flush=True))  # fmt: skip
    train_s = time.time() - t0
    n_train = int(len(train_worlds.t))
    del train_worlds, probe_worlds

    strides = {"train": args.probe_stride, "val": 1, "test": 1}
    worlds = {s: Worlds(SPLITS[s], args.data, meta, strides[s], fmeta) for s in SPLITS}
    ev, fitted = evaluate_forecasts(model, temporal, worlds, strides)
    del worlds
    pairs = load_pairs(args.pairs)
    cf = evaluate_counterfactuals({"action": model, "temporal": temporal,
                                   "raw+forcing": None}, fitted, pairs, meta, fmeta)  # fmt: skip
    print(summary(ev, cf))

    torch.save({"config": model.config, "meta": meta, "forcing_meta": fmeta,
                "state_dict": {k: v.half() if v.is_floating_point() else v
                               for k, v in model.state_dict().items() if k != "mask"}},
               args.out)  # fmt: skip
    metrics_path = args.out.with_name(args.out.stem + "_metrics.json")
    metrics_path.write_text(json.dumps({
        "splits": {k: list(v) for k, v in SPLITS.items()}, "data": str(args.data),
        "train_samples": n_train,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "init_from": str(args.init), "forcing_meta": fmeta,
        "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t0, 1),
        "weights": "fp16 checkpoint; metrics from the full-precision model",
        "curve": curve, "collapse_monitor": monitor.report(), "forecast": ev,
        "counterfactual": cf,
    }, indent=1))  # fmt: skip
    print(f"wrote {args.out} and {metrics_path} ({time.time() - t0:.0f}s)")


def load_model(path: Path = DEFAULT_PATH) -> tuple[TemporalJEPA, dict, dict]:
    ckpt = torch.load(path, weights_only=True)
    model = TemporalJEPA(**ckpt["config"])
    missing, unexpected = model.load_state_dict(
        {k: v.float() for k, v in ckpt["state_dict"].items()}, strict=False
    )
    assert set(missing) == {"mask"} and not unexpected, (missing, unexpected)  # mask is rebuilt
    return model.eval(), ckpt["meta"], ckpt["forcing_meta"]


if __name__ == "__main__":
    main()
