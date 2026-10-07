"""Train the temporal JEPA (latents at t -> latents at t + k) and score it with forecast probes.

    uv run python -m worldmodel.train_jepa              # static JEPA first (its encoders seed this)
    uv run python -m worldmodel.train_jepa_temporal     # ~8 min on 2 cores

Self-supervised: the context encoder sees the whole snapshot at t (and t - 7 d,
for the trend, like the GNN's inputs); the predictor, told the horizon k, predicts
what the EMA target encoder makes of the snapshot at t + k, for k in
targets.HORIZONS_DAYS (1, 7, 30 days). No forecast targets are used in training.

Evaluation is a probe, not Gate C: freeze everything, take the predicted latents
for each horizon, fit one ridge probe per head and horizon onto the Phase 4
forecast targets (train worlds 0-19; ridge strength by the reported metric on
val worlds 20-23), and score on test worlds 24-27 with targets.scores (MAE, Brier
for links), the metric forecaster_metrics.json reports. Compared with:
    persistence      the Phase 4 baselines
    gnn              the supervised Phase 4 GNN, read from forecaster_metrics.json
    raw              the same probes on the GNN's raw inputs (features now, 7-day change, weather)
    raw+temporal     raw inputs and predicted latents together: do the latents add anything?
    constant         an intercept-only probe: the best constant change (a control; for the
                     mostly-zero die-back target it is "no die-back", which MAE rewards)
    static           the same probes on the static JEPA's latents at t
    temporal         the same probes on the temporal JEPA's predicted latents at t + k

Writes worldmodel/jepa_temporal.pt and worldmodel/jepa_temporal_metrics.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from worldmodel.dataset import OUT, load
from worldmodel.graph import F, on_network
from worldmodel.jepa import StaticJEPA, TemporalJEPA, momentum_at
from worldmodel.probes import collapse_stats, forecast_probes
from worldmodel.targets import HORIZONS_DAYS, TREND_DAYS, build_samples, scores, transform
from worldmodel.train import SPLITS
from worldmodel.train_jepa import DEFAULT_PATH as STATIC_PATH
from worldmodel.train_jepa import load_model as load_static
from worldmodel.train_jepa import normalizers

DEFAULT_PATH = Path(__file__).resolve().parent / "jepa_temporal.pt"
GNN_METRICS = Path(__file__).resolve().parent / "forecaster_metrics.json"
HEADS = ("biomass", "contamination", "moisture", "mortality", "links")


def snapshot_tensors(x: np.ndarray, g: np.ndarray, meta: dict) -> dict[str, torch.Tensor]:
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    return {"nodes": t((transform(x) - meta["node_mean"]) / meta["node_std"]),
            "glob": t((g - meta["glob_mean"]) / meta["glob_std"]),
            "on": t(on_network(x)), "elev": t(x[..., F["elevation"]])}  # fmt: skip


class Worlds:
    """Raw daily snapshots of several worlds, indexed by (world, day) for t, t - 7, t + k."""

    def __init__(self, seeds, data_dir: Path, meta: dict, stride: int = 1):
        self.runs = [load(data_dir / f"run_{s}.npz") for s in seeds]
        self.snaps = [snapshot_tensors(r["x"], r["g"], meta) for r in self.runs]
        offsets = np.cumsum([0] + [len(r["x"]) for r in self.runs])
        self.flat = {k: torch.cat([s[k] for s in self.snaps]) for k in self.snaps[0]}
        self.n_nodes = int(self.flat["nodes"].shape[1])
        idx = []
        for w, run in enumerate(self.runs):
            back = max(max(HORIZONS_DAYS), TREND_DAYS)  # = targets.valid_days
            days = np.arange(back, len(run["x"]) - max(HORIZONS_DAYS))[::stride]
            idx.append(offsets[w] + days)
        self.t = torch.as_tensor(np.concatenate(idx))

    def at(self, rows: torch.Tensor, shift: int = 0) -> dict[str, torch.Tensor]:
        return {k: v[rows + shift] for k, v in self.flat.items()}


def train(model: TemporalJEPA, worlds: Worlds, *, epochs: int, batch: int, lr: float,
          var_weight: float, seed: int, log=print) -> list[dict]:  # fmt: skip
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.04)
    n = len(worlds.t)
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1),
                                                pct_start=0.1)  # fmt: skip
    probe_rows = worlds.t[torch.randperm(n, generator=gen)[: min(n, 512)]]
    curve, step, t0 = [], 0, time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        sums = {"loss": 0.0, "jepa": 0.0, "var": 0.0}
        per_h = np.zeros(len(HORIZONS_DAYS))
        for i in range(0, n, batch):
            rows = worlds.t[perm[i : i + batch]]
            out = model.loss(worlds.at(rows), worlds.at(rows, -TREND_DAYS),
                             [worlds.at(rows, h) for h in HORIZONS_DAYS], var_weight)  # fmt: skip
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            model.ema_update(momentum_at(step, steps))
            for k in sums:
                sums[k] += float(out[k].detach()) * len(rows)
            per_h += out["per_horizon"].detach().numpy() * len(rows)
        z = predicted(model, worlds, probe_rows)
        stats = collapse_stats(z[:, 1])  # the 7-day predictions
        row = {"epoch": epoch, **{k: v / n for k, v in sums.items()},
               **{f"jepa@{h}d": float(v / n) for h, v in zip(HORIZONS_DAYS, per_h, strict=True)},
               "pred_std": stats["std"], "pred_rank": stats["effective_rank"],
               "seconds": round(time.time() - t0, 1)}  # fmt: skip
        curve.append(row)
        log(f"epoch {epoch:2d} loss {row['loss']:.4f} | "
            + " ".join(f"{h}d {row[f'jepa@{h}d']:.4f}" for h in HORIZONS_DAYS)
            + f" | pred std {stats['std']:.3f} eff. rank {stats['effective_rank']:.1f}"
            f"/{stats['dims']}  ({row['seconds']:.0f}s)")  # fmt: skip
    return curve


def predicted(model: TemporalJEPA, worlds: Worlds, rows: torch.Tensor,
              batch: int = 256) -> np.ndarray:  # fmt: skip
    """(S, H, N, D) frozen predicted latents for t + k, every horizon."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            r = rows[i : i + batch]
            now, past = worlds.at(r), worlds.at(r, -TREND_DAYS)
            per_h = []
            for k in range(len(HORIZONS_DAYS)):
                kk = torch.full((len(r),), k, dtype=torch.long)
                per_h.append(model.predict(now, past, kk)[1])
            out.append(torch.stack(per_h, dim=1))
    return torch.cat(out).numpy()


def static_latents(model: StaticJEPA, worlds: Worlds, rows: torch.Tensor,
                   batch: int = 512) -> np.ndarray:  # fmt: skip
    """(S, 1, N, D) static JEPA target-encoder latents at t."""
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            s = worlds.at(rows[i : i + batch])
            out.append(model.encode(s["nodes"], s["glob"], s["on"], s["elev"]))
    return torch.cat(out).numpy()[:, None]


def samples_for(worlds: Worlds, stride: int) -> dict[str, np.ndarray]:
    """Phase 4 samples (targets.build_samples) at the same days as worlds.t."""
    parts = [build_samples(r) for r in worlds.runs]
    return {k: np.concatenate([p[k][::stride] for p in parts]) for k in parts[0]}


def raw_inputs(s: dict) -> np.ndarray:
    """(S, N, 2F + G) the GNN's inputs: features now, 7-day change, atmosphere node."""
    g = np.broadcast_to(s["glob"][:, None], (*s["nodes"].shape[:2], s["glob"].shape[-1]))
    return np.concatenate([s["nodes"], g], -1)


def evaluate(model: TemporalJEPA, static: StaticJEPA | None, worlds: dict[str, Worlds],
             strides: dict[str, int]) -> dict:  # fmt: skip
    samples = {s: samples_for(w, strides[s]) for s, w in worlds.items()}
    for s, w in worlds.items():
        assert len(samples[s]["day"]) == len(w.t)
    reps = {
        "temporal": {s: predicted(model, w, w.t) for s, w in worlds.items()},
        "raw": {s: raw_inputs(samples[s]).astype(np.float32)[:, None] for s in worlds},
        "constant": {
            s: np.zeros((len(w.t), 1, w.n_nodes, 1), np.float32) for s, w in worlds.items()
        },
    }
    reps["raw+temporal"] = {s: (reps["raw"][s], reps["temporal"][s]) for s in worlds}
    if static is not None:
        reps["static"] = {s: static_latents(static, w, w.t) for s, w in worlds.items()}
    out = {"test_samples": int(len(samples["test"]["day"])), "probes": {}}
    for name, feats in reps.items():
        out["probes"][name] = scores(forecast_probes(feats, samples), samples["test"])
    gnn = json.loads(GNN_METRICS.read_text())["test"]
    out["gnn"] = gnn
    out["collapse"] = {f"pred@{hd}d": collapse_stats(reps["temporal"]["test"][:, k])
                       for k, hd in enumerate(HORIZONS_DAYS)}  # fmt: skip
    return out


def summary_table(ev: dict) -> str:
    """Skill vs persistence (1 - error / persistence error) on test worlds."""
    models = [
        m for m in ("temporal", "static", "raw", "raw+temporal", "constant") if m in ev["probes"]
    ]
    lines = [f"{'skill vs persistence':<22}" + "".join(f"{m:>13}" for m in (*models, "gnn"))]
    for name in (*HEADS, "mean_skill"):
        for h in HORIZONS_DAYS:
            key = f"{name}@{h}d"
            vals = [ev["probes"][m][key] for m in models] + [ev["gnn"][key]]
            vals = [v if isinstance(v, float) else v["skill"] for v in vals]
            lines.append(f"  {key:<20}" + "".join(f"{v:13.3f}" for v in vals))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--var-weight", type=float, default=0.1)
    ap.add_argument("--train-stride", type=int, default=2, help="every k-th day for training")
    ap.add_argument("--probe-stride", type=int, default=2, help="every k-th day, probe train set")
    ap.add_argument(
        "--init",
        type=Path,
        default=STATIC_PATH,
        help="static JEPA checkpoint to start the encoders from ('none' to skip)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()

    static, init = None, str(args.init) != "none"
    if init:
        static, meta = load_static(args.init)
    else:
        from worldmodel.train_jepa import load_snapshots

        meta = normalizers(load_snapshots(SPLITS["train"], args.data, args.train_stride))
    train_worlds = Worlds(SPLITS["train"], args.data, meta, args.train_stride)
    print(f"training samples: {len(train_worlds.t)} from {len(SPLITS['train'])} worlds", flush=True)

    model = TemporalJEPA(n_in=train_worlds.flat["nodes"].shape[-1],
                         n_glob=train_worlds.flat["glob"].shape[-1])  # fmt: skip
    if init:
        model.init_encoders(torch.load(args.init, weights_only=True)["state_dict"])
    curve = train(model, train_worlds, epochs=args.epochs, batch=args.batch, lr=args.lr,
                  var_weight=args.var_weight, seed=args.seed,
                  log=lambda s: print(s, flush=True))  # fmt: skip
    train_s = time.time() - t0
    n_train = int(len(train_worlds.t))
    del train_worlds  # free memory for the probes

    strides = {"train": args.probe_stride, "val": 1, "test": 1}
    worlds = {s: Worlds(SPLITS[s], args.data, meta, strides[s]) for s in SPLITS}
    ev = evaluate(model, static, worlds, strides)
    print(summary_table(ev))
    print("collapse (test predictions):", json.dumps(ev["collapse"]))

    torch.save({"config": model.config, "meta": meta,
                "state_dict": {k: v.half() if v.is_floating_point() else v
                               for k, v in model.state_dict().items() if k != "mask"}},
               args.out)  # fmt: skip
    metrics_path = args.out.with_name(args.out.stem + "_metrics.json")
    ev_out = {k: v for k, v in ev.items() if k != "gnn"}
    metrics_path.write_text(json.dumps({
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "train_samples": n_train,
        "probe_samples": {s: int(len(w.t)) for s, w in worlds.items()},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "init_from_static": init,
        "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t0, 1),
        "curve": curve, **ev_out,
        "gnn_test_from": "worldmodel/forecaster_metrics.json",
    }, indent=1))  # fmt: skip
    print(f"wrote {args.out} and {metrics_path} ({time.time() - t0:.0f}s)")


def load_model(path: Path = DEFAULT_PATH) -> tuple[TemporalJEPA, dict]:
    ckpt = torch.load(path, weights_only=True)
    model = TemporalJEPA(**ckpt["config"])
    missing, unexpected = model.load_state_dict(
        {k: v.float() for k, v in ckpt["state_dict"].items()}, strict=False
    )
    assert set(missing) == {"mask"} and not unexpected, (missing, unexpected)  # mask is rebuilt
    return model.eval(), ckpt["meta"]


if __name__ == "__main__":
    main()
