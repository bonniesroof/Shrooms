"""Train the static graph JEPA and evaluate it with linear probes.

    uv sync --group worldmodel
    uv run python -m worldmodel.dataset --seeds 0-27        # ~15 min on 2 cores (Phase 4 data)
    uv run python -m worldmodel.train_jepa                  # ~9 min on 2 cores

Self-supervised: no forecast targets are used in training. Each sample is one
daily snapshot; a block of patches is hidden and the model predicts their
target-encoder latents from the rest.

Evaluation, with the same world split as train.py (train 0-19, val 20-23 picks
the ridge strength, test 24-27 is reported):
    full    probe each patch's latent from the whole snapshot (target encoder).
            Raw features contain the probed quantities, so the raw probe is a
            near-perfect ceiling here; this checks the latents kept the information.
    masked  hide a block of patches and probe the *predicted* latents of the hidden
            ones: can the model infer a patch it can't see? The raw baseline is the
            mean raw features of the visible neighbours plus the patch's own elevation.
Each is run on the trained JEPA, the same network at random init, and raw features.

Writes worldmodel/jepa_static.pt (weights + config + normalization) and
worldmodel/jepa_static_metrics.json.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from worldmodel.dataset import OUT, load
from worldmodel.graph import F, on_network
from worldmodel.jepa import StaticJEPA, block_mask, momentum_at
from worldmodel.probes import NODE_PROBES, collapse_stats, probe_targets, run_probes
from worldmodel.targets import transform
from worldmodel.train import SPLITS

DEFAULT_PATH = Path(__file__).resolve().parent / "jepa_static.pt"


def load_snapshots(seeds, data_dir: Path, stride: int = 1) -> dict[str, np.ndarray]:
    """Raw daily snapshots from runs: x (S, N, F), g (S, G), world (S,)."""
    xs, gs, ws = [], [], []
    for seed in seeds:
        run = load(data_dir / f"run_{seed}.npz")
        xs.append(run["x"][::stride])
        gs.append(run["g"][::stride])
        ws.append(np.full(len(xs[-1]), seed))
    return {"x": np.concatenate(xs), "g": np.concatenate(gs), "world": np.concatenate(ws)}


def normalizers(train: dict) -> dict:
    t = transform(train["x"]).reshape(-1, train["x"].shape[-1])
    sd, gsd = t.std(0), train["g"].std(0)
    return {"node_mean": t.mean(0).tolist(), "node_std": np.where(sd > 1e-8, sd, 1.0).tolist(),
            "glob_mean": train["g"].mean(0).tolist(),
            "glob_std": np.where(gsd > 1e-8, gsd, 1.0).tolist()}  # fmt: skip


def inputs(d: dict, meta: dict) -> dict[str, torch.Tensor]:
    """Raw snapshots -> model tensors: normalized nodes and globals, network flags, elevation."""
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    return {
        "nodes": t((transform(d["x"]) - meta["node_mean"]) / meta["node_std"]),
        "glob": t((d["g"] - meta["glob_mean"]) / meta["glob_std"]),
        "on": t(on_network(d["x"])),
        "elev": t(d["x"][..., F["elevation"]]),
    }  # fmt: skip


def train(model: StaticJEPA, tt: dict, *, epochs: int, batch: int, lr: float, mask_ratio: float,
          var_weight: float, seed: int, log=print) -> list[dict]:  # fmt: skip
    """Self-supervised training. Returns a per-epoch curve with loss and collapse stats."""
    rows, cols = model.config["rows"], model.config["cols"]
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.04)
    n = len(tt["nodes"])
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1),
                                                pct_start=0.1)  # fmt: skip
    probe_idx = torch.randperm(n, generator=gen)[: min(n, 512)]
    curve, step, t0 = [], 0, time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        sums = {"loss": 0.0, "jepa": 0.0, "var": 0.0}
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            hidden = block_mask(len(idx), rows, cols, mask_ratio, gen)
            out = model.loss(tt["nodes"][idx], tt["glob"][idx], tt["on"][idx], tt["elev"][idx],
                             hidden, var_weight)  # fmt: skip
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            model.ema_update(momentum_at(step, steps))
            for k in sums:
                sums[k] += float(out[k].detach()) * len(idx)
        model.eval()
        z = encode(model, {k: v[probe_idx] for k, v in tt.items()})
        stats = collapse_stats(z)
        row = {"epoch": epoch, **{k: v / n for k, v in sums.items()},
               "target_std": stats["std"], "target_rank": stats["effective_rank"],
               "seconds": round(time.time() - t0, 1)}  # fmt: skip
        curve.append(row)
        log(f"epoch {epoch:2d} loss {row['loss']:.4f} (jepa {row['jepa']:.4f} var {row['var']:.4f})"
            f"  latent std {stats['std']:.3f} eff. rank {stats['effective_rank']:.1f}"
            f"/{stats['dims']}  ({row['seconds']:.0f}s)")  # fmt: skip
    return curve


def encode(model: StaticJEPA, tt: dict, batch: int = 512) -> np.ndarray:
    """(S, N, D) target-encoder latents of full snapshots."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tt["nodes"]), batch):
            sl = slice(i, i + batch)
            out.append(model.encode(tt["nodes"][sl], tt["glob"][sl], tt["on"][sl], tt["elev"][sl]))
    return torch.cat(out).numpy()


def predict_masked(model: StaticJEPA, tt: dict, hidden: torch.Tensor, batch: int = 512):
    """(S, N, D) predicted latents with `hidden` patches masked out of the context."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tt["nodes"]), batch):
            sl = slice(i, i + batch)
            _, pred = model.predict(tt["nodes"][sl], tt["glob"][sl], tt["on"][sl],
                                    tt["elev"][sl], hidden[sl])  # fmt: skip
            out.append(pred)
    return torch.cat(out).numpy()


def raw_features(tt: dict) -> np.ndarray:
    """(S, N, F+G) normalized patch features with the atmosphere node appended to each patch."""
    nodes, glob = tt["nodes"].numpy(), tt["glob"].numpy()
    g = np.broadcast_to(glob[:, None], (*nodes.shape[:2], glob.shape[-1]))
    return np.concatenate([nodes, g], -1)


def raw_masked(tt: dict, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Raw baseline for hidden patches: mean features of visible neighbours, own elevation."""
    nodes, glob = tt["nodes"].numpy(), tt["glob"].numpy()
    vis = (~hidden).astype(np.float64)  # (S, N)
    w = mask[None] * vis[:, None, :]  # [s, target, source]
    deg = w.sum(-1, keepdims=True)
    neigh = np.einsum("stn,snf->stf", w, nodes) / np.maximum(deg, 1)
    elev = nodes[..., F["elevation"] : F["elevation"] + 1]
    g = np.broadcast_to(glob[:, None], (*nodes.shape[:2], glob.shape[-1]))
    return np.concatenate([neigh, (deg > 0).astype(np.float64), elev, g], -1)


def evaluate(model: StaticJEPA, rand: StaticJEPA, tt: dict, targets: dict, mask_ratio: float,
             seed: int = 1234) -> dict:  # fmt: skip
    """Probe R² for JEPA, random init and raw features, full and masked."""
    rows, cols = model.config["rows"], model.config["cols"]
    gen = torch.Generator().manual_seed(seed)
    hidden = {s: block_mask(len(v["nodes"]), rows, cols, mask_ratio, gen) for s, v in tt.items()}
    keep = {s: h.numpy() for s, h in hidden.items()}
    mask = model.mask.numpy()
    feats = {
        "full": {
            "jepa": {s: encode(model, v) for s, v in tt.items()},
            "random": {s: encode(rand, v) for s, v in tt.items()},
            "raw": {s: raw_features(v) for s, v in tt.items()},
        },
        "masked": {
            "jepa": {s: predict_masked(model, v, hidden[s]) for s, v in tt.items()},
            "random": {s: predict_masked(rand, v, hidden[s]) for s, v in tt.items()},
            "raw": {s: raw_masked(v, keep[s], mask) for s, v in tt.items()},
        },
    }
    out = {"full": {}, "masked": {}, "collapse": {}}
    for setting, reps in feats.items():
        for name, f in reps.items():
            rows_mask = keep if setting == "masked" else None
            out[setting][name] = run_probes(f, targets, rows, cols, rows_mask=rows_mask)
    for name in ("jepa", "random"):
        out["collapse"][name] = collapse_stats(feats["full"][name]["test"])
    return out


def summary_table(ev: dict) -> str:
    lines = []
    for setting in ("full", "masked"):
        names = [*NODE_PROBES, "links"] if setting == "full" else list(NODE_PROBES)
        lines.append(f"{setting:<7} " + " ".join(f"{n[:9]:>9}" for n in names) + "      mean")
        for rep in ("jepa", "random", "raw"):
            r = ev[setting][rep]
            lines.append(f"  {rep:<6}" + " ".join(f"{r[n]['test_r2']:9.3f}" for n in names)
                         + f" {r['mean_test_r2']:9.3f}")  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--var-weight", type=float, default=0.1)
    ap.add_argument("--train-stride", type=int, default=2, help="use every k-th day for training")
    ap.add_argument("--probe-stride", type=int, default=5, help="every k-th day for probes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    t0 = time.time()
    snaps = load_snapshots(SPLITS["train"], args.data, args.train_stride)
    meta = normalizers(snaps)
    tt = inputs(snaps, meta)
    print(f"training snapshots: {len(snaps['x'])} from {len(SPLITS['train'])} worlds", flush=True)

    model = StaticJEPA(n_in=tt["nodes"].shape[-1], n_glob=tt["glob"].shape[-1],
                       hidden=args.hidden)  # fmt: skip
    rand = copy.deepcopy(model)  # the same network at its initial weights, never trained
    curve = train(model, tt, epochs=args.epochs, batch=args.batch, lr=args.lr,
                  mask_ratio=args.mask_ratio, var_weight=args.var_weight, seed=args.seed,
                  log=lambda s: print(s, flush=True))  # fmt: skip
    train_s = time.time() - t0

    probe = {s: load_snapshots(v, args.data, args.probe_stride) for s, v in SPLITS.items()}
    ptt = {s: inputs(v, meta) for s, v in probe.items()}
    targets = {s: probe_targets(v["x"]) for s, v in probe.items()}
    ev = evaluate(model, rand, ptt, targets, args.mask_ratio)
    print(summary_table(ev))
    print("collapse (test latents):", json.dumps(ev["collapse"]))

    torch.save({"config": model.config, "meta": meta,
                "state_dict": {k: v.half() if v.is_floating_point() else v
                               for k, v in model.state_dict().items() if k != "mask"}},
               args.out)  # fmt: skip
    metrics_path = args.out.with_name(args.out.stem + "_metrics.json")
    metrics_path.write_text(json.dumps({
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "snapshots": {"train": len(snaps["x"]), **{f"probe_{k}": len(v["x"])
                                                   for k, v in probe.items()}},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t0, 1),
        "curve": curve, **ev,
    }, indent=1))  # fmt: skip
    print(f"wrote {args.out} and {metrics_path} ({time.time() - t0:.0f}s)")


def load_model(path: Path = DEFAULT_PATH) -> tuple[StaticJEPA, dict]:
    ckpt = torch.load(path, weights_only=True)
    model = StaticJEPA(**ckpt["config"])
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()}, strict=False)
    return model.eval(), ckpt["meta"]


if __name__ == "__main__":
    main()
