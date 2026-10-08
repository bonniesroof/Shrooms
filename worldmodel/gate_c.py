"""Gate C: prediction heads on a pretrained backbone vs the supervised Phase 4 GNN.

    uv run python -m worldmodel.gate_c --pretrain                    # generative baseline, ~3.5 min
    uv run python -m worldmodel.gate_c --backbone temporal --mode frozen    # ~7.5 min, 30 epochs
    uv run python -m worldmodel.gate_c --backbone temporal --mode finetune  # ~8 min, 12 epochs
    uv run python -m worldmodel.gate_c --summary          # collect into gate_c_metrics.json

Roadmap gate: "the JEPA model with prediction heads beats the GNN baseline at 7
and 30 days". Everything that isn't the backbone is the GNN's protocol
(train.py): Phase 4 data, world split (train 0-19, val 20-23 picks the epoch by
mean 7 d + 30 d skill, test 24-27 reported), the same heads, loss (train.loss_fn)
and metric (targets.scores: MAE, Brier for links).

Backbones, all the same network (encoder + horizon-conditioned predictor):
    temporal     the step-2 temporal JEPA (worldmodel/jepa_temporal.pt)
    generative   the same network trained the same way (same init from the static
                 JEPA, data, epochs) but to regress the patch features at t + k in
                 input space instead of target-encoder latents (--pretrain trains it
                 and writes worldmodel/generative.pt)
    scratch      the same network, randomly initialized (supervised from scratch)

Heads: per horizon k, a node head on [predicted latent at t + k, latent at t]
-> biomass, contamination, moisture and die-back, and a link head on each
adjacent pair (the GNN's link head, with its persistence prior).
    frozen       backbone fixed; only the heads train (30 epochs, as the GNN)
    finetune     everything trains; backbone at a tenth of the heads' learning rate
                 (12 epochs: about the same wall time as 30 frozen epochs)

Writes worldmodel/gate_c_<backbone>_<mode>.json (and the generative backbone's
generative.pt + generative_metrics.json with --pretrain). Metrics come from the
full-precision model.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from worldmodel.collapse import CollapseMonitor
from worldmodel.dataset import OUT
from worldmodel.gnn import mlp
from worldmodel.jepa import GenerativeForecaster, TemporalJEPA
from worldmodel.probes import forecast_probes
from worldmodel.targets import CONTINUOUS, HORIZONS_DAYS, NODE_HEADS, TREND_DAYS, scores
from worldmodel.train import SPLITS, loss_fn
from worldmodel.train import normalizers as gnn_normalizers
from worldmodel.train_jepa import DEFAULT_PATH as STATIC_PATH
from worldmodel.train_jepa_temporal import DEFAULT_PATH as TEMPORAL_PATH
from worldmodel.train_jepa_temporal import (
    Worlds,
    predicted,
    samples_for,
    train,
    with_ex_mortality,
)
from worldmodel.train_jepa_temporal import load_model as load_temporal

HERE = Path(__file__).resolve().parent
GENERATIVE_PATH = HERE / "generative.pt"
GNN_METRICS = HERE / "forecaster_metrics.json"


class Heads(nn.Module):
    """The GNN's heads, fed by a backbone's latents instead of its message passing."""

    def __init__(self, backbone: TemporalJEPA, frozen: bool):
        super().__init__()
        self.backbone, self.frozen = backbone, frozen
        d = backbone.config["hidden"]
        n_h = len(HORIZONS_DAYS)
        self.node = nn.ModuleList(mlp(2 * d, d, len(NODE_HEADS)) for _ in range(n_h))
        self.link = nn.ModuleList(mlp(2 * d + 1, d, 1) for _ in range(n_h))
        self.link_prior = nn.Parameter(torch.tensor(4.0))  # as in gnn.Forecaster
        if frozen:
            for p in backbone.parameters():
                p.requires_grad_(False)

    def latents(self, now: dict, past: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
        bb = self.backbone
        with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
            z_now, z_past = bb._encode(bb.context, now), bb._encode(bb.context, past)
            gl = bb.context.embed_glob(now["glob"])[:, None, :]
            adj = bb.adjacency_now(now)
            preds = []
            for k in range(len(HORIZONS_DAYS)):
                kk = torch.full((len(z_now),), k, dtype=torch.long)
                preds.append(bb.predictor(z_now, z_past, gl, adj, kk))
        return z_now, preds

    def forward(self, now: dict, past: dict, link_now: torch.Tensor):
        """Node outputs (B, N, heads, horizons) and link logits (B, E, horizons), like the GNN."""
        z_now, preds = self.latents(now, past)
        pairs = self.backbone.pairs
        sign = (2 * link_now - 1)[..., None]
        node, link = [], []
        for k, zk in enumerate(preds):
            h = torch.cat([zk, z_now], -1)
            node.append(self.node[k](h))
            hi, hj = zk[:, pairs[:, 0]], zk[:, pairs[:, 1]]
            link.append(self.link[k](torch.cat([hi + hj, (hi - hj).abs(), sign], -1))[..., 0])
        return torch.stack(node, -1), torch.stack(link, -1) + self.link_prior * sign


def attach_pairs(model: TemporalJEPA) -> TemporalJEPA:
    from worldmodel.gnn import adjacency
    from worldmodel.graph import adjacent_pairs

    c = model.config
    model.pairs = torch.as_tensor(adjacent_pairs(c["rows"], c["cols"]))
    model.adjacency_now = lambda s: adjacency(model.mask, s["elev"], s["on"])
    return model


def targets(samples: dict, meta: dict) -> dict[str, torch.Tensor]:
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    out = {f"y_{n}": t(samples[f"y_{n}"] / np.array(meta["target_scale"][n])) for n in CONTINUOUS}
    out["y_mortality"] = t(samples["y_mortality"])
    out["y_links"] = t(samples["y_links"])
    out["link_now"] = t(samples["link_now"])
    return out


def predict(model: Heads, w: Worlds, tt: dict, meta: dict, batch: int = 256) -> dict:
    model.eval()
    chunks = {h: [] for h in (*NODE_HEADS, "links")}
    with torch.no_grad():
        for i in range(0, len(w.t), batch):
            r = w.t[i : i + batch]
            node, link = model(w.at(r), w.at(r, -TREND_DAYS), tt["link_now"][i : i + batch])
            for k, name in enumerate(NODE_HEADS):
                v = node[:, :, k, :]
                v = (torch.sigmoid(v) if name == "mortality"
                     else v * torch.tensor(meta["target_scale"][name]))  # fmt: skip
                chunks[name].append(v.numpy())
            chunks["links"].append(torch.sigmoid(link).numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in chunks.items()}


def train_heads(model: Heads, worlds: dict, samples: dict, meta: dict, *, epochs: int,
                lr: float, batch: int, seed: int, log=print) -> tuple[list, dict]:  # fmt: skip
    """train.py's loop: AdamW + cosine, keep the epoch with the best val mean 7 d + 30 d skill."""
    gen = torch.Generator().manual_seed(seed)
    tt = {s: targets(samples[s], meta) for s in samples}
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.") and p.requires_grad]  # fmt: skip
    groups = [{"params": head_params, "lr": lr}]
    if not model.frozen:
        groups.append({"params": list(model.backbone.parameters()), "lr": lr / 10})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w, n = worlds["train"], len(worlds["train"].t)
    best, best_state, curve, t0 = -1e9, None, [], time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            rows = w.t[idx]
            yb = {k: v[idx] for k, v in tt["train"].items()}
            node, link = model(w.at(rows), w.at(rows, -TREND_DAYS), yb["link_now"])
            loss = loss_fn(node, link, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            if not model.frozen:
                model.backbone.ema_update(0.999)
            total += float(loss.detach()) * len(idx)
        sched.step()
        val = scores(predict(model, worlds["val"], tt["val"], meta), samples["val"])
        target = (val["mean_skill@7d"] + val["mean_skill@30d"]) / 2
        skills = {f"val_skill@{h}d": val[f"mean_skill@{h}d"] for h in HORIZONS_DAYS}
        curve.append({"epoch": epoch, "train_loss": total / n,
                      "seconds": round(time.time() - t0), **skills})  # fmt: skip
        log(f"epoch {epoch:2d} loss {total / n:.4f} val skill "
            + " ".join(f"{h}d {val[f'mean_skill@{h}d']:+.3f}" for h in HORIZONS_DAYS)
            + f"  ({time.time() - t0:.0f}s)")  # fmt: skip
        if target > best:
            best, best_state = target, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    report = {s: with_ex_mortality(scores(predict(model, worlds[s], tt[s], meta), samples[s]))
              for s in ("val", "test")}  # fmt: skip
    return curve, report


def load_backbone(name: str) -> tuple[TemporalJEPA, dict]:
    if name == "temporal":
        model, meta = load_temporal(TEMPORAL_PATH)
    elif name == "generative":
        ckpt = torch.load(GENERATIVE_PATH, weights_only=True)
        cfg = {k: v for k, v in ckpt["config"].items() if k != "generative"}
        model = GenerativeForecaster(**cfg)
        missing, unexpected = model.load_state_dict(
            {k: v.float() for k, v in ckpt["state_dict"].items()}, strict=False)  # fmt: skip
        assert set(missing) == {"mask"} and not unexpected, (missing, unexpected)
        meta = ckpt["meta"]
    elif name == "scratch":
        _, meta = load_temporal(TEMPORAL_PATH)  # only for its input normalization
        torch.manual_seed(0)
        model = TemporalJEPA(n_in=len(meta["node_mean"]), n_glob=len(meta["glob_mean"]))
    else:
        raise ValueError(name)
    return attach_pairs(model.eval()), meta


def pretrain_generative(data: Path, epochs: int, seed: int, out: Path = GENERATIVE_PATH) -> None:
    """The generative baseline: same init, data, stride, epochs and optimizer as step 2."""
    from worldmodel.train_jepa import load_model as load_static

    torch.manual_seed(seed)
    t0 = time.time()
    _, meta = load_static(STATIC_PATH)
    w = Worlds(SPLITS["train"], data, meta, stride=2)
    model = GenerativeForecaster(n_in=w.flat["nodes"].shape[-1], n_glob=w.flat["glob"].shape[-1])
    model.init_encoders(torch.load(STATIC_PATH, weights_only=True)["state_dict"])
    monitor = CollapseMonitor()
    curve = train(model, w, epochs=epochs, batch=64, lr=5e-4, var_weight=0.0, seed=seed,
                  log=lambda s: print(s, flush=True), monitor=monitor)  # fmt: skip
    train_s = time.time() - t0
    del w
    strides = {"train": 2, "val": 1, "test": 1}
    worlds = {s: Worlds(SPLITS[s], data, meta, strides[s]) for s in SPLITS}
    samples = {s: samples_for(w, strides[s]) for s, w in worlds.items()}
    feats = {s: predicted(model, w, w.t) for s, w in worlds.items()}
    probes = with_ex_mortality(scores(forecast_probes(feats, samples), samples["test"]))
    print("generative probe skill:", {k: round(v, 3) for k, v in probes.items()
                                      if k.startswith("mean")}, flush=True)  # fmt: skip
    torch.save({"config": model.config, "meta": meta,
                "state_dict": {k: v.half() if v.is_floating_point() else v
                               for k, v in model.state_dict().items() if k != "mask"}},
               out)  # fmt: skip
    out.with_name(out.stem + "_metrics.json").write_text(json.dumps({
        "objective": "smooth-L1 on normalized patch features at t+k (input space)",
        "init_from": str(STATIC_PATH), "epochs": epochs, "train_seconds": round(train_s, 1),
        "curve": curve, "collapse_monitor": monitor.report(), "probes_test": probes,
        "weights": "fp16 checkpoint; metrics from the full-precision model",
    }, indent=1))  # fmt: skip
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


RUNS = [(b, m) for b in ("temporal", "generative", "scratch") for m in ("frozen", "finetune")]


def summarize(path: Path = HERE / "gate_c_metrics.json", extra: Path | None = None) -> dict:
    """Collect every gate_c_<backbone>_<mode>.json into one table, with the GNN's numbers.

    `extra` is a directory holding a second sweep's run files (e.g. --train-stride 2);
    its table is kept alongside as a reading of run-to-run variation.
    """
    gnn = with_ex_mortality(json.loads(GNN_METRICS.read_text())["test"])
    keys = [f"mean_skill{x}@{h}d" for x in ("", "_ex_mortality") for h in HORIZONS_DAYS]
    table_extra: dict = {}
    table = {"gnn (Phase 4, supervised)": {k: gnn[k] for k in keys}}
    runs = {}
    for b, m in RUNS:
        f = HERE / f"gate_c_{b}_{m}.json"
        if extra is not None:
            g = extra / f.name
            if g.exists():
                r2 = json.loads(g.read_text())
                key = f"train_stride_{r2['args']['train_stride']}"
                second = table_extra.setdefault(key, {})
                second[f"{b}/{m}"] = {k: r2["test"][k] for k in keys}
        if f.exists():
            r = json.loads(f.read_text())
            runs[f"{b}/{m}"] = r
            table[f"{b}/{m}"] = {k: r["test"][k] for k in keys}
    per_head = {name: {k: v["skill"] for k, v in r["test"].items() if isinstance(v, dict)}
                for name, r in runs.items()}  # fmt: skip
    per_head["gnn (Phase 4, supervised)"] = {k: v["skill"] for k, v in gnn.items()
                                            if isinstance(v, dict)}  # fmt: skip
    jepa = [n for n in table if n.startswith("temporal/")]
    gate = {n: {f"{h}d": table[n][f"mean_skill@{h}d"] > gnn[f"mean_skill@{h}d"] for h in (7, 30)}
            for n in jepa}  # fmt: skip
    out = {"metric": "test mean skill vs persistence (targets.scores), worlds 24-27",
           "gate_c": "a JEPA backbone with heads beats the GNN at 7 and 30 d",
           "gate_c_passed": any(all(v.values()) for v in gate.values()), "jepa_vs_gnn": gate,
           "table": table, "per_head": per_head,
           "seconds": {n: r["seconds"] for n, r in runs.items()},
           "epochs": {n: r["args"]["epochs"] for n, r in runs.items()},
           "train_stride": {n: r["args"]["train_stride"] for n, r in runs.items()},
           "other_sweeps": table_extra}  # fmt: skip
    path.write_text(json.dumps(out, indent=1))
    w = max(len(n) for n in table) + 2
    label = lambda k: k.replace("mean_skill", "").replace("_ex_mortality", "ex-mort ")  # noqa: E731
    print(f"{'':<{w}}" + "".join(f"{label(k):>15}" for k in keys))
    for n, row in table.items():
        print(f"{n:<{w}}" + "".join(f"{row[k]:15.3f}" for k in keys))
    print("Gate C passed:", out["gate_c_passed"], gate)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbone", choices=("temporal", "generative", "scratch"), default="temporal")
    ap.add_argument("--mode", choices=("frozen", "finetune"), default="frozen")
    ap.add_argument("--pretrain", action="store_true", help="train the generative backbone")
    ap.add_argument("--summary", action="store_true", help="collect runs into gate_c_metrics.json")
    ap.add_argument("--extra", type=Path, default=None, help="--summary: a second sweep's dir")
    ap.add_argument("--data", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=None, help="default 30 frozen, 12 finetune")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-stride", type=int, default=1, help="1 = every day, as the GNN")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.pretrain:
        pretrain_generative(args.data, epochs=10, seed=args.seed)
        return
    if args.summary:
        summarize(extra=args.extra)
        return
    if args.epochs is None:
        args.epochs = 30 if args.mode == "frozen" else 12
    torch.manual_seed(args.seed)
    t0 = time.time()
    backbone, meta = load_backbone(args.backbone)
    strides = {"train": args.train_stride, "val": 1, "test": 1}
    worlds = {s: Worlds(SPLITS[s], args.data, meta, strides[s]) for s in SPLITS}
    samples = {s: samples_for(w, strides[s]) for s, w in worlds.items()}
    tmeta = gnn_normalizers(samples["train"])
    model = Heads(backbone, frozen=args.mode == "frozen")
    n_head = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("backbone."))
    print(f"{args.backbone}/{args.mode}: {len(worlds['train'].t)} training samples, "
          f"{n_head} head parameters", flush=True)  # fmt: skip
    curve, report = train_heads(model, worlds, samples, tmeta, epochs=args.epochs, lr=args.lr,
                                batch=args.batch, seed=args.seed,
                                log=lambda s: print(s, flush=True))  # fmt: skip
    gnn = with_ex_mortality(json.loads(GNN_METRICS.read_text())["test"])
    test = report["test"]
    gate = {f"{h}d": test[f"mean_skill@{h}d"] > gnn[f"mean_skill@{h}d"] for h in (7, 30)}
    print("test:", " ".join(f"{k} {test[k]:+.3f} (gnn {gnn[k]:+.3f})" for k in test
                            if k.startswith("mean")))  # fmt: skip
    print("Gate C (beats GNN at 7 and 30 d):", gate)
    out = HERE / f"gate_c_{args.backbone}_{args.mode}.json"
    out.write_text(json.dumps({
        "backbone": args.backbone, "mode": args.mode, "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_samples": int(len(worlds["train"].t)), "head_parameters": int(n_head),
        "seconds": round(time.time() - t0, 1), "curve": curve, **report,
        "gnn_test": {k: v for k, v in gnn.items() if k.startswith("mean")},
        "beats_gnn": gate,
    }, indent=1))  # fmt: skip
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
