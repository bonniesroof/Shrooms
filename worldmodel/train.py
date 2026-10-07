"""Train the GNN forecaster, compare it with persistence, and export it for numpy inference.

    uv sync --group worldmodel
    uv run python -m worldmodel.dataset --seeds 0-27        # ~15 min on 2 cores
    uv run python -m worldmodel.train                       # ~10 min on 2 cores

Splits are by world (seed), so validation and test worlds are never seen in
training: train 0-19, validation 20-23 (model selection), test 24-27 (reported).
Writes worldmodel/forecaster.npz and worldmodel/forecaster_metrics.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fn

from worldmodel.dataset import OUT, load
from worldmodel.forecast import DEFAULT_PATH
from worldmodel.gnn import Forecaster
from worldmodel.targets import CONTINUOUS, HORIZONS_DAYS, NODE_HEADS, build_samples, scores

SPLITS = {"train": range(0, 20), "val": range(20, 24), "test": range(24, 28)}
HEADS = (*NODE_HEADS, "links")


def collect(seeds, data_dir: Path) -> dict[str, np.ndarray]:
    parts = []
    for seed in seeds:
        parts.append(build_samples(load(data_dir / f"run_{seed}.npz")))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def normalizers(train: dict) -> dict:
    node_std = train["nodes"].reshape(-1, train["nodes"].shape[-1]).std(0)
    glob_std = train["glob"].std(0)
    return {
        "node_mean": train["nodes"].reshape(-1, train["nodes"].shape[-1]).mean(0).tolist(),
        "node_std": np.where(node_std > 1e-8, node_std, 1.0).tolist(),
        "glob_mean": train["glob"].mean(0).tolist(),
        "glob_std": np.where(glob_std > 1e-8, glob_std, 1.0).tolist(),
        "target_scale": {
            n: np.maximum(train[f"y_{n}"].reshape(-1, len(HORIZONS_DAYS)).std(0), 1e-6).tolist()
            for n in CONTINUOUS
        },  # fmt: skip
    }


def tensors(d: dict, meta: dict) -> dict[str, torch.Tensor]:
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    out = {
        "nodes": t((d["nodes"] - meta["node_mean"]) / meta["node_std"]),
        "glob": t((d["glob"] - meta["glob_mean"]) / meta["glob_std"]),
        "on": t(d["on"]), "elev": t(d["elev"]), "link_now": t(d["link_now"]),
    }  # fmt: skip
    for n in CONTINUOUS:
        out[f"y_{n}"] = t(d[f"y_{n}"] / np.array(meta["target_scale"][n]))
    out["y_mortality"] = t(d["y_mortality"])
    out["y_links"] = t(d["y_links"])
    return out


def predict(model, tt: dict, meta: dict, batch: int = 256) -> dict[str, np.ndarray]:
    model.eval()
    chunks = {h: [] for h in HEADS}
    with torch.no_grad():
        for i in range(0, len(tt["nodes"]), batch):
            sl = slice(i, i + batch)
            node, link = model(tt["nodes"][sl], tt["glob"][sl], tt["on"][sl], tt["elev"][sl],
                               tt["link_now"][sl])  # fmt: skip
            for k, name in enumerate(NODE_HEADS):
                v = node[:, :, k, :]
                v = (
                    torch.sigmoid(v)
                    if name == "mortality"
                    else v * torch.tensor(meta["target_scale"][name])
                )
                chunks[name].append(v.numpy())
            chunks["links"].append(torch.sigmoid(link).numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in chunks.items()}


def loss_fn(node, link, yb) -> torch.Tensor:
    loss = 0.0
    for k, name in enumerate(NODE_HEADS):
        out = node[:, :, k, :]
        if name == "mortality":
            loss = loss + 4.0 * fn.l1_loss(torch.sigmoid(out), yb["y_mortality"])
        else:
            loss = loss + fn.l1_loss(out, yb[f"y_{name}"])  # MAE: the metric we report
    return loss + fn.binary_cross_entropy_with_logits(link, yb["y_links"])


def export(model: Forecaster, meta: dict, path: Path) -> None:
    weights = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()
               if k not in ("mask", "pairs")}  # fmt: skip
    np.savez_compressed(path, meta=json.dumps({**meta, "config": model.config}), **weights)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()
    torch.manual_seed(0)

    t0 = time.time()
    data = {k: collect(v, args.data) for k, v in SPLITS.items()}
    meta = normalizers(data["train"])
    tt = {k: tensors(v, meta) for k, v in data.items()}
    print("samples: " + ", ".join(f"{k} {len(v['day'])}" for k, v in data.items())
          + f" ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip

    model = Forecaster(n_in=tt["train"]["nodes"].shape[-1], n_glob=tt["train"]["glob"].shape[-1],
                       hidden=args.hidden)  # fmt: skip
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = len(tt["train"]["nodes"])
    best, best_state, curve = -1e9, None, []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            yb = {k: v[idx] for k, v in tt["train"].items()}
            node, link = model(yb["nodes"], yb["glob"], yb["on"], yb["elev"], yb["link_now"])
            loss = loss_fn(node, link, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
        sched.step()
        val = scores(predict(model, tt["val"], meta), data["val"])
        target = (val["mean_skill@7d"] + val["mean_skill@30d"]) / 2
        curve.append({"epoch": epoch, "train_loss": total / n, **{
            f"val_skill@{h}d": val[f"mean_skill@{h}d"] for h in HORIZONS_DAYS}})  # fmt: skip
        print(f"epoch {epoch:2d} loss {total / n:.4f} val skill "
              + " ".join(f"{h}d {val[f'mean_skill@{h}d']:+.3f}" for h in HORIZONS_DAYS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
        if target > best:
            best = target
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    report = {
        split: scores(predict(model, tt[split], meta), data[split]) for split in ("val", "test")
    }
    export(model, meta, args.out)
    metrics_path = args.out.with_name(args.out.stem + "_metrics.json")
    metrics_path.write_text(json.dumps({
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "samples": {k: int(len(v["day"])) for k, v in data.items()},
        "epochs": args.epochs, "hidden": args.hidden, "curve": curve, **report,
    }, indent=1))  # fmt: skip
    print("test skill:", {k: round(v["skill"], 3) for k, v in report["test"].items()
                          if isinstance(v, dict)})  # fmt: skip
    print(f"wrote {args.out} and {metrics_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
