"""Collapse monitoring for self-supervised encoders (numpy).

A joint-embedding model can cheat by mapping everything to the same point
(complete collapse) or onto a few directions (dimensional collapse): the latent
loss goes to zero and the latents carry nothing. Two cheap statistics catch it
on a batch of embeddings:

    std             per-dimension standard deviation across rows, mean and minimum
    spread          mean std divided by the embedding's RMS: scale-free, so a final
                    LayerNorm can't hide collapse by shrinking everything; complete
                    collapse (every row the same vector) drives it to 0
    effective rank  exp(entropy of the normalized singular values) (RankMe, Garrido
                    et al. 2023); dimensional collapse drives it toward 1

CollapseMonitor records them every epoch for any number of named embeddings and
flags collapse. Every trainer (static, temporal, action JEPA and the generative
baseline) writes monitor.history and monitor.flags into its metrics.
"""

import numpy as np

RANK_FLOOR = 4.0  # effective rank below this (of 64 dims) counts as dimensional collapse
SPREAD_FLOOR = 0.25  # spread below this counts as complete collapse (healthy runs: ~0.7-1)


def collapse_stats(z: np.ndarray) -> dict:
    """Embedding spread and effective rank of (rows, D) embeddings (any leading shape)."""
    z = z.reshape(-1, z.shape[-1]).astype(np.float64)
    zc = z - z.mean(0)
    s = np.linalg.svd(zc, compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    rank = float(np.exp(-(p * np.log(p)).sum())) if len(p) else 0.0
    sd = z.std(0)
    rms = float(np.sqrt(np.mean(z**2)))
    return {"std": float(sd.mean()), "min_std": float(sd.min()),
            "spread": float(sd.mean() / rms) if rms > 0 else 0.0,
            "effective_rank": rank, "dims": int(z.shape[1])}  # fmt: skip


def is_collapsed(stats: dict, rank_floor: float = RANK_FLOOR,
                 spread_floor: float = SPREAD_FLOOR) -> str | None:  # fmt: skip
    """None if healthy, else which kind of collapse."""
    if stats["spread"] < spread_floor:
        return "complete"
    if stats["effective_rank"] < rank_floor:
        return "dimensional"
    return None


class CollapseMonitor:
    """Per-epoch collapse statistics for named embeddings, e.g. target and context latents."""

    def __init__(self, rank_floor: float = RANK_FLOOR, spread_floor: float = SPREAD_FLOOR):
        self.rank_floor, self.spread_floor = rank_floor, spread_floor
        self.history: list[dict] = []
        self.flags: list[dict] = []

    def record(self, epoch: int, **embeddings: np.ndarray) -> dict:
        row = {"epoch": epoch}
        for name, z in embeddings.items():
            st = collapse_stats(z)
            row[name] = st
            kind = is_collapsed(st, self.rank_floor, self.spread_floor)
            if kind:
                flag = {"epoch": epoch, "embedding": name, "kind": kind}
                self.flags.append({**flag, "spread": st["spread"], "rank": st["effective_rank"]})
        self.history.append(row)
        return row

    @property
    def collapsed(self) -> bool:
        return bool(self.flags)

    def line(self) -> str:
        """The latest epoch, one short line for logs."""
        row = self.history[-1]
        parts = [f"{k} std {v['std']:.3f} spread {v['spread']:.2f} rank "
                 f"{v['effective_rank']:.1f}/{v['dims']}"
                 for k, v in row.items() if k != "epoch"]  # fmt: skip
        flag = " COLLAPSED" if any(f["epoch"] == row["epoch"] for f in self.flags) else ""
        return " | ".join(parts) + flag

    def report(self) -> dict:
        return {
            "rank_floor": self.rank_floor, "spread_floor": self.spread_floor,
            "collapsed": self.collapsed, "flags": self.flags, "history": self.history,
        }  # fmt: skip
