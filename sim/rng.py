"""Seeded RNG, one independent stream per system.

Each system gets its own generator derived from (master seed, system name),
so adding a random draw to one system never shifts another system's numbers.
Names are hashed with CRC32, which is stable across processes (unlike hash()).
"""

import zlib

import numpy as np

SYSTEMS = ("init", "weather", "plants")


def stream(seed: int, system: str) -> np.random.Generator:
    key = zlib.crc32(system.encode("utf-8"))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, key])))


def make_streams(seed: int) -> dict[str, np.random.Generator]:
    return {name: stream(seed, name) for name in SYSTEMS}
