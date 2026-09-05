"""Shape contract for the opt-in pilot; these are safety limits, not tuning."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    source_rows: int
    routes: int
    experts: int
    k: int
    n: int
    group_size: int
    bits: int
    dtype: str

    def validate(self) -> None:
        for name in ("source_rows", "routes", "experts", "k", "n"):
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 < value < 2**31:
                raise ValueError(f"{name} must be a positive signed-32-bit integer")
        if self.routes < 8:
            raise ValueError("At least 8 routes are required by the JIT pointer ABI")
        if self.dtype not in ("float16", "bfloat16"):
            raise ValueError("The pilot supports only float16 and bfloat16")
        if self.bits not in (4, 8) or self.group_size not in (64, 128):
            raise ValueError("The pilot supports affine 4/8-bit, group size 64/128")
        if self.k % 64 or self.n % 64 or self.k % self.group_size:
            raise ValueError("K/N must be multiples of 64 and K of group_size")
        if self.k * self.n * self.bits // 8 >= 2**31:
            raise ValueError("One expert's packed matrix must be smaller than 2 GiB")
        if self.routes > 32768 and self.routes % 64:
            raise ValueError("Large ragged route counts are excluded from this pilot")

    @property
    def bm(self) -> int:
        return 32 if self.routes // self.experts < 64 else 64

    @property
    def eliminated_gather_bytes(self) -> int:
        return self.routes * self.k * 2


def compatible_device(info: dict, macos_version: str) -> bool:
    """Mirror the pinned NAX gate, additionally limiting the pilot to M5+."""
    parts = macos_version.split(".")
    try:
        version = tuple(int(p) for p in parts[:2])
    except ValueError:
        return False
    if len(version) == 1:
        version += (0,)
    if version < (26, 2):
        return False
    name = str(info.get("device_name", info.get("name", "")))
    chip = re.search(r"\bApple M(\d+)\b", name)
    arch = re.fullmatch(
        r"applegpu_g(\d+)([A-Za-z]+)", str(info.get("architecture", ""))
    )
    if not chip or not arch or int(chip.group(1)) < 5:
        return False
    minimum = 18 if arch.group(2).endswith("p") else 17
    return int(arch.group(1)) >= minimum
