"""Shared helpers for the kernel parity tests."""

import math
import struct

from gradtuity.cuda_mem import (
    cuda_malloc,
    cuda_memcpy_dtoh,
    cuda_memcpy_htod,
    cuda_memset,
)


def alloc_floats(values: list[float]) -> int:
    n = len(values)
    p = cuda_malloc(n * 4)
    cuda_memcpy_htod(p, struct.pack(f"{n}f", *values))
    return p


def alloc_zeros(n: int) -> int:
    p = cuda_malloc(n * 4)
    cuda_memset(p, 0, n * 4)
    return p


def read_floats(p: int, n: int) -> list[float]:
    return list(struct.unpack(f"{n}f", cuda_memcpy_dtoh(p, n * 4)))


def close_enough(a: list[float], b: list[float], rtol: float, atol: float) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if math.isnan(x) and math.isnan(y):
            continue
        if abs(x - y) > atol + rtol * max(abs(x), abs(y)):
            return False
    return True


def max_abs_diff(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))
