"""Timing utilities: a context-manager ``Timer`` and a ``benchmark`` helper.

All latency measurements in TerraSem use ``benchmark`` so that results are
comparable (same warm-up / iteration protocol).  See BUILD.md §0.4.

Usage::

    from terrasem.utils.timing import Timer, benchmark

    with Timer("projection") as t:
        project_lidar(cloud)
    print(f"projection took {t.elapsed_ms:.1f} ms")

    median_ms, p95_ms = benchmark(lambda: model(x), warmup=20, iters=200)
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable


class Timer:
    """Context manager that measures wall-clock elapsed time.

    Attributes:
        elapsed_ms: Milliseconds elapsed between ``__enter__`` and ``__exit__``.
            Only valid after the context has exited.
        elapsed_s:  Same in seconds.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed_ms: float = 0.0
        self.elapsed_s: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_s = time.perf_counter() - self._start
        self.elapsed_ms = self.elapsed_s * 1_000.0

    def __repr__(self) -> str:  # pragma: no cover
        return f"Timer({self.name!r}, elapsed={self.elapsed_ms:.3f} ms)"


def benchmark(
    fn: Callable[[], object],
    warmup: int = 20,
    iters: int = 200,
) -> tuple[float, float]:
    """Benchmark *fn* and return (median_ms, p95_ms).

    Per BUILD.md §0.4:  latency = median over ≥200 warm iterations after
    ≥20 warm-up iterations.

    Args:
        fn:     Zero-argument callable to benchmark.
        warmup: Number of warm-up calls (results discarded).
        iters:  Number of timed calls.

    Returns:
        ``(median_ms, p95_ms)`` — both as floats in milliseconds.

    Example::

        median_ms, p95_ms = benchmark(lambda: model(dummy), warmup=20, iters=200)
        print(f"median {median_ms:.1f} ms  p95 {p95_ms:.1f} ms")
    """
    for _ in range(warmup):
        fn()

    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1_000.0)

    sorted_times = sorted(times_ms)
    median_ms = statistics.median(sorted_times)
    p95_idx = max(0, int(0.95 * len(sorted_times)) - 1)
    p95_ms = sorted_times[p95_idx]

    return median_ms, p95_ms
