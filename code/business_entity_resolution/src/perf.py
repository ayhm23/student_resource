"""Shared performance-reporting helpers: machine/resource summary and a per-stage timer."""

import time
from contextlib import contextmanager

import psutil

from resources import describe, profile


def print_sysinfo():
    """Print the machine's resource profile (RAM budget, cores, scratch, derived sizes).

    Returns ``True`` when the RAM budget is small (<= 12 GB), for callers that
    still want a simple low-RAM flag.
    """
    print(describe())
    return profile().budget_gb <= 12


def fmt_hms(seconds):
    """Format a duration in seconds as H:MM:SS (or M:SS under an hour)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def progress_line(label, done, total, start_time):
    """One-line progress: done/total, %, elapsed, and a linear-extrapolation ETA."""
    elapsed = time.perf_counter() - start_time
    pct = 100.0 * done / total if total else 100.0
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else float("nan")
    eta = fmt_hms(remaining) if remaining == remaining else "?"
    return (f"{label}: {done}/{total} ({pct:.1f}%) | elapsed {fmt_hms(elapsed)} | "
            f"ETA {eta} remaining")


@contextmanager
def stage(name):
    """Time a pipeline stage and report its wall time and process RSS at the end."""
    proc = psutil.Process()
    t0 = time.perf_counter()
    print(f"[stage] {name}: start")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        rss_gb = proc.memory_info().rss / (1024 ** 3)
        print(f"[stage] {name}: done in {fmt_hms(dt)} ({dt:.1f}s), rss={rss_gb:.2f} GB")
