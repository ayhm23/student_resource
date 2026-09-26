"""Shared performance-reporting helpers: machine sysinfo and a per-stage timer.

Used by every Step 2 script so each pipeline stage prints wall time and peak
process memory, and so each script can decide whether to partition work by
country based on how much RAM is actually available.
"""

import time
from contextlib import contextmanager

import psutil

LOW_RAM_GB_THRESHOLD = 16


def print_sysinfo():
    """Print total RAM, currently available RAM, and CPU core count.

    Returns ``True`` when total RAM is at or below ``LOW_RAM_GB_THRESHOLD`` GB,
    the signal callers use to decide whether to process per-country/in chunks
    instead of holding a full source's data at once.
    """
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    avail_gb = vm.available / (1024 ** 3)
    cores = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    low_ram = total_gb <= LOW_RAM_GB_THRESHOLD
    print(f"[sysinfo] RAM: {total_gb:.1f} GB total, {avail_gb:.1f} GB available | "
          f"CPU: {physical} physical / {cores} logical cores | "
          f"low_ram_mode={'ON' if low_ram else 'off'} (threshold {LOW_RAM_GB_THRESHOLD} GB)")
    return low_ram


@contextmanager
def stage(name):
    """Time a pipeline stage and report its wall time and peak process memory.

    Peak memory is approximated as the process's resident set size (RSS) right
    after the stage finishes, since Windows has no cheap per-block peak-RSS
    counter; it is still useful as a relative, per-stage signal.
    """
    proc = psutil.Process()
    t0 = time.perf_counter()
    print(f"[stage] {name}: start")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        rss_gb = proc.memory_info().rss / (1024 ** 3)
        print(f"[stage] {name}: done in {dt:.1f}s, rss={rss_gb:.2f} GB")
