"""Machine resource profile: every memory/CPU/size knob in the pipeline, derived
from the machine the code is actually running on.

The same code has to run on a 16 GB Windows laptop that often has only a few GB
free, and on a large Linux box where the old hard-coded "2 workers, 3 GB DuckDB"
settings would leave most of the machine idle. ``profile()`` sizes everything
from (a) free RAM at startup (the OOM-safety budget), (b) usable cores (honours
taskset/cgroup CPU affinity on Linux), and (c) container memory limits (cgroup
v1/v2), and every value can be pinned with a ``BER_*`` environment variable --
see ``ENV_OVERRIDES`` -- when a run has to be reproduced exactly.

Knobs that change *results* (not just speed): ``cap_default``/``cap_strict``
(blocking block-size caps: bigger caps = more recall, more candidates),
``top_k`` (candidates kept per S1 for the model), and ``train_max_rows`` (how
much of the training set fits in memory). They scale up with RAM; each run
logs the values it used.
"""

import os
import platform
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import shutil

import psutil

from config import DATA_DIR

GB = 1024 ** 3

# Rough steady-state RSS of one feature worker process: normalize dictionaries
# plus the name/address IDF maps (a few million (country, token) entries).
FEATURE_WORKER_GB = 0.5
NORMALIZE_WORKER_GB = 0.25
# Bytes per training cell while the model trains: float32 feature matrix +
# LightGBM's binned copy + per-fold validation slices.
TRAIN_BYTES_PER_CELL = 4 * 1.6
TRAIN_FEATURE_COUNT = 60

ENV_OVERRIDES = {
    "BER_SCRATCH_DIR": "DuckDB warehouse + spill directory (needs tens of GB free)",
    "BER_RAM_BUDGET_GB": "total RAM the pipeline may use (default: min(80% of total, free - 1 GB))",
    "BER_CORES": "CPU cores to use (default: all usable cores)",
    "BER_CAP_DEFAULT": "blocking block-size cap for normal keys",
    "BER_CAP_STRICT": "blocking block-size cap for the leaky no-state/name-pair keys",
    "BER_TOP_K": "candidates kept per S1 after cheap scoring (fed to the model)",
    "BER_S1_BATCH": "S1 rows per blocking batch",
    "BER_TRAIN_MAX_ROWS": "max candidate rows the model trains on",
    "BER_WORKERS": "feature/normalize worker processes",
}


def _env_float(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


def _env_int(name):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else None


def usable_cores():
    """Cores this process may run on (CPU affinity aware on Linux)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _cgroup_memory_limit_bytes():
    """Container memory limit if one is set (psutil reports the host's RAM inside Docker)."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < (1 << 60):
            return int(raw)
    return None


def default_scratch_dir():
    """BER_SCRATCH_DIR, else D:/ber_scratch on the Windows laptop (C: is nearly full), else data/scratch."""
    env = os.environ.get("BER_SCRATCH_DIR")
    if env:
        return Path(env)
    if platform.system() == "Windows" and Path("D:/").exists():
        return Path("D:/ber_scratch")
    return DATA_DIR / "scratch"


@dataclass(frozen=True)
class Profile:
    """Resolved resource settings for this run (see module docstring)."""

    total_ram_gb: float
    avail_ram_gb: float
    budget_gb: float
    cores: int
    physical_cores: int
    scratch_dir: str
    scratch_free_gb: float
    duckdb_heavy_gb: float
    duckdb_light_gb: float
    duckdb_threads: int
    workers_features: int
    workers_normalize: int
    s1_batch_size: int
    cap_default: int
    cap_strict: int
    top_k: int
    train_max_rows: int
    lgb_threads: int
    predict_batch_rows: int


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


@lru_cache(maxsize=1)
def profile():
    """Compute (once per process) the resource profile for this machine."""
    vm = psutil.virtual_memory()
    total = vm.total
    avail = vm.available
    cg = _cgroup_memory_limit_bytes()
    if cg:
        total = min(total, cg)
        avail = min(avail, cg)
    total_gb, avail_gb = total / GB, avail / GB

    budget = _env_float("BER_RAM_BUDGET_GB")
    if budget is None:
        budget = max(1.5, min(0.80 * total_gb, avail_gb - 1.0))

    cores = _env_int("BER_CORES") or usable_cores()
    physical = min(psutil.cpu_count(logical=False) or cores, cores)

    scratch = default_scratch_dir()
    scratch.mkdir(parents=True, exist_ok=True)
    scratch_free = shutil.disk_usage(scratch).free / GB

    duckdb_heavy = round(max(1.0, budget * 0.70), 1)
    duckdb_light = round(_clip(budget * 0.15, 1.0, 8.0), 1)

    workers_features = _env_int("BER_WORKERS") or int(_clip(
        (budget - duckdb_light) // FEATURE_WORKER_GB, 1, cores))
    workers_normalize = _env_int("BER_WORKERS") or int(_clip(
        (budget - 0.5) // NORMALIZE_WORKER_GB, 1, cores))

    if budget < 12:
        caps = (50, 15)
    elif budget < 32:
        caps = (100, 25)
    else:
        caps = (150, 40)
    cap_default = _env_int("BER_CAP_DEFAULT") or caps[0]
    cap_strict = _env_int("BER_CAP_STRICT") or caps[1]
    top_k = _env_int("BER_TOP_K") or (40 if budget < 12 else 50)

    # 75k rows/batch was the measured safe size at 4 GB DuckDB and caps 50/15;
    # scale with DuckDB memory, inversely with the cap (block size drives the
    # join's row explosion).
    s1_batch = _env_int("BER_S1_BATCH") or int(_clip(
        75_000 * (duckdb_heavy / 4.0) * (50.0 / cap_default), 25_000, 1_000_000))

    train_max_rows = _env_int("BER_TRAIN_MAX_ROWS") or int(
        budget * 0.60 * GB / (TRAIN_FEATURE_COUNT * TRAIN_BYTES_PER_CELL))
    predict_batch = int(_clip(budget * 100_000, 200_000, 2_000_000))

    return Profile(
        total_ram_gb=round(total_gb, 1), avail_ram_gb=round(avail_gb, 1), budget_gb=round(budget, 1),
        cores=cores, physical_cores=physical,
        scratch_dir=str(scratch), scratch_free_gb=round(scratch_free, 1),
        duckdb_heavy_gb=duckdb_heavy, duckdb_light_gb=duckdb_light, duckdb_threads=cores,
        workers_features=workers_features, workers_normalize=workers_normalize,
        s1_batch_size=s1_batch, cap_default=cap_default, cap_strict=cap_strict, top_k=top_k,
        train_max_rows=train_max_rows, lgb_threads=physical, predict_batch_rows=predict_batch,
    )


def describe():
    """Human-readable one-block summary of the profile, for logs."""
    p = profile()
    lines = ["[resources] " + ", ".join(f"{k}={v}" for k, v in asdict(p).items())]
    if p.avail_ram_gb < 6:
        lines.append(f"[resources] WARNING: only {p.avail_ram_gb} GB RAM free at startup -- every "
                     f"memory-driven setting above was sized down to stay out of OOM. Close other "
                     f"applications before a long run for a bigger training set and faster blocking.")
    if p.scratch_free_gb < 80:
        lines.append(f"[resources] WARNING: only {p.scratch_free_gb} GB free in {p.scratch_dir}; "
                     f"blocking spill + warehouse can need 60-120 GB. Point BER_SCRATCH_DIR at a "
                     f"bigger disk if a step dies with an out-of-disk error.")
    return "\n".join(lines)
