"""Shared DuckDB connection factory.

Uses an on-disk database file (not ``:memory:``) in the scratch directory so
DuckDB's buffer manager can evict cold pages under memory pressure instead of
failing with OutOfMemoryException, and caps its memory/threads from the machine
resource profile (``resources.py``) so every script stays inside the same RAM
budget on any machine.

``role="heavy"`` is for SQL-dominated steps (blocking, IDF): DuckDB gets most of
the RAM budget. ``role="light"`` is for steps that run a Python worker pool next
to DuckDB (normalize, features): DuckDB gets a small slice so the workers fit.

Scratch location: ``BER_SCRATCH_DIR`` if set, else ``D:/ber_scratch`` on the
Windows laptop (its C: drive is nearly full), else ``data/scratch``.
"""

from pathlib import Path

import duckdb

from resources import profile

SCRATCH_DIR = Path(profile().scratch_dir)


def connect(role="heavy", memory_limit_gb=None, threads=None):
    """Return a DuckDB connection to the shared warehouse with a memory cap and disk spill."""
    p = profile()
    if memory_limit_gb is None:
        memory_limit_gb = p.duckdb_heavy_gb if role == "heavy" else p.duckdb_light_gb
    if threads is None:
        threads = p.duckdb_threads
    tmp_dir = SCRATCH_DIR / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(SCRATCH_DIR / "warehouse.duckdb"))
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    # Insertion order is irrelevant everywhere in this pipeline; dropping the
    # guarantee lets DuckDB stream large CREATE TABLE AS / COPY without buffering.
    con.execute("PRAGMA preserve_insertion_order=false")
    return con
