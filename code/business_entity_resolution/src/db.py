"""Shared DuckDB connection factory.

Centralises conservative memory/thread/spill settings so every Step 2 script
behaves the same way on this machine (16 GB RAM, often with several GB already
used by other running applications): cap DuckDB's memory budget explicitly and
point its spill-to-disk temp files at a scratch directory (gitignored) instead
of letting it default to the working directory or unbounded memory use.

Uses an on-disk database file rather than ``:memory:``: DuckDB's buffer
manager can then evict cold table pages to that file under memory pressure
instead of requiring every intermediate table to stay fully resident, which
is what caused real OutOfMemoryExceptions (and once, a native-level segfault
under combined system memory pressure) when blocking.py held several large
intermediate tables at once in pure in-memory mode.

The scratch directory defaults to ``D:/ber_scratch`` rather than under the
repo on ``C:`` -- this machine's C: drive is a laptop OS drive running at
90%+ full with as little as ~14 GB free, which is not enough headroom for
blocking.py's temp spill files (observed growing past 14 GB on the full
test-candidate pass) and caused a real disk-full IOException. D: has ~130 GB
free. Falls back to ``data/`` under the repo if D: isn't present (e.g. on a
different machine or CI).
"""

from pathlib import Path

import duckdb

from config import DATA_DIR

SCRATCH_DIR = Path("D:/ber_scratch") if Path("D:/").exists() else DATA_DIR


def connect(memory_limit_gb=4, threads=4):
    """Return a DuckDB connection configured with a memory cap and disk spill."""
    tmp_dir = SCRATCH_DIR / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = SCRATCH_DIR / "warehouse.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    return con
