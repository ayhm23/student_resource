"""Run the whole pipeline end to end on any machine (Linux, macOS, Windows).

    python code/business_entity_resolution/run_pipeline.py            # everything
    python code/business_entity_resolution/run_pipeline.py --from normalize
    python code/business_entity_resolution/run_pipeline.py --only train_model
    python code/business_entity_resolution/run_pipeline.py --list

Each step runs as its own process with the same interpreter (the venv's), so
nothing depends on ``.venv/Scripts`` vs ``.venv/bin``. Output is streamed to the
console and to ``output/logs/<step>.log``; ``output/logs/pipeline_status.json``
always holds the current step, per-step timings and overall state, and every
step ends with one ``[pipeline] STEP_DONE`` / ``STEP_FAILED`` line (easy to grep
or to watch remotely). Memory/CPU sizing is automatic (``src/resources.py``);
override with ``BER_*`` environment variables if needed.

``ingest`` and ``folds`` are skipped when their outputs already exist (they are
deterministic and slow to redo for no gain); pass ``--force`` to rerun them.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
ROOT = HERE.parents[1]
sys.path.insert(0, str(SRC))

from config import DATASET_DIR, FOLDS_PARQUET, RAW_DIR, TRAIN_SAMPLE_S1_PARQUET  # noqa: E402
from perf import fmt_hms  # noqa: E402
from resources import describe  # noqa: E402

LOG_DIR = ROOT / "output" / "logs"
STATUS_PATH = LOG_DIR / "pipeline_status.json"

# (name, script, args, relative cost weight for the overall ETA, skip-if-exists outputs)
STEPS = [
    ("ingest", "ingest.py", [], 1, [RAW_DIR / "train_gt_long.parquet", RAW_DIR / "test_S3.parquet"]),
    ("folds", "folds.py", [], 1, [FOLDS_PARQUET, TRAIN_SAMPLE_S1_PARQUET]),
    ("evaluate", "evaluate.py", [], 1, None),
    ("mine_dicts", "mine_dicts.py", [], 8, None),
    ("normalize", "normalize.py", [], 14, None),
    ("blocking_train", "blocking.py", ["train-only"], 90, None),
    ("blocking_test", "blocking.py", ["test-only"], 60, None),
    ("features_train", "features_candidates.py", ["train"], 80, None),
    ("features_test", "features_candidates.py", ["test"], 65, None),
    ("train_model", "train_model.py", [], 120, None),
    ("oof_report", "oof_report.py", [], 2, None),
]
STEP_NAMES = [s[0] for s in STEPS]


def write_status(status):
    status["updated"] = datetime.now().isoformat(timespec="seconds")
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_step(name, script, args):
    """Run one step, teeing its output to the console and its log file. Returns the exit code."""
    log_path = LOG_DIR / f"{name}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(SRC / script), *args]
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(f"[{name}] {line}")
            log.write(line)
            log.flush()
        return proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=STEP_NAMES, help="start at this step")
    ap.add_argument("--to", dest="end", choices=STEP_NAMES, help="stop after this step")
    ap.add_argument("--only", choices=STEP_NAMES, help="run just this step")
    ap.add_argument("--force", action="store_true", help="rerun ingest/folds even if their outputs exist")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    opts = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    if opts.list:
        for name, script, args, weight, _ in STEPS:
            print(f"  {name:15s} {script} {' '.join(args)}")
        return

    if not (DATASET_DIR / "train" / "train_source1.tsv").exists():
        raise SystemExit(f"Dataset not found under {DATASET_DIR} -- copy the challenge dataset/ folder "
                         f"(train/ and test/ TSVs) into the repo root first.")

    names = STEP_NAMES
    if opts.only:
        names = [opts.only]
    else:
        lo = STEP_NAMES.index(opts.start) if opts.start else 0
        hi = STEP_NAMES.index(opts.end) + 1 if opts.end else len(STEP_NAMES)
        names = STEP_NAMES[lo:hi]
    steps = [s for s in STEPS if s[0] in names]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(describe())
    total_weight = sum(s[3] for s in steps)
    status = {"state": "running", "started": datetime.now().isoformat(timespec="seconds"),
              "steps": {s[0]: {"state": "pending"} for s in steps}, "current": None}
    write_status(status)

    t_run = time.perf_counter()
    done_weight = 0
    for i, (name, script, args, weight, outputs) in enumerate(steps, 1):
        if outputs and not opts.force and all(Path(p).exists() for p in outputs):
            print(f"[pipeline] STEP_SKIPPED {name} (outputs exist; --force to rerun)")
            status["steps"][name] = {"state": "skipped"}
            done_weight += weight
            continue
        status["current"] = name
        status["steps"][name] = {"state": "running", "started": datetime.now().isoformat(timespec="seconds")}
        write_status(status)
        print(f"[pipeline] STEP_START {name} ({i}/{len(steps)}) -- log: {LOG_DIR / (name + '.log')}")
        t0 = time.perf_counter()
        code = run_step(name, script, args)
        dt = time.perf_counter() - t0
        if code != 0:
            status["steps"][name] = {"state": "failed", "exit_code": code, "seconds": round(dt)}
            status["state"] = "failed"
            write_status(status)
            print(f"[pipeline] STEP_FAILED {name} exit={code} after {fmt_hms(dt)} -- fix it, then resume with:\n"
                  f"  {Path(sys.executable).name} {Path(__file__).relative_to(ROOT)} --from {name}")
            raise SystemExit(code)
        done_weight += weight
        elapsed = time.perf_counter() - t_run
        frac = done_weight / total_weight if total_weight else 1.0
        eta = elapsed / frac * (1 - frac) if frac > 0 else 0
        status["steps"][name] = {"state": "done", "seconds": round(dt)}
        status["progress_pct"] = round(100 * frac, 1)
        write_status(status)
        print(f"[pipeline] STEP_DONE {name} in {fmt_hms(dt)} -- {i}/{len(steps)} steps, "
              f"~{100 * frac:.0f}% of total work, elapsed {fmt_hms(elapsed)}, rough ETA {fmt_hms(eta)}")

    status["state"] = "done"
    status["current"] = None
    write_status(status)
    print(f"[pipeline] ALL_DONE in {fmt_hms(time.perf_counter() - t_run)} -- submission: "
          f"output/matching_results.tsv + output/candidate_pairs.tsv; report: output/MODEL_REPORT.md")


if __name__ == "__main__":
    main()
