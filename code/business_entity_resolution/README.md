# Business Entity Resolution

End-to-end pipeline: dictionary mining, normalization, blocking (candidate
generation), pair features, a two-stage LightGBM model with a calibrated
expected-F0.5 decision layer, and a validated test submission.

The same code runs on Linux, macOS and Windows and sizes itself to the machine:
RAM budget, worker processes, DuckDB memory/threads, blocking caps, batch sizes
and training-set size all come from `src/resources.py` (see "Machine sizing").

## Setup

The dataset (`dataset/train/*.tsv`, `dataset/test/*.tsv`) must be in the repo
root; it is gitignored, so copy it onto each new machine.

**Ubuntu / Linux**

```bash
bash setup_ubuntu.sh          # creates .venv, installs pinned deps, checks dataset + prints the machine profile
tmux new -s ber               # optional: keeps the run alive if SSH drops
.venv/bin/python code/business_entity_resolution/run_pipeline.py
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r code\business_entity_resolution\requirements.txt
.venv\Scripts\python.exe code\business_entity_resolution\run_pipeline.py
```

Python >= 3.10. Dependencies are pinned in `requirements.txt` (pandas, numpy,
scikit-learn, rapidfuzz, lightgbm, unidecode, pyarrow, duckdb, psutil).

## Running

```bash
python code/business_entity_resolution/run_pipeline.py              # all steps
python code/business_entity_resolution/run_pipeline.py --list       # show steps
python code/business_entity_resolution/run_pipeline.py --from features_train   # resume
python code/business_entity_resolution/run_pipeline.py --only train_model
```

Steps, in order: `ingest` → `folds` → `evaluate` (metric unit tests) →
`mine_dicts` → `normalize` → `blocking_train` → `blocking_test` →
`features_train` → `features_test` → `train_model` → `oof_report`.
`ingest`/`folds` are skipped when their outputs exist (`--force` reruns them).

Progress: every step streams to the console and to `output/logs/<step>.log`
(percent done, elapsed, ETA); `output/logs/pipeline_status.json` holds the
current step and per-step timings; each step ends with a
`[pipeline] STEP_DONE` / `STEP_FAILED` line. A failed step prints the exact
`--from` command to resume.

Outputs: `output/matching_results.tsv` + `output/candidate_pairs.tsv` (the
submission, validated by `utils/validate_submission.py` at the end of
`train_model`), `output/MODEL_REPORT.md` (OOF score, decision rule, LOCO,
feature importance), `output/BLOCKING_REPORT_train_only.md` (blocking recall),
a row in `output/results_log.csv`, and `output/worst20_oof_entities.txt`.

## Machine sizing

`src/resources.py` reads total/free RAM (respecting container cgroup limits),
usable cores (respecting CPU affinity) and scratch-disk space at startup, and
derives everything from a RAM budget of `min(80% of total, free - 1 GB)`, so a
run stays out of OOM even with other applications open. Printed at the top of
every step as `[resources] ...`.

| Setting | 16 GB laptop, ~3 GB free | 64 GB+ server |
|---|---|---|
| DuckDB memory (blocking) | ~1.7 GB | ~36 GB |
| feature / normalize worker processes | 2 / 7 | all cores |
| blocking caps (normal / strict keys) | 50 / 15 | 150 / 40 |
| candidates kept per S1 (top-K) | 40 | 50 |
| training candidate rows | ~4M | all (~80M) |

Bigger caps and top-K mean more recall; a bigger training set means a better
model. Override anything with environment variables:

| Variable | Meaning |
|---|---|
| `BER_SCRATCH_DIR` | DuckDB warehouse + spill dir (needs ~100 GB free; default `data/scratch`, or `D:/ber_scratch` on the Windows laptop) |
| `BER_RAM_BUDGET_GB` | total RAM the pipeline may use |
| `BER_CORES` | cores to use |
| `BER_CAP_DEFAULT`, `BER_CAP_STRICT` | blocking block-size caps |
| `BER_TOP_K` | candidates kept per S1 |
| `BER_S1_BATCH` | S1 rows per blocking batch |
| `BER_TRAIN_MAX_ROWS` | max training candidate rows |
| `BER_WORKERS` | worker processes |

Pin them when you need a run to be exactly reproducible across machines.
LightGBM uses the GPU automatically if the installed build supports it (the
PyPI wheel is CPU-only on Windows) and falls back to CPU otherwise.

## Layout

```
code/business_entity_resolution/
├── run_pipeline.py           # one command for the whole pipeline (any OS)
├── requirements.txt
└── src/
    ├── resources.py          # machine profile: RAM/cores/disk -> every size knob
    ├── config.py             # all data paths, relative to the repo root
    ├── db.py                 # DuckDB connection (profile-sized memory/threads, disk spill)
    ├── perf.py               # stage timing, progress/ETA lines
    ├── io_utils.py           # TSV/id-list helpers, f05 metric
    ├── ingest.py             # raw TSV -> parquet
    ├── folds.py              # 300k recall sample + 5-fold assignment over all train S1
    ├── evaluate.py           # macro F0.5 scorer (+ vectorized version) and its unit tests
    ├── mine_dicts.py         # dictionaries mined from train pairs (native-script maps,
    │                         #   native state aliases, states, abbreviations, legal forms...)
    ├── normalize.py          # name/address normalization
    ├── blocking.py           # IDF, key families K1-K7, candidates for ALL train + test S1, recall
    ├── features.py           # pair feature functions
    ├── features_candidates.py# features for every candidate pair (+ competition features)
    ├── train_model.py        # two-stage LightGBM, calibration, decision layer, submission
    ├── oof_report.py         # worst-20 OOF entities
    ├── make_package.py       # builds the submission zip
    └── (Step 1 EDA / diagnostics: first_look, noise_mining, miss_diagnosis, translit_compare, ...)
```

## Method summary

- **Normalization**: NFKC; native-script tokens of every Indic script mapped to
  Latin via mined dictionaries (fallback unidecode) -- the tokenizer keeps Indic
  combining vowel signs attached (plain `\w` split "शक्ति" into "शक"+"त");
  state detection prefers full state names over 2-letter codes and the last
  code over the first; native-script state/city aliases fill in states whose
  transliteration never matches the Latin name.
- **Blocking**: key families K1 (number+address word), K2/K5 (rare name /
  skeleton token + state), K3/K4 (space-less name, wrapper-alt name), K6 (rare
  name-token pair), K7 (house number with one digit dropped: 1951<->195,
  A-192<->A-92); block-size caps; cheap score; top-K per S1. Candidates are
  generated for every train S1 so competition features match test.
- **Model**: stage 1 LightGBM on ~40 pair features; stage 2 adds how the pair's
  stage-1 score compares with the S1's other candidates and with the other S1
  records competing for the same S2/S3 record; cross-fitted isotonic
  calibration.
- **Decision**: per S1, the subset (possibly empty) with the highest expected
  F0.5 under the calibrated probabilities (exact Poisson-binomial), competing
  against threshold / exclusivity / relative-margin rules; the best by OOF
  macro F0.5 over every training S1 (blocking misses count) is used.

## Packaging the final submission

```bash
python code/business_entity_resolution/src/make_package.py
```

Writes `<TEAM_NAME>_submission.zip` (edit `TEAM_NAME` in `make_package.py`
first) with both output TSVs, this code folder and `Documentation_template.md`.
