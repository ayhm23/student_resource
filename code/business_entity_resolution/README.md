# Business Entity Resolution

Full pipeline: environment setup, EDA, dictionary mining, normalization,
blocking, pair features, and a trained LightGBM model producing a validated
test submission.

## Setup

Run all commands from the `student_resource/` root (this repo's root).

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r code/business_entity_resolution/requirements.txt
```

Dependencies are pinned in `requirements.txt` (pandas, numpy, scikit-learn,
rapidfuzz, lightgbm, unidecode, pyarrow, duckdb, psutil).

**Scratch space**: `code/business_entity_resolution/src/db.py` points
DuckDB's on-disk warehouse and spill files at `D:/ber_scratch` by default
(falls back to `data/` under the repo if `D:` doesn't exist). Blocking and
feature computation need real headroom here — tens of GB of temp spill are
normal at this dataset's scale. If your `C:`/primary drive is tight on
space, make sure whatever drive `SCRATCH_DIR` in `db.py` resolves to has at
least ~50 GB free.

## Layout

```
code/business_entity_resolution/
├── requirements.txt
├── README.md
└── src/
    ├── config.py                 # all data paths, resolved relative to student_resource/
    ├── db.py                     # shared DuckDB connection (memory cap, disk spill/scratch)
    ├── perf.py                   # print_sysinfo() / stage() timing+memory logging
    ├── io_utils.py                # read_tsv, split_ids, write_id_list_tsv, f05 metric
    │
    ├── first_look.py             # Step 1: EDA / data-quality checks (a-i)
    ├── noise_mining.py           # Step 1: name/address abbreviation-substitution mining
    ├── make_empty_submission.py  # Step 1: all-empty first submission
    ├── sanity_check_metric.py    # Step 1: verifies f05 against known baselines
    │
    ├── ingest.py                 # Step 2: raw TSV -> typed parquet (data/raw/)
    ├── mine_dicts.py             # Step 2 Track 2: dictionaries mined from train pairs (data/dicts/)
    ├── normalize.py              # Step 2 Track 3a: name/address normalization (data/norm/)
    ├── blocking.py               # Step 2 Track 3b-f: IDF, blocking keys, recall measurement, test candidates
    ├── evaluate.py               # Step 2 Track 1: macro F0.5 / precision / recall scoring
    ├── folds.py                  # Step 2 Track 1: 300k stratified sample + 5-fold assignment
    ├── features.py               # Step 2 Track 4 Phase 1: pair feature functions + GT-vs-negatives sanity check
    ├── features_candidates.py    # Step 2 Track 4 Phase 2: features on real blocking candidates
    ├── train_model.py            # Step 2 Track 5: LightGBM, decision-rule tuning, LOCO, test submission
    └── oof_report.py             # Step 2 Track 5: saves OOF probabilities, worst-20-entities report
```

## Run order

```bash
# --- Step 1: setup, EDA, all-empty baseline ---
.venv/Scripts/python.exe code/business_entity_resolution/src/first_look.py > output/step1_output.txt
.venv/Scripts/python.exe code/business_entity_resolution/src/noise_mining.py
.venv/Scripts/python.exe code/business_entity_resolution/src/sanity_check_metric.py
.venv/Scripts/python.exe code/business_entity_resolution/src/make_empty_submission.py

# --- Step 2: full pipeline ---
# 1. Ingest raw TSVs to typed parquet (one-time; ~10s)
.venv/Scripts/python.exe code/business_entity_resolution/src/ingest.py

# 2. Track 1: evaluation harness + train sample/folds (independent of everything else)
.venv/Scripts/python.exe code/business_entity_resolution/src/folds.py
.venv/Scripts/python.exe code/business_entity_resolution/src/evaluate.py

# 3. Track 2: mine dictionaries from train ground truth (~7 min)
.venv/Scripts/python.exe code/business_entity_resolution/src/mine_dicts.py

# 4. Track 3a: normalize every record (~14 min)
.venv/Scripts/python.exe code/business_entity_resolution/src/normalize.py

# 5. Track 3b-f: IDF, blocking keys, recall measurement, test candidates (~65 min)
#    Writes output/BLOCKING_REPORT.md, data/cand/test_candidates.parquet,
#    output/candidate_pairs.tsv (blocking's own top-60 view; train_model.py
#    below overwrites this with the model's actual inference-input set).
.venv/Scripts/python.exe code/business_entity_resolution/src/blocking.py

# 6. Track 4 Phase 1: feature functions, unit-tested on GT pairs vs. negatives (~15 min)
.venv/Scripts/python.exe code/business_entity_resolution/src/features.py

# 7. Track 4 Phase 2: features on the real candidate pairs (~16 min train, ~102 min test)
.venv/Scripts/python.exe code/business_entity_resolution/src/features_candidates.py train
.venv/Scripts/python.exe code/business_entity_resolution/src/features_candidates.py test

# 8. Track 5: train, tune the decision rule, LOCO check, predict test, validate (~60 min)
#    Writes output/matching_results.tsv, output/candidate_pairs.tsv (final),
#    appends a row to output/results_log.csv.
.venv/Scripts/python.exe code/business_entity_resolution/src/train_model.py

# 9. Track 5 extra: save OOF probabilities + worst-20-entities error analysis (~11 min)
.venv/Scripts/python.exe code/business_entity_resolution/src/oof_report.py

# 10. Validate the final submission
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

All large source TSVs/parquet are streamed in chunks (DuckDB batches or
`multiprocessing.Pool` over fixed-size row batches, see `CHUNK`/`BATCH_ROWS`
constants in each script), so no script needs to hold a full multi-GB file
in memory at once. `blocking.py` and `train_model.py`'s test-prediction step
additionally partition work by country (and, within the largest countries,
by fixed-size id batches) — this machine has 16 GB RAM and repeatedly hit
memory/disk limits without it; see `output/BLOCKING_REPORT.md`'s
"Engineering notes" section for the full story.

## Timings & peak RAM per stage

Measured on this team's dev machine: **Windows laptop, 16 GB RAM (16.8 GB
as reported by `psutil`), 8 physical / 12 logical CPU cores, CPU only**.
Wall times are cross-referenced from `output/STEP2_REPORT.md`'s
cumulative timing table (authoritative for all Step 2 tracks); peak RSS
figures are from `output/BLOCKING_REPORT.md`'s per-part memory table
where that finer-grained breakdown exists. Where no peak-RAM number was
logged for a stage, it's marked "not logged" rather than guessed.

| Stage | Script | Wall time | Peak RAM |
|---|---|---|---|
| Step 1: EDA + noise mining + baseline | `first_look.py`, `noise_mining.py`, `sanity_check_metric.py`, `make_empty_submission.py` | seconds-to-low-minutes each (see `output/step1_output.txt`) | not logged |
| Ingest raw TSVs → parquet | `ingest.py` | ~10s | not logged |
| Track 1: eval harness + folds/sample | `folds.py`, `evaluate.py` | ~5 min | not logged |
| Track 2: mine dictionaries | `mine_dicts.py` | ~7 min | <0.25 GB |
| Track 3a: normalize | `normalize.py` | ~14 min (final run; ~28 min cumulative across the run + the mid-session state-code-bug rerun) | <0.5 GB/worker (`multiprocessing.Pool(4)`) |
| Track 3b-f: IDF + blocking + candidates | `blocking.py` | ~65 min (final successful run; several hours cumulative across the OOM/segfault/disk-full iterations documented in `BLOCKING_REPORT.md`) | IDF/token tables <1 GB; 300k-sample recall pass (Part E) 1.4 GB; full test candidates (Part F) 3.4 GB |
| Track 4 Phase 1: feature functions + sanity check | `features.py` | ~15 min | not logged |
| Track 4 Phase 2: features on real candidates | `features_candidates.py train` / `test` | ~16 min train (×2, one rerun for a missing column), ~102 min test | not logged |
| Track 5: train + tune + LOCO + predict + validate | `train_model.py` | ~60 min (~43 min train/tune/LOCO + ~17 min test predict/submit/validate) | not logged |
| Track 5 extra: OOF + worst-20 report | `oof_report.py` | ~11 min | not logged |
| Package the submission zip | `make_package.py` | seconds (zip/deflate of already-computed outputs; not re-running any pipeline stage) | not logged (dominated by I/O of the ~790 MB `candidate_pairs.tsv`, not compute) |

The "not logged" cells are stages where `perf.py`'s `stage()` timing
wrapper was used but its RSS logging wasn't captured into either report —
not evidence those stages are free; `train_model.py` and
`features_candidates.py` in particular hold multi-million-row feature
frames and are not expected to be lighter than the Track 3 numbers above.

## Packaging the final submission

Once `output/matching_results.tsv` and `output/candidate_pairs.tsv` are
final (i.e. Step 3's blocking/model changes are done and the last
`train_model.py` run reflects them), build the submission zip:

```bash
.venv/Scripts/python.exe code/business_entity_resolution/src/make_package.py
```

This writes `<TEAM_NAME>_submission.zip` at the repo root with
`output/matching_results.tsv` + `output/candidate_pairs.tsv`, a clean copy
of `code/business_entity_resolution/` (`src/`, excluding `__pycache__`,
plus `README.md` and `requirements.txt`), and `Documentation_template.md`
— the exact structure required by the root `README.md`'s "Final
Submission Package" section. **Edit the `TEAM_NAME` constant at the top of
`make_package.py` first** — it defaults to the placeholder `"team"`. The
script prints the full file tree inside the zip and the zip's total size
when done; it never touches `data/`, `dataset/`, or `.venv/`.

## Status

Steps 1 and 2 complete. See `output/EDA_REPORT.md` (Step 1),
`output/BLOCKING_REPORT.md` (Step 2 Tracks 2-3 in full detail), and
`output/STEP2_REPORT.md` (every track summarized, including Track 4/5
numbers, top feature importances, the worst-20-OOF-entities error analysis,
and "what I'd try next"). Final test submission
(`output/matching_results.tsv` + `output/candidate_pairs.tsv`) passes
`utils/validate_submission.py`.

A Step 3 effort is in progress (separate session/teammate) to raise
blocking recall past the current 90.75% (chiefly non-Devanagari Indian
script coverage) and refine the model/decision layer; `requirements.txt`
already reflects an in-progress addition (`indic_transliteration`) from
that work as of this writing. Re-run `make_package.py` only after that
work lands and a fresh `train_model.py` run has produced final outputs.
