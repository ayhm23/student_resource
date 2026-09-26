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

## Status

Steps 1 and 2 complete. See `output/EDA_REPORT.md` (Step 1),
`output/BLOCKING_REPORT.md` (Step 2 Tracks 2-3 in full detail), and
`output/STEP2_REPORT.md` (every track summarized, including Track 4/5
numbers, top feature importances, the worst-20-OOF-entities error analysis,
and "what I'd try next"). Final test submission
(`output/matching_results.tsv` + `output/candidate_pairs.tsv`) passes
`utils/validate_submission.py`.
