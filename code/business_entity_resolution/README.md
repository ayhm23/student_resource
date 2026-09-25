# Business Entity Resolution — Step 1

Setup, EDA, and a safe first (all-empty) submission. No blocking or matching model
is implemented yet — this covers environment setup, data-quality checks, and the
metric implementation only.

## Setup

Run all commands from the `student_resource/` root (this repo's root).

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r code/business_entity_resolution/requirements.txt
```

Dependencies are pinned in `requirements.txt` (pandas, numpy, scikit-learn,
rapidfuzz, lightgbm, unidecode, pyarrow).

## Layout

```
code/business_entity_resolution/
├── requirements.txt
├── README.md
└── src/
    ├── config.py               # data paths, resolved relative to student_resource/
    ├── io_utils.py              # read_tsv, split_ids, write_id_list_tsv, f05 metric
    ├── first_look.py            # EDA / data-quality checks (a-i)
    ├── noise_mining.py          # name/address abbreviation-substitution mining
    ├── make_empty_submission.py # writes an all-empty first submission
    └── sanity_check_metric.py   # verifies f05 against known baselines
```

## Run order

```bash
# 1. EDA / data-quality checks -> output/step1_output.txt
.venv/Scripts/python.exe code/business_entity_resolution/src/first_look.py > output/step1_output.txt

# 2. Noise mining (abbreviation dictionary) -> output/substitutions.txt
.venv/Scripts/python.exe code/business_entity_resolution/src/noise_mining.py

# 3. Sanity-check the F0.5 metric implementation against known baselines
.venv/Scripts/python.exe code/business_entity_resolution/src/sanity_check_metric.py

# 4. Write the safe first submission (all entities, no matches)
.venv/Scripts/python.exe code/business_entity_resolution/src/make_empty_submission.py

# 5. Validate the submission format
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

All large source TSVs are streamed in chunks (see `CHUNK` in each script), so
none of these scripts need to hold a full multi-GB file in memory at once.

## Status

Step 1 only: environment, EDA (`output/EDA_REPORT.md`, `output/step1_output.txt`),
an abbreviation dictionary (`output/substitutions.txt`), a metric sanity check, and
an all-empty baseline submission that passes `utils/validate_submission.py`.
Blocking and a matching model are not implemented yet.
