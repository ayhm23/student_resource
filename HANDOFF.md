# Handoff — Amazon ML Challenge 2026, Business Entity Resolution

Read this first in a new chat/session to pick up exactly where this one left
off. This doc is a snapshot as of **2026-09-26, ~14:35 IST**. Give this whole
file to Claude at the start of a new conversation and say "continue from
HANDOFF.md" — that's enough context to resume without re-deriving anything.

## TL;DR

- **Public leaderboard: 0.92** (submission #2, LightGBM baseline). Real,
  submitted, working. `output/archive/sub2_lb092/` has the exact files that
  scored it (kept locally only — gitignored, too large for GitHub).
- We're mid-way through a "Step 3" push to improve blocking recall (was
  90.75%, root cause = non-Devanagari Indic scripts + address-number
  corruption not handled). Track A2/A4/A5 changes are written and validated
  on the train sample (recall 90.75% → **91.41%**, India gained more than
  US as hoped). A **full pipeline rerun** (blocking → features → retrain →
  new submission) was in progress when this session ended.
- This machine is the bottleneck: 16GB RAM, C: drive often <30GB free. Every
  pipeline stage has been engineered around that (see "Machine constraints"
  below). On a bigger machine, things will be much simpler and faster — see
  "If you're now on a better machine" at the bottom.

## TODO (not yet started): the post-bug-fix rebuild chain

Three real bugs (below) were found and fixed in `normalize.py`/`mine_dicts.py`/
`blocking.py`/`features_candidates.py` on 2026-09-26, but **the rebuild these
fixes require has NOT been run yet** — code is fixed and pushed (commit
`d19161c`), data/artifacts are still the pre-fix versions. This is next,
whenever/wherever (this machine or the lab machine) picks it up:

```bash
cd student_resource
.venv/Scripts/python.exe code/business_entity_resolution/src/mine_dicts.py                       # ~7 min
.venv/Scripts/python.exe code/business_entity_resolution/src/normalize.py                        # ~17 min
.venv/Scripts/python.exe code/business_entity_resolution/src/blocking.py                         # ~90 min
.venv/Scripts/python.exe code/business_entity_resolution/src/blocking.py trainfull               # ~90-115 min (NEW, bug #4 fix)
.venv/Scripts/python.exe code/business_entity_resolution/src/features.py                         # ~15 min
.venv/Scripts/python.exe code/business_entity_resolution/src/features_candidates.py train        # ~16 min
.venv/Scripts/python.exe code/business_entity_resolution/src/features_candidates.py patch-competitors  # ~1-2 min (NEW)
.venv/Scripts/python.exe code/business_entity_resolution/src/features_candidates.py test          # ~100 min
.venv/Scripts/python.exe code/business_entity_resolution/src/train_model.py                       # ~60 min
```

**Total estimate: ~6.5-7 hours on this machine** (should be much faster on
the lab machine once its environment issue is fixed — see "If you're now on
a better machine"). Cheaper alternative if you just want to validate the
bug-#1/#3 recall improvement before committing to the full chain: run only
the first 3 steps above (`mine_dicts.py` → `normalize.py` → `blocking.py
train-only`, ~114 min) and check `output/BLOCKING_REPORT_v2_trainonly.md`
for the new recall number before deciding whether to continue.

## Bugs found + fixed (2026-09-26, this continuation)

A pasted report describing bug fixes made on a different (lab) machine's
variant of this pipeline was checked line-by-line against our actual code,
not assumed to transfer. 3 of 11 items described real, currently-present
bugs here; the rest either didn't apply or are already-known deferred work.

**Fixed:**

1. **Indic word-splitting** (`normalize.py`, `mine_dicts.py`,
   `translit_compare.py`). `\w+` does not match Devanagari/Indic combining
   vowel signs or virama (Unicode categories Mn/Mc) — confirmed empirically:
   "शक्ति" (shakti) tokenized as `['शक', 'त']`, silently losing the vowel
   sign in between. This shredded every native-script name/address before
   comparison, and is almost certainly why the Devanagari name-token
   dictionary only reached 13.3% coverage. Fix: extended `WORD_RE` to
   include the full Brahmic Unicode blocks (Devanagari, Bengali, Gurmukhi,
   Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam), not just the
   combining-mark subranges — additive only, so pure-Latin tokenization
   (US/France) is byte-identical to before. Verified: "शक्ति" now tokenizes
   whole. `mine_dicts.py`/`translit_compare.py` now import `WORD_RE` from
   `normalize.py` instead of keeping their own stale copies.

2. **State-priority bug** (`normalize.py::_apply_state_and_abbrev`). A
   single left-to-right pass over address tokens let a bare short state
   code (e.g. "ap", which collides with an "Apartment" abbreviation) win
   over a full state name appearing later in the same address — e.g. "Ap
   Xiv/326, Kannur, Kerala" was read as Andhra Pradesh instead of Kerala.
   Fix: full state names (multi-word or single-word) now always take
   priority over short codes regardless of position; only if no full name
   is found does a short code apply, and now the *last* one wins, not the
   first. Verified on the exact example above: now resolves to `kl`
   (Kerala).

3. **Train/test `n_competitors` mismatch** (narrow fix, in progress).
   `folds.py`'s 300k-row train S1 sample means `n_competitors`/
   `rank_among_competitors` (`features_candidates.py`) were computed
   against only the 300k sample for training but against the full 1.73M
   test S1 for test — a real covariate shift on two live features. Chose
   the narrow fix over a full retrain-on-2.2M-rows rebuild (that risks
   16GB RAM and 4-5+ extra hours for a ~65M-row training table): re-block
   the full 2.2M train S1 once into a new `trainfull_scored` table
   (reusing `build_all_keys`, which already handles arbitrary row counts),
   recompute the 2 columns from that full-population table, and patch
   them into the existing 9.9M-row `trainsample_candidates_features.parquet`
   by `(s1_id, match_id)` join — no need to re-run `compute_pair_features`.

**Checked, doesn't apply to us:** native-script state dictionaries (we
don't have one — nothing to be "half-mapped"), blocking-OOM-from-joining-
on-common-words (our join is already keyed on `(country, key_type,
key_value)` + chunked, different mechanism, already solved), missing
top-K cut before featurizing (we already have `TOP_K_PER_S1`).

**Checked, real ideas but correctly deferred** (already tracked as Step 3
P1/P2, intentionally not done this session): raising blocking caps on a
bigger machine (biggest lever, see "If you're now on a better machine"
below), a fuzzy house-number blocking key (new idea, not yet built), 4 new
pair features beyond what we have (name containment ratio, domain-style
name match, off-by-one house number, unshared-word count), the two-stage
model (Track C4), and a per-S1 expected-F0.5 decision rule (Track D).

**Because bugs #1 and #2 above are inside `normalize.py`, everything
downstream must be rebuilt from `mine_dicts.py` onward** — dictionaries,
norm parquet, blocking keys/candidates, and features were all built from
the buggy tokenizer/state logic. The "just run it" sequence below is
UPDATED for this; the old one (skip straight to `features_candidates.py`)
is now stale. No trained model exists yet on any version of this data, so
nothing already-submitted (LB 0.92) is invalidated.

## What "just run it" means right now

Rebuild order after the 2026-09-26 bug fixes above (full sequence, not the
shortcut): `mine_dicts.py` → `normalize.py` → `blocking.py` →
`features_candidates.py train` → `features_candidates.py test` → new
trainfull-blocking pass for the n_competitors fix → `train_model.py`. See
"Exact resume sequence" below for the commands; expect several hours total
on this machine (see timing table) — print/confirm the estimate before
starting per the standing rule.

If the background run from this session finished cleanly (check
`output/BLOCKING_REPORT_v2_trainonly.md` mtime and look for a completed
`blocking.py` process), the very next steps are:

```bash
cd student_resource
.venv/Scripts/python.exe code/business_entity_resolution/src/features_candidates.py train
.venv/Scripts/python.exe code/business_entity_resolution/src/features_candidates.py test
.venv/Scripts/python.exe code/business_entity_resolution/src/train_model.py
```
(~16 min, ~100 min, ~60 min respectively — see timing table below.) That
gets you a new checkpoint submission (C1 in the Step 3 plan). **If the
background run did NOT finish** (check for a Python traceback in whatever
log it was writing to, or just that `data/norm/*.parquet` files look stale),
rerun `normalize.py` then `blocking.py` first — see "Exact resume
sequence" below.

After that checkpoint, there is real unfinished work (not just running
things) — Track A3, C2, C3, C4, D, and most of Track E from the Step 3 plan
below were not attempted this session due to time. Be honest with the user
about this; don't silently skip and claim done.

## Project facts (don't re-derive these)

- Working directory: `student_resource/` inside this repo. Dataset (~2.5GB,
  gitignored) already present under `dataset/`.
- Train: S1 2.21M / S2 5.03M / S3 5.29M. Test: S1 1.73M / S2 4.89M / S3
  5.08M. Test countries: US, India, France (France = 15% of test S1, zero
  training labels).
- Metric: macro F0.5 per S1 entity (singleton = true empty match set scores
  1.0 iff predicted empty). Implemented in `src/io_utils.py::f05` and
  `src/evaluate.py::score` (also handles per-country breakdown, blocking
  misses count against recall).
- Matched pairs ALWAYS share country (0 mismatches in 7.64M train pairs) —
  block within country, open-set string equality, never a fixed list.
  Exclusivity (S2/S3 id belongs to ≤1 S1) is also perfect in train.
- GitHub repo: `https://github.com/ayhm23/student_resource` (private).
  Latest pushed commit as of this handoff: `e26f6f8` ("step2: dictionaries,
  normalization, blocking, features, and LightGBM model"). **The user does
  not want Claude/AI attribution in any commit** — no `Co-Authored-By`
  lines, ever, for this repo. Commits so far are authored as the user
  (`ayhm23 <archit.jaju@iiitb.ac.in>`), which is already the local git
  config — don't change it.
- `output/results_log.csv` row 1: CV macro F0.5 0.9336 (US 0.9496, India
  0.9097), LOCO US→India 0.7995, India→US 0.9020, public_lb **0.92**.

## Machine constraints (why the pipeline looks the way it does)

- 16GB RAM total, often only 3–9GB actually free (browser/IDE already using
  a lot). `code/business_entity_resolution/src/db.py`'s `connect()` caps
  DuckDB's memory and uses an **on-disk** database (not `:memory:`) so it
  can spill instead of crashing.
- **Scratch/spill location is `D:/ber_scratch`**, not under the repo on C:.
  C: has repeatedly run down to ~14–30GB free and caused real `disk full`
  crashes; D: has ~130GB free. If you're on a different machine, check
  `db.py`'s `SCRATCH_DIR` logic — it falls back to `data/` under the repo if
  `D:` doesn't exist, which is fine on a machine with real disk space.
- `blocking.py` processes **per country, then per 150,000-row S1 batch
  within each country** — even after per-country chunking, test's largest
  country (India, ~810k S1) alone blew a 116GB temp-directory limit before
  this second level of batching was added. This chunking is pure
  partitioning (doesn't change results), so it's safe to keep even on a
  bigger machine, just maybe less necessary.
- Getting `blocking.py` to run at all took **7 separate crash-and-fix
  cycles** this session (OOM, a native segfault, disk-full, a DuckDB
  `BinderException` on a prepared-statement-in-CREATE-VIEW, a dead
  `key_types` list aggregate, a stale-table bug after refactoring). All
  fixed; see `output/BLOCKING_REPORT.md`'s "Engineering notes" section for
  the full story if something breaks again in a similar way.
- `train_model.py`'s test-time prediction is **chunked** (reads test
  features in 200k-row batches via DuckDB, never loads the full ~3GB/59M-row
  test feature parquet into one pandas DataFrame) for the same reason.

## Timing table (this machine; expect much faster on a bigger one)

| Stage | Time | Notes |
|---|---|---|
| `ingest.py` | ~10s | raw TSV → parquet |
| `mine_dicts.py` | ~7 min | dictionaries from train GT |
| `normalize.py` | ~14–17 min | v1 was 14min, v2 (+skeleton) ~17min |
| `blocking.py` (train-only) | ~25 min | Part E only, for cheap validation |
| `blocking.py` (full) | ~90 min | Part E + Part F (full test) |
| `blocking.py trainfull` | ~90-115 min (est.) | NEW Part G, bug #4 fix: full 2.2M train S1 |
| `features.py` (Phase 1) | ~15 min | GT pairs + negatives, sanity check |
| `features_candidates.py train` | ~16 min | 9.9M candidate pairs |
| `features_candidates.py test` | ~100 min | 59.7M candidate pairs |
| `train_model.py` | ~43 min train+tune+LOCO, ~17 min test predict+submit | |
| `oof_report.py` | ~11 min | saves OOF probs + worst-20 report |

**Always print/estimate runtime before starting anything over ~20 min** —
this was an explicit ask from the user and is good practice given how many
things have gone wrong on this machine.

## Exact resume sequence (from a cold start, e.g. new machine)

```bash
cd student_resource
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r code/business_entity_resolution/requirements.txt

.venv/Scripts/python.exe code/business_entity_resolution/src/ingest.py
.venv/Scripts/python.exe code/business_entity_resolution/src/folds.py
.venv/Scripts/python.exe code/business_entity_resolution/src/evaluate.py   # unit tests, should all PASS
.venv/Scripts/python.exe code/business_entity_resolution/src/mine_dicts.py
.venv/Scripts/python.exe code/business_entity_resolution/src/normalize.py
.venv/Scripts/python.exe code/business_entity_resolution/src/blocking.py   # ~90 min; pass "train-only" as arg to skip Part F for a cheap check
.venv/Scripts/python.exe code/business_entity_resolution/src/blocking.py trainfull   # NEW, ~90-115 min: bug #4 fix, full 2.2M train S1 competitor counts
.venv/Scripts/python.exe code/business_entity_resolution/src/features.py
.venv/Scripts/python.exe code/business_entity_resolution/src/features_candidates.py train
.venv/Scripts/python.exe code/business_entity_resolution/src/features_candidates.py patch-competitors   # NEW, ~1-2 min: joins in the trainfull fix
.venv/Scripts/python.exe code/business_entity_resolution/src/features_candidates.py test
.venv/Scripts/python.exe code/business_entity_resolution/src/train_model.py
.venv/Scripts/python.exe code/business_entity_resolution/src/oof_report.py  # optional, for the worst-20 report

python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## What's DONE vs NOT STARTED in the Step 3 plan

The user's Step 3 spec has tracks A–F with priorities P0 (must) → P1 → P2,
plus "always keep a valid submission" and "skip P2 if it risks not finishing
by Sunday 2PM IST, and say so."

**Done this session:**
- Setup: `public_lb=0.92` logged, old submission archived to
  `output/archive/sub2_lb092/` (local only, gitignored), RAM/cores/disk
  printed (15.7GB RAM, 8physical/12logical cores, confirmed <48GB so caps
  stayed at 50/15 per the plan's own rule).
- **A1 (miss diagnosis)**: done, `src/miss_diagnosis.py` +
  `output/miss_diagnosis_A1.txt`. Key finding: **73% of misses are
  Latin/Latin** (same-script corruption: number truncation, word reorder,
  domain-style names), not script-related — this reprioritized A2 below A4/A5
  in practice. Only ~27% of misses involve a non-Latin script.
- **A2 (transliteration)**: done, `src/translit_compare.py`. Surprising
  result: **unidecode beats indic_transliteration on 7 of 9 Indic scripts**
  (already measured on real train pairs, not assumed) — so the fix was
  building a "skeleton" (consonant-folded, vowel-stripped) representation on
  top of unidecode's output, not switching transliteration libraries.
  `indic_transliteration` was installed, tested, then **removed** from
  `requirements.txt` since it didn't win. `normalize.py` now has
  `skeleton_token()`/`skeleton_join()` and stores `name_skeleton`/
  `addr_skeleton` fields.
- **A4 (numbers, partial)**: `normalize.py`'s `extract_numbers()` now also
  recovers hyphen-split short numbers ("2-0"→20, "23-27"→2327 in addition to
  23,27). The full spec's "#/leading-zero/letter stripping" was already
  working via the existing `\d+` regex (verified, not re-implemented). The
  **1-deletion-variant fuzzy-number idea (K7) was NOT implemented** — time
  ran out; it's probably the next-highest-leverage item given "no shared
  number" was the #2 miss cause after plain vocabulary mismatch.
- **A5 (partial)**: two new key families added to `blocking.py`:
  - **K5** (skeleton-based, mirrors K2's state/nostate pattern) — 24.2%
    recall alone on the train sample.
  - **K6** (sorted pair of the 2 rarest name_core tokens, no address needed
    at all) — 44.9% recall alone, biggest single addition, though most of
    that overlaps with what K1-K4 already found (net new pairs are smaller
    than the raw number suggests).
  - **K3 filler-word stripping, K7 (fuzzy number), K8 (number+city),
    reserved-slots-per-S1 scheme: NOT implemented.**
- **A3 (state/city aliases in Indic scripts): NOT STARTED at all.**
- **A6 (reblock + measure)**: done on train sample only (`train-only` mode).
  v1→v2: union recall 90.75%→**91.41%** (US 92.97%→93.43%, India
  87.41%→88.39%). A **full test-candidate rerun (Part F) was in progress**
  when this session ended — check if it finished (see "TL;DR" above).
- **Track B (features v2, partial)**: added `name_skeleton_ratio`,
  `name_skeleton_token_set_ratio`, `name_skeleton_jaccard` to
  `features.py`/`train_model.py`. Smoke-tested: correctly scores a garbled
  vs. clean name pair as 100% skeleton-similar vs. only 73.7% on raw
  `name_ratio`. **Char n-gram TF-IDF cosine, acronym/first-token features,
  numbers_v2-based features, locality/city-alias features: NOT
  implemented.**
- **C1 (retrain + checkpoint): NOT YET RUN** — this is the very next step
  once blocking v2's Part F finishes (see "What just run it means" above).
- **C2, C3, C4, Track D, most of Track E: NOT STARTED.** Given the amount of
  time A1-A6+B already took (multiple hours, several real crashes), these
  P1/P2 items are realistically the next session's work, not something to
  rush through.
- **Track F (docs + packaging): DONE** by a parallel subagent —
  `Documentation_template.md` filled in (Step 1+2 content; Step 3 numbers
  marked `[TODO: Step 3 numbers pending]` in 4 places), new
  `src/make_package.py` (builds the submission zip, not yet run — waits for
  final outputs), `code/business_entity_resolution/README.md` updated with
  a timing/RAM table. **Important flag from that agent: `Unidecode` is
  GPL-licensed**, not MIT/Apache — everything else pinned is MIT/Apache/BSD.
  The competition's license rule (README.md constraint #5) is scoped to the
  "final model" (LightGBM, MIT, clean), but whether a GPL *preprocessing*
  dependency is fine under the letter of that rule is unconfirmed — worth
  asking the organizers rather than assuming either way. Not fixed/swapped
  this session (unidecode is deeply load-bearing and swapping it isn't
  free); just flagged.

## Key files map

```
student_resource/
├── HANDOFF.md                      <- this file
├── README.md                       <- original problem statement (competition rules)
├── Documentation_template.md       <- filled in through Step 2; Step 3 TODOs marked
├── output/
│   ├── EDA_REPORT.md               Step 1
│   ├── BLOCKING_REPORT.md          Step 2 Tracks 2-3 (v1 blocking, 90.75% recall)
│   ├── BLOCKING_REPORT_v2_trainonly.md   Step 3 Track A6 (v2 blocking, 91.41% recall on train sample)
│   ├── STEP2_REPORT.md             Step 2 all-tracks summary (the 0.9336 CV / 0.92 LB numbers)
│   ├── miss_diagnosis_A1.txt       Step 3 Track A1
│   ├── worst20_oof_entities.txt    Step 2 Track 5 error analysis
│   ├── results_log.csv             one row so far (0.9336 CV / 0.92 LB)
│   ├── matching_results.tsv, candidate_pairs.tsv  <- CURRENT submission (gitignored, regenerate via train_model.py)
│   └── archive/sub2_lb092/         <- the exact files that scored 0.92 (gitignored, local fallback)
├── docs/AWS_SETUP.md               <- EC2 runbook if you want to move off this laptop
└── code/business_entity_resolution/
    ├── requirements.txt            pinned deps (indic_transliteration removed after A2 testing)
    ├── README.md                   run order + timing/RAM table
    └── src/
        ├── config.py                all paths
        ├── db.py                    DuckDB connection factory (D: scratch, memory cap)
        ├── perf.py                  print_sysinfo()/stage() timing+memory logging
        ├── io_utils.py              f05 metric, split_ids, write_id_list_tsv
        ├── ingest.py, mine_dicts.py, normalize.py, blocking.py   <- pipeline, v2 changes this session
        ├── evaluate.py, folds.py    Track 1 eval harness
        ├── features.py, features_candidates.py   Track 4 (features.py has new skeleton features)
        ├── train_model.py, oof_report.py          Track 5 model
        ├── miss_diagnosis.py, translit_compare.py  Step 3 Track A1/A2 diagnostics (one-off scripts)
        └── make_package.py          Track F submission zip builder (not yet run)
```

## If you're now on a better machine

Most of the defensive chunking/capping in this codebase exists because of
this specific 16GB laptop. On a machine with ≥48GB RAM:
- `blocking.py`'s own docstring/plan says to raise `CAP_DEFAULT`/
  `CAP_K2_NOSTATE` from 50/15 back toward 300/100 — this was a real,
  measured recall cost (the original cap=300 run produced 56M candidate
  rows and genuinely found more true pairs before the memory crash forced
  the cap down). Try this first; it's probably the single biggest lever
  left for the recall numbers above.
- You can likely drop the per-country-per-batch chunking in `blocking.py`
  and the chunked test prediction in `train_model.py` back to simpler
  single-pass code, though there's no harm in leaving them (they're correct,
  just conservative).
- `db.py`'s `SCRATCH_DIR` will fall back to `data/` under the repo
  automatically if `D:/` doesn't exist as a path — check it still points
  somewhere with real free space on the new machine.
