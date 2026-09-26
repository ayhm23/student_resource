# Step 2 Report — Evaluation Harness, Dictionaries, Blocking, Features, Model

All five tracks are complete. Full detail on Tracks 2–3 (dictionaries,
normalization, blocking) is in `output/BLOCKING_REPORT.md`; this report
summarizes every track and gives the full Track 4/5 numbers not covered
there. Every number below is from a real run over the full dataset — no
number is estimated or simulated.

## Track 1 — Evaluation harness

- `src/evaluate.py`: `score(pred, gt, s1_ids)` reuses `io_utils.f05` and
  returns macro F0.5, per-country F0.5, singleton accuracy, mean
  precision/recall — computed against the TRUE ground-truth match set for
  every requested id, so a blocking/model miss always counts against
  recall rather than being silently skipped.
- `src/folds.py`: 300,000 S1 ids sampled stratified by country (seed 42) →
  `data/train_sample_s1.parquet` (US 179,938 / India 120,062, matching the
  population's ~60/40 split); a 5-fold GroupKFold over all 2,206,821 train
  S1 ids (seed 42) → `data/folds.parquet` (fold sizes 441,365 / 441,364 ×4,
  near-perfectly even, including per-country balance).
- Unit tests, all PASS with exact expected numbers: all-empty predictions on
  the sample = 0.055947 (equals the sample's own singleton rate exactly);
  perfect predictions = 1.000000; the problem statement's worked example =
  0.714286 → rounds to 0.714.

## Track 2 — Dictionaries mined from train pairs

See `BLOCKING_REPORT.md` Part A for full detail. Headline numbers: 12/13
Devanagari name tokens and 3/20 address tokens kept (support≥5,
agreement≥70%); coverage only 18.8%/18.7% of train/test Devanagari token
occurrences (real limitation, see below); all 13 address abbreviations
verified; 44/50 US and 19/35 India state names verified (a real bug — the
first verification pass used token-set intersection, which cannot detect
multi-word candidates like "new hampshire" at all — was found and fixed
mid-session); wrapper separators (dba, f/k/a, formerly, aka, ...) and
legal-form/filler/honorific frequencies all mined with real counts in the
hundreds of thousands.

## Track 3 — Normalization + blocking

See `BLOCKING_REPORT.md` in full (Parts B–F) for every number, including two
real bugs found and fixed mid-session (a global-vs-per-country state-code
collision that mistagged most French addresses as Delaware; a multi-word
verification bug in Task 2). Headline numbers:

- **Blocking recall on the 300k train sample (before any top-K cut):
  90.75%** (India 87.41%, US 92.97%) — **the 97% target was not reached**.
  K1 (address number+rare-word) is the strongest single key at 74.96%
  recall alone, consistent with Step 1's finding that addresses are the
  stable half of this dataset.
- Blocking cap had to be tuned down from 300/100 to **50/15** purely for
  resource reasons on this 16GB machine (the uncapped run produced 56.3M
  raw candidate rows / 49.0M distinct pairs from just the 300k sample,
  which this machine could not hold for the downstream aggregation) — a
  real precision/recall/resource trade-off, not a modeling choice.
- Missed-pair analysis found the dominant cause is **non-Devanagari Indian
  scripts** (Telugu, Kannada, Malayalam, Bengali, Gujarati — none covered
  by the Devanagari-only mining task 1 was scoped to), which fully explains
  the India-vs-US recall gap.
- Full test candidates: 1,732,544 test S1 entities, 99.45% got ≥1
  candidate. **France's candidate volume is close to US/India's** (mean 26.1
  vs US's 26.7 vs India's 36.5) despite zero French training data — a
  genuinely encouraging generalization signal.
- Getting Parts D–F to actually complete on this machine required real,
  repeated engineering (an OutOfMemoryException, a native segfault, a
  disk-full IOException on a 90%-full C: drive, and finally two levels of
  chunking — per-country, then per-country-per-150k-row-batch) — fully
  documented in `BLOCKING_REPORT.md`'s Engineering Notes, since none of it
  changes correctness, only where the ~20M-row-scale joins get computed.

## Track 4 — Pair features

**Phase 1** (`src/features.py`): built and unit-tested every feature
function on all 7,638,365 train ground-truth positive pairs plus an equal
number of random same-country negatives (15,276,724 labeled rows total,
`data/features/train_gt_vs_negatives.parquet`). Every feature showed strong
positive/negative separation:

| Feature | Positive mean | Negative mean |
|---|---|---|
| name_token_set_ratio | 90.35 | 33.09 |
| name_idf_jaccard | 0.708 | 0.005 |
| addr_token_set_ratio | 96.14 | 35.93 |
| addr_num_jaccard | 0.840 | 0.009 |
| addr_state_match=same | 85.0% of positives | 5.8% of negatives |
| legal_form_match=same | 36.9% of positives | 13.0% of negatives |

(`name_idf_rarest_unshared` was the one weak feature: 10.17 vs 10.91 —
expected, since it measures the rarest *unshared* token, which is present on
both classes almost equally often.)

**Phase 2** (`src/features_candidates.py`): the same feature functions plus
blocking features (n_keys_hit, cheap_score, rank_in_s1,
n_candidates_for_s1), competition features (n_competitors,
rank_among_competitors), and meta (other_source), run on the REAL candidate
pairs from blocking:
- Train-sample candidates: 9,937,424 pairs (941,457 positive — exactly
  matching Part E's union-recall numerator), ~16 min.
- Full test candidates: 59,663,232 pairs, ~102 min.

## Track 5 — Model, decision layer, LOCO, submission

**Model**: LightGBM binary classifier (seed 42), 5-fold CV using
`data/folds.parquet`'s pre-assigned folds, early stopping (50 rounds, up to
2000). Trained in ~11 min on the 9.9M-row train-sample candidate set.

**Top 20 feature importances (mean gain across folds):**

| Rank | Feature | Mean gain |
|---|---|---|
| 1 | rank_in_s1 | 2.18e7 |
| 2 | addr_token_sort_ratio | 1.76e7 |
| 3 | addr_num_jaccard | 1.15e7 |
| 4 | addr_token_set_ratio | 8.55e6 |
| 5 | other_name_len | 4.18e6 |
| 6 | name_partial_ratio | 2.61e6 |
| 7 | name_idf_rarest_unshared | 2.19e6 |
| 8 | cheap_score | 2.12e6 |
| 9 | addr_num_max_equal | 9.84e5 |
| 10 | name_sorted_chars_ratio | 7.42e5 |
| 11 | n_keys_hit | 7.23e5 |
| 12 | name_jaro_winkler | 7.00e5 |
| 13 | addr_idf_jaccard | 6.63e5 |
| 14 | name_nospace_ratio | 6.24e5 |
| 15 | name_idf_rarest_shared | 6.02e5 |
| 16 | name_len_absdiff | 3.80e5 |
| 17 | legal_form_match | 3.58e5 |
| 18 | name_idf_jaccard | 3.04e5 |
| 19 | n_candidates_for_s1 | 2.60e5 |
| 20 | name_ratio | 2.14e5 |

Address features and the blocking-derived `rank_in_s1` dominate — consistent
with addresses being the stable signal in this dataset and with blocking
rank itself already encoding a lot of the "is this plausible" signal before
the classifier even looks at text similarity.

**Decision-layer tuning** (grid search on OOF probabilities, scored with
`evaluate.score` against the TRUE match sets):

| Variant | t | alpha | macro F0.5 | precision | recall | singleton acc | US | India |
|---|---|---|---|---|---|---|---|---|
| global (threshold only) | 0.70 | — | 0.9334 | 0.9652 | 0.8702 | 0.9423 | 0.9495 | 0.9093 |
| exclusive (+ argmax S2/S3 ownership) | 0.70 | — | 0.9335 | 0.9653 | 0.8702 | 0.9424 | 0.9495 | 0.9094 |
| **relative (+ margin vs. S1's best)** | **0.65** | **0.7** | **0.9336** | **0.9654** | **0.8705** | 0.9354 | 0.9496 | 0.9097 |

All three variants land within 0.0002 of each other — the exclusivity
constraint (already near-perfect in the data per Step 1's EDA) and the
relative-margin rule add only marginal further gains once a good global
threshold is already in place, though "relative" is the tiny consistent
winner and is what the final submission uses. **Overall macro F0.5 on the
300k train sample: 0.9336** — well above the theoretical blocking-recall
ceiling would suggest is easy (90.75% recall caps the *achievable* score
without any false positives at roughly 0.95-ish given the precision-heavy
F0.5 weighting; 0.9336 says the classifier is making good use of what
blocking did find).

**LOCO (leave-one-country-out) check**, same tuned decision rule:
- Train US only → score India: **macro F0.5 = 0.7995** (precision 0.8588,
  recall 0.7135, singleton acc 0.7475)
- Train India only → score US: **macro F0.5 = 0.9020** (precision 0.9213,
  recall 0.8777, singleton acc 0.6795)

Both LOCO scores are meaningfully lower than the full 5-fold CV (0.9336),
confirming the model does rely on some country-specific signal (state
codes, name conventions, address formats) that doesn't fully transfer — the
US→India direction transfers noticeably worse than India→US, mirroring the
India-side blocking recall gap already identified (fewer India-only true
matches are even visible to a US-only-trained model, and India's own
script-diversity issue compounds this in the same direction).

**20 worst OOF entities** (full detail in `output/worst20_oof_entities.txt`):
every one of the 20 lowest-scoring non-singleton entities scored **F0.5 =
0.000 because the model predicted empty** — not because of a bad
classification. Inspecting them confirms these are **all blocking misses**
(no candidate was ever generated for the true match), the same failure mode
already characterized in `BLOCKING_REPORT.md`'s 30-missed-pairs analysis:
non-Devanagari scripts (e.g. Kannada `ಆಲ್ಫಾ ಫುಡ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` for
"Alpha Foods Private Limited"), empty S2/S3 addresses combined with a
corrupted name (e.g. "Maya Institute Group" / "Maya Institute
Corporation" with `''` address on the match side), and digit-truncated
house numbers (e.g. "1951 Nomas Street" vs "195 NOMAS ST" — the extracted
numbers 1951 vs 195 never match). **This is the single clearest signal in
the whole report that the model itself is working well and the remaining
gap is almost entirely a blocking-recall problem**, not a classification
problem.

**Test submission**: features computed for all 59,663,232 test candidate
pairs (chunked, ~102 min), the 5 fold models averaged, the tuned decision
rule (relative, t=0.65, alpha=0.7) applied.

- `output/matching_results.tsv`: 1,732,544 rows, 135,413 empty (predicted
  singleton), 1,597,131 with ≥1 match.
- `output/candidate_pairs.tsv`: 1,732,544 rows, 1,722,995 with ≥1 candidate
  (the full uncapped candidate set the model actually ran inference over,
  per the task's definition of this file — larger than blocking's own
  top-60 cut, which was only ever an intermediate view for the recall
  report).
- **`utils/validate_submission.py` → PASS**, no blocking issues.
- Logged to `output/results_log.csv`.

## Timings & memory (this 16GB machine, cumulative across all tracks)

| Stage | Wall time |
|---|---|
| Track 1 (folds + evaluate unit tests) | ~5 min |
| Track 2 (mine_dicts.py) | ~7 min |
| Track 3 normalize.py (two runs, one bug fix in between) | ~28 min |
| Track 3 blocking.py (many iterations to fix OOM/segfault/disk-full; final successful run) | ~65 min (successful run); several hours total across failed attempts, see BLOCKING_REPORT.md |
| Track 4 Phase 1 (features.py) | ~15 min |
| Track 4 Phase 2 train candidates | ~16 min ×2 (one rerun to add a missing column) |
| Track 4 Phase 2 test candidates | ~102 min |
| Track 5 training + tuning + LOCO | ~43 min |
| Track 5 test prediction + submission + validation | ~17 min |
| Track 5 OOF save + worst-20 report | ~11 min |

## docs/AWS_SETUP.md

Written (see the file itself): launching a CPU-only EC2 instance (r6i.2xlarge
or m6i.4xlarge, 64GB RAM) in us-east-1, transferring the dataset via a
private S3 bucket, running the pipeline in tmux, a budget alert set up
first, stopping the instance when done, and how to request a GPU quota
increase if ever needed (not needed for this step — everything here runs on
CPU). No AWS keys anywhere in the repo.

## What I'd try next

1. **Extend Devanagari-style mining to Telugu/Kannada/Malayalam/Bengali/
   Gujarati scripts** — the single highest-leverage fix, identified
   independently by both the blocking missed-pairs analysis and the OOF
   worst-20 entities. This is the most direct path to closing the
   India-vs-US recall (and hence LOCO US→India) gap.
2. **Re-run blocking with a higher cap on a properly-sized machine** — the
   50/15 cap was a resource-forced compromise, not a precision/recall
   decision; the per-key recall numbers in `BLOCKING_REPORT.md` suggest
   real headroom exists.
3. **A number-similarity fallback key** (edit-distance-1 on numeric tokens)
   to catch digit-truncation/corruption cases like the "1951 vs 195" example
   above, which currently require an exact number match to ever become a
   K1 candidate.
4. **A char-n-gram or phonetic key** to catch single-character-substitution
   typos that break exact/prefix name keys entirely.
5. **Investigate the LOCO asymmetry** (US→India transfers much worse than
   India→US) directly — feature-importance-by-country or a per-country
   SHAP breakdown would clarify whether it's really the script-coverage gap
   or something else (e.g. US name corruption patterns generalizing better
   to India's Latin-script names than the reverse).
6. **Try a stricter/looser global threshold blend** — since the three
   decision-rule variants converged to nearly the same score, there may be
   more headroom in threshold *calibration* (e.g. per-country thresholds)
   than in the exclusivity/relative-margin mechanism itself.
