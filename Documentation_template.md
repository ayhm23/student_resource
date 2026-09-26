# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 0. Executive Summary

**Approach, in plain terms.** We resolve Source 1 (the deduplicated
reference) against Source 2/Source 3 in two stages: a rule-based
**blocking** stage cuts the ~24M-record cross product down to a small
per-entity candidate set using normalized name/address keys, and a
**LightGBM binary classifier** scores every (S1, candidate) pair on ~25
similarity/blocking-derived features, followed by a tuned decision rule
that turns per-pair probabilities into the final match lists. No external
data of any kind is used anywhere in the pipeline (see the Compliance
section).

**What's been tried.** Step 1 was pure EDA (noise-pattern discovery: legal
suffix swaps, Devanagari transliteration, DBA/trade-name wrapping,
character-level typos, address abbreviation/reordering noise; see
`output/EDA_REPORT.md`). Step 2 built the full pipeline end to end:
dictionary mining from train ground-truth pairs (transliteration maps,
address abbreviations, state names, legal-form/wrapper vocabularies),
field normalization, four families of blocking keys, ~25 pair features, a
5-fold LightGBM classifier, and a tuned probability-to-match decision
rule. A **Step 3** effort is in progress in parallel (by another
teammate/session) to raise blocking recall — chiefly by extending script
coverage beyond Devanagari to Telugu/Kannada/Malayalam/Bengali/Gujarati —
and to add features/refine the decision layer; those numbers are not
final yet and are marked `[TODO: Step 3 numbers pending]` below rather
than guessed.

**Current results** (Step 1+2, final): 5-fold CV macro F0.5 **0.9336**
(US 0.9496, India 0.9097), public leaderboard **0.92**, blocking union
recall **90.75%** (US 92.97%, India 87.41%) on a 300k-entity stratified
train sample. LOCO (leave-one-country-out) checks: training on US only
and scoring India gives 0.7995; training on India only and scoring US
gives 0.9020 — the model relies on some country-specific signal that
doesn't fully transfer, and the US→India direction transfers noticeably
worse, mirroring the India-side blocking gap.

**Main open problem.** Blocking recall is below the 97% target set by the
task spec. The single dominant cause, identified independently by both
the blocking missed-pairs analysis and the worst-20-OOF-entities error
analysis, is **non-Devanagari Indian scripts** (Telugu, Kannada,
Malayalam, Bengali, Gujarati) falling through to a generic `unidecode`
fallback that produces near-unusable garbled tokens sharing almost
nothing with the Latin-script S1 side. This fully explains the
India-vs-US recall gap and is the direct target of the in-progress Step 3
work. A secondary, smaller cause is corrupted/truncated address numbers
(e.g. "1951" vs "195") defeating exact-match number-based blocking keys.

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

We block candidates using four families of normalized name/address keys
mined and verified against real train pairs (no external data), then
classify each (S1, candidate) pair with a 5-fold LightGBM model over ~25
similarity and blocking-derived features, and finally apply a tuned
relative-margin decision rule to produce the match lists. The main
technical contribution is the normalization layer itself: a
frequency-mined, support/agreement-verified Devanagari transliteration
dictionary, address-abbreviation and state-name canonicalization tables,
and wrapper/DBA/legal-form/honorific splitting — all mined exclusively
from the training ground truth and checked for real bugs (see Section
2.1) rather than assumed correct.

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

Full detail in `output/EDA_REPORT.md`. Headline findings that shaped every
later design decision:

- **Names are the noisy half, addresses are the stable half.** Feature
  separation later confirms this directly (Section 4): `addr_num_jaccard`
  separates positives/negatives far more cleanly (0.840 vs 0.009) than
  most name features, and address features dominate the model's top
  feature-importance ranking.
- **Exclusivity and country agreement are near-perfect in train**: 0 of
  7,638,365 matched pairs have differing country labels; 0 S2/S3 records
  are matched to more than one S1 entity across all of train. Both were
  used as strong priors (country-scoped blocking; a downstream
  one-to-one-ish assignment expectation) but never as unconditional hard
  filters, since test's France segment has no ground truth to re-verify
  either property against.
- **~26% of S2 and ~25% of S3 are pure distractors** (no S1 match at all)
  — blocking must discriminate against a large true-negative pool, not
  just find true positives.
- **Devanagari transliteration pairs share zero alphanumeric tokens** with
  their Latin-script counterpart (e.g. "Sai Products Private Limited" vs
  "साईं प्रोडक्ट्स प्राइवेट लिमिटेड") — pure string-similarity features
  cannot see these at all without a transliteration step first.
- **DBA/trade-name wrapping** ("Xyloyuma Sys dba Costner Bsp Inc") and
  **social-handle-style names** ("#integratedconstruction") are genuinely
  hard true positives that need substring/wrapper-aware handling, not
  just edit distance.
- **France (test-only, 14.98% of test S1, zero training examples)** uses
  distinct legal forms (SARL/SASU/EURL) and French administrative regions
  instead of US states or Indian PIN codes/states — `country` must be
  treated as an open string label throughout, never hard-coded to
  `{US, India}`.

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** Blocking + Classifier (hybrid rule-based candidate
generation followed by a gradient-boosted decision-tree classifier and a
tuned decision rule).

**Core Innovation:** A normalization layer built entirely from
*mined-and-verified* dictionaries — every substitution table (Devanagari
token map, address abbreviations, state names, wrapper separators,
legal-form/honorific vocabulary) is derived from real train ground-truth
pairs with explicit support/agreement thresholds, and cross-checked for
false positives rather than assumed correct. This process caught and
fixed two real bugs before they could silently corrupt every downstream
stage:

1. **A France/Delaware state-code collision.** `normalize.py`'s first
   pass used a single global US+India merged state-code map, so the bare
   code `de` (Delaware) also fired on French addresses using "de" as the
   ordinary preposition ("Rue de Cassel"), mistagging most French
   addresses as `state='de'`. Fixed by scoping state-code recognition to
   the record's own country (US records use only the US map, India only
   the India map, everything else — the open-set case, currently just
   France — uses a small hand-seeded region-name list mined from the 8
   France samples seen in Step 1). `normalize.py` was re-run in full
   after the fix.
2. **A multi-word state-name verification bug.** The first dictionary
   verification pass used token-set intersection, which structurally
   cannot detect multi-word candidates ("new hampshire", "tamil nadu")
   at all — every multi-word US/India state was silently failing
   verification. Fixed by switching to a longest-first regex-alternation
   match over the joined token stream; multi-word states then verified
   correctly (e.g. "north carolina"→nc, "tamil nadu"→tn).

Both bugs and their fixes are documented in full in
`output/BLOCKING_REPORT.md` Parts A-B, including exact before/after
verification counts.

**Normalization pipeline summary** (`src/normalize.py`, full detail in
`BLOCKING_REPORT.md` Part B): computes `name_clean/name_main/name_alt/
name_core/name_nospace/name_sorted_chars/legal_form` and `addr_clean/
numbers/state/addr_words` per record.

- **Unicode**: NFKC normalization first, to fold compatibility characters
  and combining forms before any token-level processing.
- **Devanagari transliteration**: a mined token map (support≥5,
  agreement≥70%, 12 name tokens / 3 address tokens kept) applied first,
  falling back to `unidecode` for any token not covered. Coverage is a
  real, documented limitation: only 18.8%/18.7% of train/test Devanagari
  token *occurrences* are covered by the mined map; the rest fall back to
  a romanization that is not a real translation (e.g. "साईं प्रोडक्ट्स
  प्राइवेट लिमिटेड" → "saiin prodkts praaivett limaaittedd"-style output)
  and shares little with the true Latin name.
- **Address abbreviations**: all 13 candidates (road→rd, street→st,
  avenue→ave, parkway→pkwy, ...) verified with high train support
  (hundreds of thousands of hits each).
- **State canonicalization**: US 44/50 and India 19/35 state names
  verified against real train pairs (the remainder genuinely low-volume
  in train, not a matching bug) using the fixed multi-word-aware matcher
  above.
- **Wrapper/legal-form/honorific handling**: separators like `dba`,
  `f/k/a`, `formerly`, `aka`, `d/b/a`, bigram separators (`business as`,
  `trading as`, `known as`) split a wrapped name into `name_main` /
  `name_alt`. Indian honorifics (sri/smt/m/s/shri/mr/dr) are recognized
  both as separators and as leading tokens (~8,300-55k train occurrences
  each). Legal forms (limited, llc, inc, ltd, plus hand-added French
  forms sarl/sas/sasu/eurl/sa/snc/sci, trusted for test-time-only use
  since train has zero French records) are stripped into a dedicated
  `legal_form` field.
- **Known mining artifacts excluded from use**: `private→center`,
  `inc→nc`, `and→nd`, `street→saint`, `plot→h` — high-count pairs that
  are multi-word diffs or typos coincidentally reducing to one
  differing token, not real synonyms.

`[TODO: Step 3 numbers pending]` — non-Devanagari Indian script coverage
(Telugu/Kannada/Malayalam/Bengali/Gujarati) is the in-progress Step 3
work; final coverage/recall impact numbers are not yet available.

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

Full detail in `output/BLOCKING_REPORT.md` Parts C-F. Every blocking key
includes `country` as an open-set string-equality component (never a
fixed `{US, India}` allowlist), since 0 of 7,638,365 train matched pairs
cross a country boundary.

- **Blocking keys used** — four key families, all IDF-weighted where
  relevant (`data/dicts/idf_name_*.parquet`, `idf_addr_*.parquet`):
  - **K1** — address: `(country, number, rare addr word)`, up to 3
    numbers (longest-first) × 2 rarest `addr_words`. The single strongest
    key (74.96% recall alone), consistent with addresses being the
    stable half of the data.
  - **K2** — name+place: `(country, state, rare name_core token)` when
    state is known; **K2_nostate** `(country, rare token)` fallback with
    a stricter cap when it isn't.
  - **K3** — `name_nospace`: exact match, and first-8-character prefix
    match.
  - **K4** — K2/K3 recomputed on `name_alt` (the wrapper-prefix side),
    only when a wrapper was detected.
  - A **cheap_score** (`n_keys_hit` plus summed IDF over every shared
    `name_core`/`addr_words` token) is carried into the candidate set as
    a blocking-derived feature for the model (Section 4).

- **Candidate pairs generated:** 9,046,119 candidate pairs for the 300k
  train sample (mean 30.3, median 28, p95/max 60 per S1 entity at the
  top-K=60 cut). Full test set: **59,663,232** candidate pairs across
  1,732,544 test S1 entities (99.45% got ≥1 candidate); mean candidates
  per S1 by country — US 26.7, France 26.1, India 36.5. This last number
  is a genuinely encouraging generalization signal: despite zero French
  training data, France's candidate volume and zero-candidate rate
  (0.83%) are close to US's (26.7 mean, 0.64% zero-rate), because
  country-scoped address/number blocking doesn't depend on having a
  translation dictionary the way name-based blocking does.

- **Cap tuning — a resource constraint, not a modeling choice.** The
  original plan's cap=300 (default) / 100 (K2_nostate) was tested first
  and produced 56.3M raw candidate rows / 49.0M distinct (S1, match)
  pairs on just the 300k-sample-vs-full-train-pool test — far more than
  this 16GB machine could hold for the downstream aggregation (see the
  engineering notes in `BLOCKING_REPORT.md`: an `OutOfMemoryException`,
  a native segfault, and a disk-full `IOException` on a 90%-full C:
  drive were all hit before the cap was lowered). The cap was reduced to
  **50 / 15**, which is what every recall number below reflects. This is
  an explicit precision/recall/resource trade-off: a tighter cap drops
  some legitimately-common-key true matches, and the per-key recall
  numbers suggest real headroom (+1-3 points) would be recovered on a
  properly-sized machine at the original cap.

- **How true matches were checked against being lost:** recall was
  measured directly against the true ground-truth match set on a
  300,000-entity stratified (by country, seed 42) train sample, both
  per-key-alone and as a union, both before and after the top-K cut —
  not estimated or assumed:

  | Key | Recall (alone) |
  |---|---|
  | K1 (address number+word) | 74.96% |
  | K2 (name+state) | 37.83% |
  | K3_exact (name_nospace) | 46.43% |
  | K3_pfx8 (name prefix-8) | 36.33% |
  | K4 (alt name, both variants) | ≤1.74% each |
  | **UNION (any key), before cut** | **90.75%** |

  By country (union, before cut): **US 92.97%**, **India 87.41%** — the
  gap is fully explained by non-Devanagari script coverage (Section 2.1,
  5). By match source: S3 90.91%, S2 90.57% (near parity, no
  source-specific weakness). At the applied cut (K=60): 76.68% of
  non-singleton entities have *every* true match found, 98.27% have *at
  least one* — most recall loss is partial (some but not all matches
  found on multi-match entities), not total misses, which is favorable
  for the downstream classifier.

  **Target status:** the 97% union-recall target from the task spec was
  **not reached** (90.75% achieved) — reported honestly rather than
  claimed as met. See Section 5 and `BLOCKING_REPORT.md`'s "what I'd try
  next" for the identified fixes (non-Devanagari script mining is the
  highest-leverage one).

`[TODO: Step 3 numbers pending]` — updated union/per-country recall
numbers once non-Devanagari script mining and any additional blocking
keys (e.g. a number-edit-distance fallback, a phonetic/char-n-gram key)
land.

---

## 4. Matching Model

**Features used** (~25 total, built and unit-tested in `src/features.py`
on all 7,638,365 train positive pairs plus an equal number of random
same-country negatives before being computed on real candidates in
`src/features_candidates.py`):

- **Name features:** `name_token_set_ratio`, `name_partial_ratio`,
  `name_ratio`, `name_sorted_chars_ratio`, `name_nospace_ratio`
  (rapidfuzz), `name_jaro_winkler`, `name_idf_jaccard` (IDF-weighted
  token Jaccard over `name_core`), `name_idf_rarest_shared` /
  `name_idf_rarest_unshared`, `name_len_absdiff`, `legal_form_match`.
- **Address features:** `addr_token_set_ratio`, `addr_token_sort_ratio`
  (rapidfuzz), `addr_num_jaccard`, `addr_num_max_equal` (numeric token
  overlap/agreement), `addr_idf_jaccard` (IDF-weighted token Jaccard over
  `addr_words`), `addr_state_match`.
- **Other:** blocking-derived features (`n_keys_hit`, `cheap_score`,
  `rank_in_s1`, `n_candidates_for_s1`), competition features
  (`n_competitors`, `rank_among_competitors`), meta (`other_source`,
  `other_name_len`).

**Positive-vs-negative separation** (Phase 1 sanity check, full table in
`output/STEP2_REPORT.md`) confirmed every feature carries real signal
before candidates were even built:

| Feature | Positive mean | Negative mean |
|---|---|---|
| name_token_set_ratio | 90.35 | 33.09 |
| name_idf_jaccard | 0.708 | 0.005 |
| addr_token_set_ratio | 96.14 | 35.93 |
| addr_num_jaccard | 0.840 | 0.009 |
| addr_state_match=same | 85.0% of positives | 5.8% of negatives |
| legal_form_match=same | 36.9% of positives | 13.0% of negatives |

(`name_idf_rarest_unshared` was the one weak feature — 10.17 vs 10.91 —
expected, since it measures the rarest *unshared* token, present on both
classes almost equally often.)

**Model type:** LightGBM binary classifier (seed 42), 5-fold
`GroupKFold` CV using pre-assigned folds (`data/folds.parquet`), early
stopping (50 rounds, up to 2000 total). Trained in ~11 min on the
9.9M-row train-sample candidate set (941,457 positive pairs — matching
Section 3's recall numerator exactly).

**Top feature importances (mean gain across folds)** — address features
and the blocking-derived `rank_in_s1` dominate, consistent with addresses
being the stable signal and blocking rank already encoding much of the
"is this plausible" judgment before the classifier looks at text
similarity at all:

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

(Full top-20 in `output/STEP2_REPORT.md`.)

**Threshold selection method:** grid search over three decision-rule
variants on OOF probabilities, scored with the true macro F0.5 metric
(not a proxy) against the challenge's own ground truth:

| Variant | t | alpha | macro F0.5 | precision | recall | singleton acc | US | India |
|---|---|---|---|---|---|---|---|---|
| global (threshold only) | 0.70 | — | 0.9334 | 0.9652 | 0.8702 | 0.9423 | 0.9495 | 0.9093 |
| exclusive (+ argmax S2/S3 ownership) | 0.70 | — | 0.9335 | 0.9653 | 0.8702 | 0.9424 | 0.9495 | 0.9094 |
| **relative (+ margin vs. S1's best)** | **0.65** | **0.7** | **0.9336** | **0.9654** | **0.8705** | 0.9354 | 0.9496 | 0.9097 |

All three variants land within 0.0002 of each other — the exclusivity
constraint (already near-perfect in the data per Step 1's EDA) and the
relative-margin rule add only marginal further gains once a good global
threshold is already in place. **"Relative" (t=0.65, alpha=0.7) is the
chosen rule** used for the final submission — accept a candidate if its
probability clears the global threshold `t` **and** is within `alpha` of
the best-scoring candidate for that same S1 entity, which recovers a
little more recall on multi-match entities without a meaningful
precision cost.

`[TODO: Step 3 numbers pending]` — any new features (e.g. targeting
non-Devanagari-script or corrupted-number cases) and any revision to the
decision layer.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), 5-fold CV:** **0.9336** overall (US 0.9496,
  India 0.9097). Public leaderboard: **0.92**.
- **LOCO (leave-one-country-out), same tuned decision rule:**

  | Direction | macro F0.5 | precision | recall | singleton acc |
  |---|---|---|---|---|
  | Train US → score India | 0.7995 | 0.8588 | 0.7135 | 0.7475 |
  | Train India → score US | 0.9020 | 0.9213 | 0.8777 | 0.6795 |

  Both are meaningfully below the full 5-fold CV (0.9336), confirming
  the model relies on some country-specific signal (state codes, name
  conventions, address formats) that doesn't fully transfer. The
  US→India direction transfers noticeably worse — consistent with the
  India-side blocking recall gap: fewer India-only true matches are even
  visible to a US-only-trained model, and India's script-diversity issue
  compounds the same direction.

- **Common false positives (wrong merges):** not the dominant error mode
  observed — see the worst-20-OOF-entities finding below, which found
  zero classifier-driven false merges among the worst-scoring entities.
  The precision-heavy F0.5 metric and the tuned relative-margin rule both
  work against false merges directly (precision 0.9654 on the full CV
  run).
- **Common false negatives (missed matches):** the **20 worst-scoring OOF
  entities** (full detail in `output/worst20_oof_entities.txt`) **all
  scored F0.5 = 0.000 because the model predicted an empty list, and
  every single one was a blocking miss** — no candidate was ever
  generated for the true match, so the classifier never had a chance to
  score it. Inspecting them confirms the same two dominant causes already
  identified in the blocking missed-pairs analysis:
  1. **Non-Devanagari Indian scripts** — e.g. Kannada `ಆಲ್ಫಾ ಫುಡ್ಸ್
     ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` for "Alpha Foods Private Limited" — none of
     Telugu/Kannada/Malayalam/Bengali/Gujarati are in the
     Devanagari-only mining scope, so they get an unusable generic
     `unidecode` fallback.
  2. **Corrupted/truncated address numbers** — e.g. "1951 Nomas Street"
     vs "195 NOMAS ST", where the extracted numbers (1951 vs 195) never
     match, defeating exact-match number-based K1 blocking. A related
     "no shared number" pattern (genuine data corruption/truncation, not
     a normalization bug) accounted for ~11 of 30 randomly-sampled missed
     pairs in the broader blocking analysis.

  **This is the clearest signal in the whole analysis that the
  classifier itself is working well and the remaining gap is almost
  entirely a blocking-recall problem, not a classification problem** —
  90.75% blocking recall caps the achievable score at roughly 0.95-ish
  under F0.5's precision-heavy weighting with zero false positives, and
  0.9336 says the classifier makes good use of what blocking did find.

- **France candidate-volume finding** (Section 3): France's mean
  candidate count (26.1) and zero-candidate rate (0.83%) sit close to
  the US's (26.7, 0.64%) despite zero French training data — a genuinely
  encouraging sign that country-scoped, address/number-driven blocking
  generalizes to an unseen country reasonably well, since it doesn't
  depend on a name-translation dictionary the way K2/K3/K4 do.

`[TODO: Step 3 numbers pending]` — updated CV/LOCO numbers, an
ablation table isolating the effect of any new blocking keys or
features, and a France-specific quality check (no France ground truth
exists, so this would necessarily be a proxy check, e.g. candidate-volume
or feature-distribution comparison against US/India rather than a direct
F0.5 number).

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

A mined-and-verified normalization layer plus country-scoped blocking
plus a LightGBM classifier achieves 0.9336 CV / 0.92 public leaderboard
macro F0.5, with the classifier itself performing well — essentially
every remaining error traces back to blocking recall (90.75%, driven
almost entirely by non-Devanagari Indian script coverage), not to
classification quality. The main lesson learned is architectural: on a
resource-constrained machine, the blocking-cap tuning (300/100 → 50/15)
and the two-level chunking needed to make Parts D-F complete at all
(per-country, then per-country-per-batch) mattered as much to the final
number as any modeling decision, and are documented as real engineering
constraints rather than modeling choices.

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

Full structure and exact run order are in
`code/business_entity_resolution/README.md`. Summary: `config.py`/`db.py`/
`perf.py`/`io_utils.py` are shared infrastructure; `first_look.py` →
`noise_mining.py` → `sanity_check_metric.py` → `make_empty_submission.py`
is Step 1 (EDA + baseline); `ingest.py` → `folds.py`/`evaluate.py` →
`mine_dicts.py` → `normalize.py` → `blocking.py` →
`features.py`/`features_candidates.py` → `train_model.py` →
`oof_report.py` is Step 2, run in that order, and is the entry-point
sequence that reproduces both `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from `dataset/train`/`dataset/test`. A new
`make_package.py` builds the final submission zip itself (see the
README's "Packaging" section) once the output files are final — it is
not part of the modeling pipeline and does not need to be re-run when
re-training.

**Compliance / fair play:** no external data, database, API, or service
of any kind is used anywhere in this pipeline — every dictionary
(transliteration map, address abbreviations, state names, wrapper/
legal-form/honorific vocabulary) is mined exclusively from
`dataset/train/train_ground_truth.tsv`'s matched pairs and verified
against real train data, per `output/BLOCKING_REPORT.md` Part A. No
geocoding, no commercial entity-resolution service, no internet lookups.
The final model (LightGBM, MIT-licensed, a single gradient-boosted tree
ensemble well under 8B parameters) satisfies the "MIT/Apache-2.0, ≤8B
parameters" model constraint.

**Dependency licenses** (verified via `pip show <name>` /
`importlib.metadata` in this project's `.venv`, not assumed from memory):

| Package | Version | License |
|---|---|---|
| pandas | 2.3.3 | BSD-3-Clause |
| numpy | 2.2.6 | BSD-3-Clause (BSD License classifier) |
| scikit-learn | 1.7.2 | BSD-3-Clause |
| rapidfuzz | 3.14.5 | MIT |
| lightgbm | 4.7.0 | MIT |
| Unidecode | 1.4.0 | **GPL — see note below** |
| pyarrow | 25.0.1 | Apache-2.0 |
| duckdb | 1.5.5 | MIT |
| psutil | 7.2.2 | BSD-3-Clause |
| indic_transliteration | 2.3.82 | MIT |

**⚠️ Unidecode is GPL-licensed, not MIT/Apache-2.0 — flagged explicitly
for review.** It is used only as a fallback transliteration step inside
`normalize.py` (never as part of the final scoring model itself, which is
LightGBM), but the competition's license constraint as stated in
`README.md` is scoped to "final model" (MIT/Apache-2.0, ≤8B parameters) —
LightGBM satisfies that constraint on its own. Whether a GPL-licensed
*preprocessing* dependency is acceptable under the letter of that rule is
a real open question this team should resolve with the organizers before
final submission, not something this document should assume an answer
to. `indic_transliteration` (MIT) was added to `requirements.txt` as part
of the in-progress Step 3 script-coverage work and is confirmed
MIT-licensed; it does not raise the same concern.

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

See `output/BLOCKING_REPORT.md` (dictionary mining, normalization,
blocking keys, recall breakdowns, engineering/OOM war story, full
missed-pairs cause analysis) and `output/STEP2_REPORT.md` (evaluation
harness, all-track summary, full top-20 feature importances, full
decision-rule comparison table, LOCO detail, timings/memory table across
every pipeline stage) for every number and table referenced above, plus
`output/worst20_oof_entities.txt` for the raw worst-20 entity records.

`[TODO: Step 3 numbers pending]` — any new charts/tables Step 3 produces
(e.g. a per-script recall breakdown, an updated feature-importance
ranking).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
