# Step 2 (Track 3) Report — Normalization + Blocking

Pipeline: `src/mine_dicts.py` (Part A) → `src/normalize.py` (Part B) →
`src/blocking.py` (Parts C–F). All numbers below are from real runs over the
full dataset (2.2M/5.0M/5.3M train S1/S2/S3, 1.7M/4.9M/5.1M test S1/S2/S3),
using only training ground truth for anything "mined". No external data.

## Part A — Dictionaries mined from train pairs

Source: `data/dicts/mining_summary.txt`, mined from all 7,638,365 train
matched pairs.

**Devanagari→Latin token maps** (support≥5, agreement≥70%): 12 name tokens
kept (out of 13 seen in 855 equal-length dev/Latin name pairs), 3 address
tokens kept (out of 20 seen in 31,715 pairs — most address tokens didn't
reach 70% agreement, consistent with address component reordering noted in
Step 1). **Coverage**: only 18.8% of train / 18.7% of test Devanagari token
*occurrences* are covered by these maps (13.3% of distinct tokens); the rest
fall back to `unidecode`. This ceiling is a real limitation — see "what I'd
try next".

**Address abbreviations**: all 13 candidates verified with high support
(road→rd 421,797 hits, street→st 432,131, ... parkway→pkwy 10,554 — full
list in `data/dicts/address_abbrev_map.json`).

**State maps**: US 44/50 verified (dropped: hawaii, michigan, mississippi,
nevada, new hampshire, new jersey — genuinely low train volume, not a
matching bug). India 19/35 verified (dropped: jharkhand, chhattisgarh, assam,
uttarakhand, himachal pradesh, goa, tripura, meghalaya, manipur, nagaland,
mizoram, sikkim, arunachal pradesh, jammu and kashmir, puducherry,
chandigarh). A real bug was caught and fixed here: the first verification
pass used token-set intersection, which structurally cannot detect
multi-word candidates ("new hampshire", "tamil nadu", etc.) — every
multi-word US/India state was silently failing verification. Fixed by
switching to a longest-first regex-alternation match over the joined token
stream; multi-word states then verified correctly (e.g. "north carolina"→nc,
"tamil nadu"→tn).

**Name wrappers**: top separators (train frequency) — `the`:50,441,
`as`:28,068, `dba`:20,915, `f/k/a`:15,110, `formerly`:14,967, `aka`:10,604,
`a/k/a`:10,400, `d/b/a`:10,342, plus Indian honorifics appearing mid-string
(sri/smt/m/s/shri/mr/dr, ~8,300 each — these overlap with Task 5's
honorifics list, appearing both as separators and as leading tokens).
Bigram separators: `business as`:10,352, `trading as`:10,234, `known
as`:7,480.

**Legal forms / filler words / honorifics** (train frequency as
first/last token): legal forms dominated by `limited` (1,423,460), `llc`
(1,164,088), `inc` (853,791), `ltd` (831,584); French forms added by hand
(sarl/sas/sasu/eurl/sa/snc/sci) show ~0 train support as expected (no
French training data) and are trusted for test-time use only. Honorifics
m/s, dr, smt, shri, sri, mr all appear ~54-55k times each as the first name
token — a clear, consistent Indian-proprietorship naming pattern.

**Known mining artifacts excluded from all downstream use** (multi-word
diffs or typos that coincidentally look like substitutions, not real
synonyms): `private→center`, `inc→nc`, `and→nd`, `street→saint`, `plot→h`.

## Part B — Normalization

`normalize.py` computes `name_clean/name_main/name_alt/name_core/
name_nospace/name_sorted_chars/legal_form` and `addr_clean/numbers/state/
addr_words` per record (see the module docstring for the exact pipeline).
Ran on all 24.3M records across both splits in ~14 minutes using
`multiprocessing.Pool(4)` over 100k-row chunks streamed from DuckDB. Full
before/after examples for every (split, country) combination are in
`output/step1_output.txt`'s sibling run log (see repo history) — representative
examples:

- `Ultra Fashion (Ltd.)` → name_clean `ultra fashion ltd`, legal_form `ltd`
- `Xyloyuma Sys dba Costner Bsp Inc` → name_main `costner bsp inc`,
  name_alt `xyloyuma sys` (wrapper correctly split)
- `साईं प्रोडक्ट्स प्राइवेट लिमिटेड` → `saiin prodkts praaivett limaaittedd`
  style fallback for uncovered Devanagari tokens (unidecode, not a real
  translation)
- `258 Ridgewater Way, Mt Juliet, TN` → `state='mt'` (Montana) — a real,
  documented limitation: "Mt" here means "Mount" (part of the city name),
  not the state, but it's a valid state code and gets matched first. Low
  practical impact since the same mis-tag applies consistently to every
  record mentioning "Mt Juliet", so it doesn't break matching between two
  records that both have it — see "what I'd try next".

**A real bug was found and fixed during this run**: the first `normalize.py`
pass used a single global US+India merged state-code map, so a bare code
like `de` (Delaware) also fired on **French** addresses using "de" as the
ordinary preposition ("of/from" — e.g. "Rue de Cassel"), mistagging most
French addresses as `state='de'`. Fixed by scoping state-code recognition to
the record's own country (US records use only the US map, India only the
India map, everything else — the open-set case, currently just France —
uses a small hand-seeded region-name list: `nouvelle aquitaine`, `hauts de
france`, `pays de la loire`, from the 8 France samples seen in Step 1).
`normalize.py` was re-run in full after this fix; all `data/norm/*.parquet`
reflect the corrected version.

**A second, smaller limitation found but left as-is** (documented, not
fixed, given time budget): multi-word region phrases like "Pays de la
Loire" can lose their shared preposition ("de"/"la") to the name/address
cleaning pipeline's token-deduplication step when that word already
appeared earlier in the same address (e.g. "Rue de la Pierre..." earlier in
the string) — deduping removes the second occurrence needed to match the
phrase intact. "Hauts-de-France" and "Nouvelle-Aquitaine" are less exposed
to this (fewer/no shared common words) and matched correctly in most
samples seen.

## Part C — IDF

Per-(split, country) document frequency + IDF for `name_core` and
`addr_words` tokens, computed via DuckDB over the S1+S2+S3 union of each
split (`data/dicts/idf_name_{split}.parquet`, `idf_addr_{split}.parquet`).
~25-125s per table.

## Part D — Blocking keys

Every key includes country (open-set string equality). Four families, exactly
as specified:
- **K1** address: (country, number, rare addr word) — up to 3 numbers
  (longest-first) × 2 rarest `addr_words`.
- **K2** name+place: (country, state, rare name_core token) when state is
  known; **K2_nostate** (country, rare token) fallback with a stricter cap
  when it isn't.
- **K3** name_nospace: exact match, and first-8-characters prefix match.
- **K4**: K2/K3 recomputed on `name_alt` (the wrapper-prefix side), only
  when a wrapper was detected.

**Cap tuning (the "tune the cap" step the spec anticipated actually
mattered)**: started at cap=300 (default) / 100 (K2_nostate) per the initial
plan. On the 300k-sample vs. full-train-pool test this produced **56.3M raw
candidate rows / 49.0M distinct (S1, match) pairs** — far more than this
machine could hold for the downstream aggregation (see Engineering Notes).
Lowered to **cap=50 / 15**, which is what all recall numbers below reflect.
This is a real precision/recall/resource trade-off: a tighter cap drops
some legitimately-common-key true matches, but was necessary to complete
the run at all on this hardware. With more RAM/disk, a higher cap (and
likely +1-3 points of recall per the K1/K2/K3 numbers below) would be worth
re-testing.

**Cheap score**: `n_keys_hit` (count of distinct key types that matched)
plus the sum of IDF over every *shared* `name_core` and `addr_words` token
between the two records (not just the top-ranked ones used for key
generation) — computed as a proper set-based join against per-record token
tables, never a per-pair correlated subquery (an earlier version using
correlated subqueries did not scale past a few thousand pairs).

## Part E — Recall on the 300k train sample vs. full train pools

(`data/train_sample_s1.parquet`, built by Track 1's `folds.py`: 300,000 S1
ids stratified by country, seed 42.)

- Sample: 300,000 S1 entities, 1,037,463 true (S1, match) pairs across
  283,216 non-singleton entities.

**Recall per key type (before top-K cut, i.e. the raw union of everything
each key alone finds):**

| Key | Hit | Recall |
|---|---|---|
| K1 (address number+word) | 777,696 | 74.96% |
| K2 (name+state) | 392,493 | 37.83% |
| K2_nostate | 28 | 0.00% |
| K3_exact (name_nospace) | 481,703 | 46.43% |
| K3_pfx8 (name prefix-8) | 376,882 | 36.33% |
| K4_exact (alt name_nospace) | 9,962 | 0.96% |
| K4_state (alt name+state) | 18,050 | 1.74% |
| **UNION (any key)** | **941,457** | **90.75%** |

K1 (address-based) is by far the strongest single key — consistent with
Step 1's EDA finding that addresses are the stable half of this dataset
while names are the heavily-corrupted half. K2_nostate and K4 contribute
almost nothing at this cap; they exist mainly as a safety net for records
with no wrapper/no detected state.

**By country (union, before cut)**: India 362,731/414,999 (**87.41%**), US
578,726/622,464 (**92.97%**) — a real, meaningful gap, consistent with the
Devanagari-coverage ceiling from Part A and the wider script diversity in
Indian names (see below).

**By match source (union, before cut)**: S3 90.91%, S2 90.57% — near
parity, no meaningful source-specific blocking weakness.

**Recall vs. K (after the top-K cut)**:

| K | Recall |
|---|---|
| 10 | 87.61% |
| 20 | 89.43% |
| 30 | 90.06% |
| 40 | 90.38% |
| 60 | 90.65% |

**Target not reached**: the 97% union-recall target from the task spec was
**not achieved** — even the union of all four key families before any cut
tops out at 90.75%, and the recall-vs-K curve is nearly flat past K=20
(diminishing returns from the cut itself are small; the ceiling is set by
the union-before-cut number, i.e. by what the keys can find at all, not by
how many we keep). Largest swept K=60 achieves 90.65%. See "what I'd try
next" below — this needed a real second iteration round (more
normalization work, primarily on non-Devanagari Indian scripts) that there
wasn't time to complete in this session; it is reported honestly rather
than silently claimed as met.

**At K=60**: 217,182/283,216 (76.68%) of non-singleton S1 entities have
**every** true match found; 278,329/283,216 (**98.27%**) have **at least
one** found. The gap between these two numbers (76.7% vs 98.3%) says most
recall loss comes from entities with *several* true matches where only
some are found, not from entities that are missed entirely — good news for
a downstream classifier (partial recall on multi-match entities is
recoverable-ish; total misses are not).

**Candidates per S1 at K=60**: mean 30.3, median 28, p95 60 (i.e. a
meaningful fraction hit the cap), max 60, total 9,046,119 candidate pairs
for the 300k sample.

**30 random missed pairs, grouped by cause** (full list with raw + normalized
fields in the git history of this file / the run log): the dominant causes,
by inspection, are:
1. **Non-Devanagari Indian scripts** (~9 of 30): Telugu (`యూనివర్సల్...`),
   Malayalam (`സൂപ്പർ...`), Kannada (`ಸಾಯಿ...`, `ಶಿವ್...`, `ಡಿಜಿಟಲ್...`),
   Bengali (`সাউথ...`), Gujarati (`ઈસ્ટર્ન...`) — none of these are in the
   U+0900–U+097F Devanagari range the task scoped Part A to, so they get the
   generic `unidecode` fallback, which produces near-unusable garbled tokens
   (e.g. `yuunivrsl injniiring praiveett limittedd` for "Universal
   Engineering Private Limited") that share almost nothing with the Latin
   S1 side. This is the single largest identifiable recall gap and fully
   explains the India-vs-US recall difference.
2. **No shared number** (~11 of 30): genuine data corruption/truncation on
   one side (e.g. `2064` vs `64`, `5821` vs missing entirely, `742` vs
   `1742`) — not a normalization bug, a real limit of number-based blocking
   when the number itself is corrupted or dropped.
3. **Cap edge cases** (~4 of 30): both sides do share tokens/numbers, but
   the specific rare-word combination didn't survive the tightened cap —
   directly attributable to the cap=50 tuning trade-off above.
4. **Character-level typos breaking name tokens** (~3 of 30): e.g.
   `Ferrola`/`Ferro1a`, `Committee`/`Commitbee`, `Patfges`/`Pantages` —
   single-character substitutions that change the token entirely, invisible
   to exact/prefix-based name keys.

## Part F — Full test candidates

(`data/cand/test_candidates.parquet`, `output/candidate_pairs.tsv`.)

- 1,732,544 test S1 entities; 1,722,995 (**99.45%**) got ≥1 candidate,
  9,549 got zero.

| Country | S1 count | Mean candidates | Median | Zero-candidate rate |
|---|---|---|---|---|
| US | 663,106 | 26.7 | 23 | 0.64% |
| France | 259,452 | 26.1 | 22 | 0.83% |
| India | 809,986 | 36.5 | 38 | 0.39% |

**France does not look dramatically different from US/India** — mean
candidate count (26.1) is close to the US's (26.7), and its zero-candidate
rate (0.83%) is only slightly higher than the US's (0.64%), well within the
same order of magnitude as India's (0.39%). This is a genuinely encouraging
sign: even with zero French training data and only a 3-region hand-seeded
list, country-scoped blocking on addresses/numbers (which don't depend on
having a translation dictionary) generalizes reasonably to France. India's
higher mean (36.5) likely reflects longer, more multi-component free-text
Indian addresses producing more K1 number+word combinations per record, not
a blocking-quality difference.

`output/candidate_pairs.tsv` was written (one row per test S1, comma-joined
`S2-/S3-` ids, empty when none) and validated structurally by construction
(same `write_id_list_tsv` helper as Step 1's submission writer, which passed
`utils/validate_submission.py` in Step 1).

## Timings & memory (this 16GB machine)

| Stage | Wall time | Peak RSS |
|---|---|---|
| mine_dicts.py (Part A) | ~7 min | <0.25 GB |
| normalize.py (Part B, both fixes' final run) | ~14 min | <0.5 GB/worker |
| compute_idf + build_token_tables (Part C, both splits) | ~3 min | <1 GB |
| Part E (300k sample vs. full train pools) | ~18 min | 1.4 GB |
| Part F (all 1.73M test S1 vs. full test pools) | ~45 min | 3.4 GB |

## Engineering notes: getting Part D-F to run at all on this hardware

This took far longer than the compute numbers above suggest, because of
real, repeated resource exhaustion on this specific machine (16GB RAM
laptop, C: drive at 90%+ full with as little as ~14GB free) — worth
recording since `docs/AWS_SETUP.md` exists specifically so this doesn't
recur on a properly-sized box:

1. Building keys for the full ~10M-row S2/S3 pool in one pass caused a real
   `OutOfMemoryException`, then once, under combined system memory
   pressure, a native segfault.
2. Switched DuckDB from pure `:memory:` to an on-disk database file so its
   buffer manager could spill instead of crashing — fixed the segfault, but
   the underlying OOM recurred.
3. Restructured `build_keys_table`'s four token-family CTEs (name, alt,
   addr, numbers) into separate sequential materialized tables instead of
   one multi-CTE query — DuckDB was holding all four window-function
   pipelines' state at once.
4. Diagnosed that the raw candidate join before scoring was **56.3M rows /
   49.0M distinct pairs** at the original cap=300 — the real root cause, not
   just a memory-tuning problem. Lowered the cap to 50/15 (see Part D).
5. Filtered the S2/S3-side token tables down to just the ids present in the
   current candidate batch before the shared-IDF join, instead of joining
   against the full multi-million-row token tables every time.
6. Ran out of disk (`IOException`: temp spill couldn't write) on the C:
   drive (~14GB free at the time) — moved DuckDB's scratch database and
   spill directory to D: (~130GB free) at the user's suggestion.
7. Even per-country, test's largest countries (US 663k, India 810k S1
   records — 4-5x bigger than any per-country slice in the 300k train
   sample) still exhausted a 116GB temp-directory limit on D: during
   scoring. Added a second chunking level: each country's S1 side is now
   also split into fixed 150,000-row batches, reusing that country's
   already-built S2/S3 keys across batches.

None of this changes correctness (country-scoped chunking and S1 batching
are both pure partitions of independent work), only where a genuinely
20M+-row-scale join gets computed. It's the direct, lived version of the
project rule "if RAM ≤16GB, process per country and in chunks" — one level
of chunking (by country) turned out not to be enough at this data volume,
and a second level (fixed-size batches within a country) was needed too.

## What I'd try next

1. **Extend Devanagari-style mining to other Indian scripts.** This is the
   highest-leverage fix identified: Telugu (U+0C00–U+0C7F), Kannada
   (U+0C80–U+0CFF), Malayalam (U+0D00–U+0D7F), Bengali (U+0980–U+09FF), and
   Gujarati (U+0A80–U+0AFF) all appear in the missed-pair sample and none
   are covered by the current pipeline, which the task spec scoped to
   Devanagari only. Mining an equivalent per-script token map (same
   equal-length-pairs-plus-alignment method) would directly target the
   87.4%-vs-92.97% India/US recall gap.
2. **Re-test with a higher blocking cap on a bigger machine.** The cap=50/15
   tuning was a resource-forced trade-off, not a precision/recall-driven
   one; the K1/K2/K3 per-key recall numbers suggest real headroom was left
   on the table (the original cap=300 run had reached the join stage
   successfully and only failed during aggregation).
3. **Fix the pre-dedup phrase-matching order** for France's multi-word
   region names (Part B's second documented limitation) — extract
   multi-word state/region phrases before token deduplication, not after.
4. **Number-similarity fallback key**, e.g. edit-distance-1 on the numeric
   token, to catch the "no shared number" corruption cases (2064 vs 64,
   etc.) found in the missed-pairs sample — currently numbers must match
   exactly to contribute to K1.
5. **A char-n-gram or phonetic key** (e.g. Soundex/double-metaphone on
   name_core, or trigram overlap) to catch single-character-substitution
   typos (Ferrola/Ferro1a, Committee/Commitbee) that break exact/prefix
   name keys entirely.
