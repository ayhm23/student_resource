# Step 1 EDA Report — Business Entity Resolution

Source data for every number below: `output/step1_output.txt` (checks a–i, produced by
`code/business_entity_resolution/src/first_look.py`) and `output/substitutions.txt`
(produced by `src/noise_mining.py`). All numbers come from streaming the raw
train/test TSVs; no external data was used.

## 1. Row counts (a)

Pandas row counts match `raw_line_count - 1` (the header row) for all 7 files — no
truncation, no embedded-newline corruption, no silent single-column parse:

| File | Rows |
| --- | --- |
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| train_ground_truth | 2,206,821 |
| test_source1 | 1,732,544 |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

## 2. Country labels (b)

- Train: `{US, India}` only, on all three sources.
- Test: `{US, India, France}`. **259,452 / 1,732,544 (14.98%)** of test Source-1 rows
  are France — a country never seen in training. This confirms `country` must be
  treated as an open string label, never hard-coded to `{US, India}`.

## 3. Match structure (c)

- Singleton rate (no match): **123,247 / 2,206,821 = 5.58%**.
- Matches-per-entity distribution is unimodal, peaking at 3 matches:

  | # matches | # entities | | # matches | # entities |
  |---|---|---|---|---|
  | 0 | 123,247 | | 6 | 164,868 |
  | 1 | 119,157 | | 7 | 63,968 |
  | 2 | 375,212 | | 8 | 18,680 |
  | 3 | 530,841 | | 9 | 4,205 |
  | 4 | 484,115 | | 10 | 534 |
  | 5 | 321,957 | | 11 | 37 |

- Matched-id instances split almost evenly: S2 = 3,693,619 (48.4%), S3 = 3,944,746
  (51.6%).

## 4. Exclusivity (d)

Across all **7,638,365** distinct matched S2/S3 ids in the training ground truth,
**zero** are matched to more than one Source-1 entity. Exclusivity (each S2/S3 record
belongs to at most one S1 entity) holds perfectly in train.

## 5. Distractors and referential integrity (e)

- Every matched id in the ground truth exists in `train_source2.tsv` /
  `train_source3.tsv` (0 missing).
- Distractors (S2/S3 records with no S1 match at all): S2 = 1,340,997 / 5,034,616
  (**26.64%**), S3 = 1,340,857 / 5,285,603 (**25.37%**). About a quarter of each
  source is pure noise from a matching standpoint — a real precision challenge for
  blocking.

## 6. Country consistency on matched pairs (f)

Of **7,638,365** matched (S1, S2/S3) pairs, **0 (0.00%)** have different country
labels between the two sides. Country is a perfectly reliable signal in training.

## 7. Empty fields (g)

- `business_name` is never empty in any file (train or test, any source).
- `business_address` is empty for a similar share in train and test:
  train S2 3.36%, train S3 3.33%; test S2 2.65%, test S3 2.68%. Source1 addresses
  are never empty.

## 8. Source2+3 volume per Source1 record (h)

- Train: (5,034,616 + 5,285,603) / 2,206,821 = **4.68** S2/S3 records per S1 record.
- Test: (4,887,273 + 5,082,316) / 1,732,544 = **5.75** S2/S3 records per S1 record —
  a higher ratio than train, consistent with test introducing an extra country
  (France) plus more distractors.

## 9. Sample matched pairs and France records (i)

15 sampled matched pairs and 8 France test records are printed in full in
`output/step1_output.txt`. Representative noise patterns observed directly in the
samples:

- Legal-suffix and abbreviation swaps: `Ltd` ↔ `Limited`, `Inc` ↔ `Incorporated`,
  parenthesized suffixes `(Ltd.)`, `(LP)`.
- Script/transliteration variants with **no English token overlap at all**, e.g.
  `Sai Products Private Limited` vs `साईं प्रोडक्ट्स प्राइवेट लिमिटेड` — a pure
  Devanagari transliteration of the same name.
- Character-level typos disguised as accented letters: `Ínsight`, `Glóbal`,
  `Prívate`, `MEDlCAL` (capital-I for lowercase-l) — these are synthetic corruption,
  not real diacritics.
- DBA / trade-name wrapping: `Xyloyuma Sys dba Costner Bsp Inc`, `Umbracira Labs
  DBA: Sai Products Private Limited`, `Arcnovi Labs née A Cure 3 All PC` — the true
  name is a substring inside an unrelated shell-company prefix.
- Social-handle-style names: `#integratedconstruction`, `#teamadvisors`,
  `énrightcastanedasettles.com`, `capitalsab.com`.
- Word-order scrambling and heavy typos: `Costner Inc Bsp` / `Csoosthnre Bsp Ínc` for
  `Costner Bsp Inc`.
- Address noise: abbreviation swaps (`Street`↔`St`, state name ↔ postal
  abbreviation, `महाराष्ट्र`↔`Maharashtra`↔`MH`), component reordering, and
  occasional fully empty addresses on S2/S3 side even when a match exists.
- French test records use French legal forms (`SARL`, `SASU`, `EURL`) and French
  administrative regions (`Nouvelle-Aquitaine`, `Hauts-de-France`, `Pays de la
  Loire`) instead of US-style `LLC`/`Inc` or Indian `Pvt Ltd`/PIN codes — see
  Section 11.

## 10. Noise mining — abbreviation dictionary

Full lists (top 60 each) are in `output/substitutions.txt`, mined from every
train matched pair by tokenizing on non-alphanumerics and keeping pairs where each
side has exactly one differing token. Highlights:

**Name**: `limited↔ltd` (162k / 26k), `inc↔incorporated` (39k), `corp↔corporation`
(7.2k/5.0k), `co↔company` (4.6k/4.5k), title/honorific swaps in Indian names
(`limited→shri/smt/sri/dr/mr`, likely proprietorship name variants).

**Address**: near-universal US abbreviation pairs (`road↔rd` 84k, `drive↔dr` 79k,
`street↔st` 77k, `avenue↔ave` 57k, `lane↔ln`, `court↔ct`, `boulevard↔blvd`), full
US state name ↔ 2-letter code pairs (`tx↔texas` 56k, `oh↔ohio`, `il↔illinois`, …),
and Indian state name ↔ abbreviation pairs (`maharashtra↔mh` 66k, `gujarat↔gj`,
`karnataka↔ka`, `telangana↔tg`, `rajasthan↔rj`, `haryana↔hr`, `kerala↔kl`/`keralam`,
`bihar↔br`, `punjab↔pb`, `orissa↔od`/`odisha`).

**Caveat**: a handful of high-count pairs (e.g. `private→center`, `inc→nc`,
`and→nd`) are mining artifacts, not real synonyms — they come from multi-word name
diffs that coincidentally reduce to exactly one differing token per side (e.g. a
name that also swapped a legal suffix elsewhere), or from single-character-drop
typos. This dictionary is a useful raw signal for future feature engineering, not
a hand-verified synonym table.

## 11. Answers to the required questions

**Is exclusivity safe to enforce (S2/S3 → at most one S1)?**
Yes, provisionally. It held with zero violations across all 7.6M matched ids in
train, so it looks like a deliberate invariant of how the dataset was generated
rather than a coincidence. We plan to enforce it downstream (e.g. via a
one-to-one assignment/dedup step on the S2/S3 side of the final matches), but since
test has no ground truth we cannot verify it holds there — we'll keep it as a
strong prior/constraint rather than an unconditional hard filter that could discard
a genuine match if the assumption is ever violated.

**Can blocking be restricted to the same country label?**
Yes. 0 of 7,638,365 matched pairs in train have differing country labels between
S1 and its match — this is the cleanest, cheapest blocking key we found. It must be
applied as an open-set string equality (block on `country_a == country_b`), not a
fixed `{US, India}` allowlist, so that France (and any other future label) is
handled automatically.

**How do France records look compared to US/India?**
Structurally the same 3-part `entity_id / business_name / business_address /
country` schema, but with distinct legal-form suffixes (`SARL`, `SASU`, `EURL`
instead of `LLC`/`Inc`/`Corp` or `Pvt Ltd`/`Private Limited`) and French
administrative region names instead of US state abbreviations or Indian
state/PIN codes. Addresses follow `<street>, <city>, <region>` with accented
characters (é, à) and French street-type words (`Rue`, `Boulevard`, `Impasse`,
`Chemin`, `Avenue`). No France-specific noise beyond what's already been generated
in US/India (abbreviations, empty addresses) was evident in the 8-record sample.

**Anything surprising or suspicious in the data?**
- Perfect exclusivity and perfect country agreement (both 0 violations) are
  stronger than typical real-world entity-resolution data — the generator appears
  to guarantee both properties, which is good news for later blocking/matching
  design but should not be over-relied on for the unseen France segment.
- ~26% distractor rate on both S2 and S3 means blocking must be reasonably precise:
  a naive "match everything in the same country" blocker would need to discriminate
  against a large true-negative pool.
- Several matched pairs share literally no alphanumeric tokens in the name field
  because one side is a Devanagari transliteration of the other — pure token/edit-
  distance similarity on the raw string will miss these; a transliteration-aware
  normalization step (e.g. `unidecode`, already pinned in `requirements.txt`) will
  likely be needed before name comparison.
- DBA/"née"/trade-name wrapping and social-handle-style names are a source of
  genuinely hard true positives that no simple string-similarity feature will catch
  without substring/contains-style matching.

## Setup

See `code/business_entity_resolution/README.md` for environment and run
instructions.
