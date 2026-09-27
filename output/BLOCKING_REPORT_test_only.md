

=== Part F: full test candidate generation ===
Test S1 entities: 1732544; with >=1 candidate: 1713536 (98.90%); with 0 candidates: 19008; candidate pairs (top-40): 55686195

Candidates per S1 by country:
  US: n_S1=663106, mean=32.5, median=40, max=40, zero-candidate S1s=817 (0.12%)
  India: n_S1=809986, mean=32.4, median=40, max=40, zero-candidate S1s=17386 (2.15%)
  France: n_S1=259452, mean=30.2, median=39, max=40, zero-candidate S1s=805 (0.31%)

(France has no training labels; a much higher zero-candidate or low-candidate rate for France above than US/India would mean the mined dictionaries/keys, which are all trained on US/India pairs, generalize poorly to French names/addresses.)
(output/candidate_pairs.tsv is written by train_model.py together with matching_results.tsv, from exactly the pairs the model scores.)