```
actual OOF (tuned rule)                                 macro F0.5 0.9504  (India 0.9281, US 0.9652)  P 0.9741 R 0.8989
oracle given candidates (blocking ceiling)              macro F0.5 0.9625  (India 0.9390, US 0.9782)  P 0.9817 R 0.9205

LOSS to blocking (1 - ceiling):          0.0375
LOSS to model + decision (ceiling - act): 0.0121

true pairs 543162: predicted 485567, missed inside candidates 12072, never a candidate 45523 (8.38%); false positives 2547

by true-set size: share of S1 whose true matches are ALL candidates
  n_true 0    :     8749 S1, all-in-candidates 1.0000
  n_true 1    :     8450 S1, all-in-candidates 0.9102
  n_true 2-3  :    64567 S1, all-in-candidates 0.8302
  n_true 4-6  :    68987 S1, all-in-candidates 0.7489
  n_true 7-10 :     6215 S1, all-in-candidates 0.6763
  n_true 11+  :        3 S1, all-in-candidates 0.3333
```
