# Dense (transformer) retrieval: go/no-go on the 300k train sample

Encoder: multilingual-e5-small (118M, MIT) fine-tuned 1 epoch on 587k train pairs + hardest blocking non-match (CachedMNRL, batch 128, RTX 3050 4GB, 1h14).

```
[dense] recall on 1037463 true pairs of the 300k sample:
  key candidates (current pipeline): 91.67%
  dense top-5: 90.74%   union(keys, dense top-5): 98.06%
  dense top-10: 97.35%   union(keys, dense top-10): 99.14%
  dense top-20: 98.57%   union(keys, dense top-20): 99.51%
  dense top-30: 98.96%   union(keys, dense top-30): 99.63%
  union by country India: 99.68%
  union by country US: 99.59%
```

Loss diagnosis of the v3 model (before dense): see LOSS_DIAGNOSIS.md -- 0.0375 of the 0.0496 gap to 1.0 was blocking.
