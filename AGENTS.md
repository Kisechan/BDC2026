# Feature configuration note

Feature-set names retain their `158+39` lineage and must state any removals. The
former `158+39` set has 191 actual continuous inputs; after removing 20 exactly
redundant features it has 171. The `158+39_reduced25` experiment has 166 inputs
after removing a further five linear duplicates (`high_low_spread`,
`open_close_spread`, `high_close_spread`, `low_close_spread`, and `kdj_j`).
The active `158+39_reduced25_relmarket12` experiment has 178 inputs after adding
seven causal cross-sectional percentile features and five causal market-state
features. The active `158+39_reduced25_relmarket12_risk15` experiment has 193
inputs after adding seven short-horizon/downside stock percentiles and eight
market-pressure features. The active model experiment is
`nested_oof_diverse_tailregime_v6_decay5y`; it keeps the 1/3/5-day soft risk
heads and adds a five-day tail-event head, while retaining the market regime
gate, Allocation Head, and Exposure Head. Risk-head gradients remain isolated
from Ranking, five-year ranking samples remain recency weighted, strategy
promotion uses nested cross-fitted OOF, and Top-5 construction enforces a
20/60-day correlation-cluster cap.
Keep each experiment's
model directory separate, because
checkpoints from different architectures or input dimensions are incompatible.

## Git workflow

- Make production implementation changes directly on `main`; do not create or
  switch to feature branches unless the user explicitly requests one.
- Keep commits small and focused, and push completed commits promptly.
- Create a descriptive Git tag for every new model/code version before comparing or sharing its results.
- Use patch tags (`v1.N.x`) for training acceleration, bug fixes,
  compatibility work, and other small behavior adjustments. Use the next minor
  version (`v1.(N+1)`) only for a new model objective, validation protocol, or
  major portfolio algorithm.
- Every tag must accurately describe the scope of its changes; ordinary
  optimizations must not advance the minor version.

## File organization

- Avoid adding new files unless the change cannot reasonably fit an existing
  module. Prefer extending the closest existing module and document why any new
  file is necessary.
