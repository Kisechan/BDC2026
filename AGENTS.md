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
`regime_lambdarank_staged_v3_5y`; it adds 1/3-day risk heads, a market
regime gate, LambdaRank@5, and staged head training. Keep each experiment's
model directory separate, because
checkpoints from different architectures or input dimensions are incompatible.

## Git workflow

- Make production implementation changes directly on `main`; do not create or
  switch to feature branches unless the user explicitly requests one.
- Keep commits small and focused, and push completed commits promptly.
- Create a descriptive Git tag for every new model/code version before comparing or sharing its results.

## File organization

- Avoid adding new files unless the change cannot reasonably fit an existing
  module. Prefer extending the closest existing module and document why any new
  file is necessary.
