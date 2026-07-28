# Feature configuration note

Feature-set names retain their `158+39` lineage and must state any removals. The
former `158+39` set has 191 actual continuous inputs; after removing 20 exactly
redundant features it has 171. The `158+39_reduced25` experiment has 166 inputs
after removing a further five linear duplicates (`high_low_spread`,
`open_close_spread`, `high_close_spread`, `low_close_spread`, and `kdj_j`).
The active `158+39_reduced25_relmarket12` experiment has 178 inputs after adding
seven causal cross-sectional percentile features and five causal market-state
features. Keep each experiment's model directory separate, because checkpoints
from different input dimensions are incompatible.

## Git workflow

- Keep commits small and focused, and push completed commits promptly.
- Create a descriptive Git tag for every new model/code version before comparing or sharing its results.

## File organization

- Avoid adding new files unless the change cannot reasonably fit an existing
  module. Prefer extending the closest existing module and document why any new
  file is necessary.
