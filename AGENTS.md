# Feature configuration note

Feature-set names retain their `158+39` lineage and must state any removals. The
former `158+39` set has 191 actual continuous inputs; after removing 20 exactly
redundant features it has 171. The active `158+39_reduced25` experiment has 166
inputs after removing a further five linear duplicates (`high_low_spread`, `open_close_spread`, `high_close_spread`,
`low_close_spread`, and `kdj_j`). Keep each experiment's model directory separate,
because checkpoints from different input dimensions are incompatible.

## Git workflow

- Keep commits small and focused, and push completed commits promptly.
- Create a descriptive Git tag for every new model/code version before comparing or sharing its results.
