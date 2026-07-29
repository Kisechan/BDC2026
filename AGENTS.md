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
market-pressure features. The active strategy experiment is
`nested_oof_forward_policy_v7_1`. It reuses the model, scaler, stock mapping,
and OOF artifacts from `nested_oof_diverse_tailregime_v6_decay5y`; its own
directory contains policy and report artifacts only. Strategy changes are
calibrated in Ranking, Allocation, and Exposure stages using strictly earlier,
fully resolved OOF labels. The bounded correlation-cluster option only
replaces names within the raw Top-10 and falls back to the original Top-5 when
infeasible.
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

## Runtime observability

- Long-running preprocessing, calibration, evaluation, or training work must
  emit visible phase or progress updates. Do not leave an operation that may
  take more than 30 seconds silent when its major units can be counted or
  reported.
- Progress output must identify the active phase and advance at meaningful
  units so users can distinguish computation from a hang.
