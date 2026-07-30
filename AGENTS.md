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
`threefold_downside_ranking_v10`. It keeps the 193 continuous inputs and does not feed
industry codes or names into the Transformer. Historical industry snapshots
are joined as-of for two Exposure portfolio summaries and soft Top-10
concentration selection; the expensive industry-residual ranking objective is
disabled. Its tail head predicts a causal five-day holding-path loss event.
Strategy changes are calibrated in Ranking, Allocation, and Exposure stages
using strictly earlier, fully resolved OOF labels. Allocation and Exposure
heads are mandatory and must not be removed by ablations.
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
- Prefer replacing, deleting, or reusing existing logic over adding parallel
  implementations for the same behavior.
- Before committing, inspect production-code changes with `git diff --numstat`.
  If the added/deleted line ratio exceeds 1.5, explain why and make another
  consolidation and deduplication pass. This is a review trigger, not a reason
  to compress readable code. Count tests and documentation separately, while
  still avoiding duplicated implementations.
- Add a helper only when it removes duplication or makes the main flow
  materially shorter.

## Runtime observability

- Long-running preprocessing, calibration, evaluation, or training work must
  emit visible phase or progress updates. Do not leave an operation that may
  take more than 30 seconds silent when its major units can be counted or
  reported.
- Progress output must identify the active phase and advance at meaningful
  units so users can distinguish computation from a hang.
