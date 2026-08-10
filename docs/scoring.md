# Scoring

## Per-run score

For a single topology, your run's score is the best physically feasible
(`is_feasible=True`) loss reached within the 4-hour wall-clock budget after
`objective.start_logging()`:

$$
s_{\text{run}} = \min_{t \in [0,\, T_{\text{budget}}],\;\mathrm{is\_feasible}(\theta_t)} \mathcal{L}(\theta_t)
$$

where $\mathcal{L}$ is the [sensitivity-derived loss](dfbench/FAQ.md#what-objective-function-is-optimized-for-uifo) returned by
[`Objective.value`](dfbench/Objective-API-Reference.md#single-point-evaluation) and $\theta_t$ are the parameters evaluated at time $t$.
Lower is better.

If a run contains no physically feasible setup, the run score is replaced by the best
feasible loss found by the organizers' random-search baseline on that same
topology.

`NaN` losses are coerced to `+inf` before aggregation.

---

## Per-month score

Each public-leaderboard evaluation runs on its own 10 new hidden topologies. The
monthly score is the arithmetic mean of the 10 per-run scores:

$$
S_{\text{month}} = \frac{1}{10} \sum_{i=1}^{10} s_{\text{run}}^{(i)}
$$

Loss magnitudes are comparable across different topologies.

You may [submit through the competition portal](https://submit.learn2design2026.com/)
as often as you want; each monthly leaderboard evaluates the last submission
received before that month's deadline.

---

## Final score

The final leaderboard is computed identically, but on its own 10 **private**
hidden topologies, which are never published. Your final score is

$$
S_{\text{final}} = \frac{1}{10} \sum_{i=1}^{10} s_{\text{run}}^{(i, \text{private})}
$$

The submission used for the final evaluation is the last submission received
before the final deadline.

---

## Tie-breaking

If two submissions are within machine precision on the final score, the
tie-breaker is, in order:

1. Lower mean wall-clock time to reach the best feasible loss (faster algorithm wins).
2. Lower mean number of `Objective.value` calls (fewer evaluations wins).
3. Earlier submission timestamp.

---

## Anti-cheating policy

See [submission.md](submission.md#prize-eligibility-and-source-disclosure)
for the source-disclosure requirement for top-10 finishers, and the
[Disqualification criteria](submission.md#disqualification-criteria) section.
