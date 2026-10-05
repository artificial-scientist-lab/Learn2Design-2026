# simbuddy — Round 3 evaluation data

## Summary

- Place: 58 of 67
- Mean feasible loss: 0.989593179
- Sample standard deviation: 0.899138505
- Standard error of the mean: 0.284332561
- Median feasible loss: 0.590233288
- Minimum / maximum run score: 0.309335443 / 2.732332142
- Mean time to best: 232.24 minutes
- Mean evaluations: 56352.1
- Failed runs: 0
- Random-search fallback runs: 0

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoints are conservative upper bounds unless marked exact. A feasible improvement is included only when it is known to have occurred by that Objective time. Missing values are left blank. The 240 minute values are exact official run scores, including any explicitly marked RandomSearch fallback.

Feasibility counts and fractions describe the retained Objective history observations. Batched calls may retain only their lowest loss candidate, so this is not the fraction of all evaluated candidates. The observation count is provided as the denominator; evaluation_count separately includes all admitted candidates.
