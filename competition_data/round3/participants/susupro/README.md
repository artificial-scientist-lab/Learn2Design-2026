# susupro — Round 3 evaluation data

## Summary

- Place: 31 of 67
- Mean feasible loss: 0.316542266
- Sample standard deviation: 0.109217801
- Standard error of the mean: 0.034537701
- Median feasible loss: 0.348107644
- Minimum / maximum run score: 0.172336105 / 0.428098582
- Mean time to best: 222.17 minutes
- Mean evaluations: 138802.4
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
