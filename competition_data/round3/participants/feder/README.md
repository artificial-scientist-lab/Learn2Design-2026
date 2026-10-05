# feder — Round 3 evaluation data

## Summary

- Place: 63 of 67
- Mean feasible loss: 1.485799452
- Sample standard deviation: 1.199179459
- Standard error of the mean: 0.379213841
- Median feasible loss: 0.965152000
- Minimum / maximum run score: 0.478591416 / 4.067830424
- Mean time to best: 150.30 minutes
- Mean evaluations: 56329.9
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
