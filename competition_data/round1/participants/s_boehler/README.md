# s_boehler — Round 1 evaluation data

## Summary

- Place: 31 of 43
- Mean feasible loss: 0.885837374
- Sample standard deviation: 1.022170146
- Standard error of the mean: 0.323238582
- Median feasible loss: 0.522416052
- Minimum / maximum run score: 0.432341805 / 3.757507436
- Mean time to best: 164.77 minutes
- Mean evaluations: 56433.1
- Failed runs: 0
- Random-search fallback runs: 0

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoint values are marked as exact when the final best was
already known to have been reached. Otherwise they are sampled upper bounds
derived from one stored candidate every 25 evaluations. The 240-minute values
are the exact official run scores.
