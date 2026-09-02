# jan_vandelden — Round 1 evaluation data

## Summary

- Place: 8 of 43
- Mean feasible loss: 0.330130459
- Sample standard deviation: 0.201998358
- Standard error of the mean: 0.063877489
- Median feasible loss: 0.392017579
- Minimum / maximum run score: -0.229153958 / 0.479621555
- Mean time to best: 187.14 minutes
- Mean evaluations: 133534.4
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
