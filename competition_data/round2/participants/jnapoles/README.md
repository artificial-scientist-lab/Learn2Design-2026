# jnapoles — Round 2 evaluation data

## Summary

- Place: 55 of 60
- Mean feasible loss: 2.490297477
- Sample standard deviation: 1.821948091
- Standard error of the mean: 0.576150575
- Median feasible loss: 2.390909020
- Minimum / maximum run score: 0.543238216 / 4.483221849
- Mean time to best: 138.57 minutes
- Mean evaluations: 55282.9
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
derived from one stored candidate every 10 evaluations. The 240-minute values
are the exact official run scores.
