# jiabao — Round 1 evaluation data

## Summary

- Place: 39 of 43
- Mean feasible loss: 2.602630024
- Sample standard deviation: 0.253587662
- Standard error of the mean: 0.080191460
- Median feasible loss: 2.623939416
- Minimum / maximum run score: 2.308760437 / 3.019803314
- Mean time to best: 118.24 minutes
- Mean evaluations: 167200.5
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
