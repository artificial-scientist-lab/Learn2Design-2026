# thelight — Round 1 evaluation data

## Summary

- Place: 36 of 43
- Mean feasible loss: 1.183956981
- Sample standard deviation: 1.035200899
- Standard error of the mean: 0.327359268
- Median feasible loss: 0.841415006
- Minimum / maximum run score: 0.359920565 / 3.742136510
- Mean time to best: 167.05 minutes
- Mean evaluations: 56416.9
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
