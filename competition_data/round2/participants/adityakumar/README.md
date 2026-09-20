# adityakumar — Round 2 evaluation data

## Summary

- Place: 60 of 60
- Mean feasible loss: 3.914990099
- Sample standard deviation: 0.689959896
- Standard error of the mean: 0.218184476
- Median feasible loss: 4.161924118
- Minimum / maximum run score: 2.829768565 / 4.556924217
- Mean time to best: 120.16 minutes
- Mean evaluations: 95792.5
- Failed runs: 0
- Random-search fallback runs: 0

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoint values are marked as exact when the final best was
already known to have been reached. Otherwise they are sampled upper bounds.
The 240-minute values are the exact official run scores.
