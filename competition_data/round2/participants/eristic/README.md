# eristic — Round 2 evaluation data

## Summary

- Place: 58 of 60
- Mean feasible loss: 3.854151408
- Sample standard deviation: 0.769468668
- Standard error of the mean: 0.243327358
- Median feasible loss: 4.132112045
- Minimum / maximum run score: 2.463738027 / 4.506899458
- Mean time to best: 104.12 minutes
- Mean evaluations: 80584.1
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
