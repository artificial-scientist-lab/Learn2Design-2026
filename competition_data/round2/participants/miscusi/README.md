# miscusi — Round 2 evaluation data

## Summary

- Place: 19 of 60
- Mean feasible loss: 0.309022265
- Sample standard deviation: 0.207414199
- Standard error of the mean: 0.065590129
- Median feasible loss: 0.382591502
- Minimum / maximum run score: -0.216032454 / 0.509044130
- Mean time to best: 230.05 minutes
- Mean evaluations: 152609.6
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
