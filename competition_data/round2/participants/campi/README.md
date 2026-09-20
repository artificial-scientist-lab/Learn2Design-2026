# campi — Round 2 evaluation data

## Summary

- Place: 28 of 60
- Mean feasible loss: 0.393844908
- Sample standard deviation: 0.187868394
- Standard error of the mean: 0.059409202
- Median feasible loss: 0.420149980
- Minimum / maximum run score: -0.060100753 / 0.674689279
- Mean time to best: 68.88 minutes
- Mean evaluations: 54946.4
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
