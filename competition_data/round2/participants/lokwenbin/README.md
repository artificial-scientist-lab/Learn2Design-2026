# lokwenbin — Round 2 evaluation data

## Summary

- Place: 57 of 60
- Mean feasible loss: 2.938542128
- Sample standard deviation: 0.566709037
- Standard error of the mean: 0.179209133
- Median feasible loss: 3.032420476
- Minimum / maximum run score: 1.995390542 / 4.021431966
- Mean time to best: 124.51 minutes
- Mean evaluations: 55535.3
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
