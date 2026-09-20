# boundedaccord — Round 2 evaluation data

## Summary

- Place: 38 of 60
- Mean feasible loss: 0.572366947
- Sample standard deviation: 0.886658695
- Standard error of the mean: 0.280386098
- Median feasible loss: 0.408894223
- Minimum / maximum run score: -0.447399294 / 2.947696958
- Mean time to best: 228.48 minutes
- Mean evaluations: 55106.4
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
