# nityawav — Round 1 evaluation data

## Summary

- Place: 33 of 43
- Mean feasible loss: 0.976708135
- Sample standard deviation: 1.114817758
- Standard error of the mean: 0.352536329
- Median feasible loss: 0.455114445
- Minimum / maximum run score: 0.316300986 / 3.715063057
- Mean time to best: 162.07 minutes
- Mean evaluations: 56475.3
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
