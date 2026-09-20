# thatdaveguy — Round 2 evaluation data

## Summary

- Place: 54 of 60
- Mean feasible loss: 2.478268440
- Sample standard deviation: 0.502784761
- Standard error of the mean: 0.158994502
- Median feasible loss: 2.361544618
- Minimum / maximum run score: 2.005237730 / 3.773098763
- Mean time to best: 103.10 minutes
- Mean evaluations: 52100.4
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
