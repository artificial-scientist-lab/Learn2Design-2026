# azhang81 — Round 2 evaluation data

## Summary

- Place: 46 of 60
- Mean feasible loss: 0.860922421
- Sample standard deviation: 1.109810163
- Standard error of the mean: 0.350952789
- Median feasible loss: 0.448188402
- Minimum / maximum run score: 0.308415141 / 3.981961494
- Mean time to best: 195.13 minutes
- Mean evaluations: 55022.2
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
