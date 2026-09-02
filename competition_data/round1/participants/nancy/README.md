# nancy — Round 1 evaluation data

## Summary

- Place: 38 of 43
- Mean feasible loss: 2.220958393
- Sample standard deviation: 0.976180355
- Standard error of the mean: 0.308695333
- Median feasible loss: 1.935653508
- Minimum / maximum run score: 1.521504764 / 4.945954879
- Mean time to best: 224.42 minutes
- Mean evaluations: 55937.8
- Failed runs: 0
- Random-search fallback runs: 1

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
