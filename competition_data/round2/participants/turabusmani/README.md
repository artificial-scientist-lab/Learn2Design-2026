# turabusmani — Round 2 evaluation data

## Summary

- Place: 59 of 60
- Mean feasible loss: 3.875208047
- Sample standard deviation: 0.343727407
- Standard error of the mean: 0.108696150
- Median feasible loss: 3.902582354
- Minimum / maximum run score: 3.166420615 / 4.457761424
- Mean time to best: 130.25 minutes
- Mean evaluations: 77547.6
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
