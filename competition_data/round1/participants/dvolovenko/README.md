# dvolovenko — Round 1 evaluation data

## Summary

- Place: 19 of 43
- Mean feasible loss: 0.552775091
- Sample standard deviation: 0.211963594
- Standard error of the mean: 0.067028774
- Median feasible loss: 0.504037767
- Minimum / maximum run score: 0.239903006 / 0.981225999
- Mean time to best: 189.17 minutes
- Mean evaluations: 56238.6
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
