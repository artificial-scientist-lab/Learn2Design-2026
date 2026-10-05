# nancy — Round 3 evaluation data

## Summary

- Place: 34 of 67
- Mean feasible loss: 0.327558956
- Sample standard deviation: 0.171628078
- Standard error of the mean: 0.054273564
- Median feasible loss: 0.404024506
- Minimum / maximum run score: -0.099493153 / 0.483134599
- Mean time to best: 235.15 minutes
- Mean evaluations: 55557.9
- Failed runs: 0
- Random-search fallback runs: 0

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoints are conservative upper bounds unless marked exact.
The 240-minute values are official run scores; missing values and fallbacks
are marked.
