# s_boehler — Round 3 evaluation data

## Summary

- Place: 17 of 67
- Mean feasible loss: 0.179214134
- Sample standard deviation: 0.242347303
- Standard error of the mean: 0.076636946
- Median feasible loss: 0.262472669
- Minimum / maximum run score: -0.314830877 / 0.409697842
- Mean time to best: 237.31 minutes
- Mean evaluations: 140145.6
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
