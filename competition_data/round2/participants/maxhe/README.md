# maxhe — Round 2 evaluation data

## Summary

- Place: 42 of 60
- Mean feasible loss: 0.716599899
- Sample standard deviation: 0.328054853
- Standard error of the mean: 0.103740053
- Median feasible loss: 0.567478493
- Minimum / maximum run score: 0.455079631 / 1.507657452
- Mean time to best: 238.57 minutes
- Mean evaluations: 112683.6
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
