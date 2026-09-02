# Learn2Design 2026 Round 1 evaluation data

This public release provides detailed evaluation statistics for all 43 Round 1
participants. It includes the ten run scores,
aggregate uncertainty, efficiency and feasibility statistics, and selected
convergence checkpoints.

## Contents

- `leaderboard.csv`: aggregate score and run statistics for every participant.
- `runs.csv`: exact final statistics for all 430 participant runs.
- `checkpoints.csv`: per-run values at 15, 30, 60, 120, 180 and 240 minutes.
- `checkpoint_summary.csv`: checkpoint mean, sample standard deviation and SEM.
- `seeds.csv`: the shared Round 1 topology and optimizer seed pairs.
- `participants/<participant_id>/`: an individual report and convergence plot.

## Checkpoint precision

The common recorder saved parameters, loss and feasibility once every 25
candidate evaluations. Intermediate checkpoint values are therefore reported
as sampled upper bounds unless the exact final best is known to have occurred
before that checkpoint. The 240-minute values are always the exact official run
scores, including the documented random-search fallback when needed.

## Shared seeds

All participants and organizer baselines used the same ten seed pairs. Round 1
public-evaluation topologies are retired; later rounds and the final evaluation
use different hidden topologies.

| Run | Seed index | Topology seed | Optimizer seed |
|---:|---:|---:|---:|
| 1 | 0 | 419164888 | 771850990 |
| 2 | 1 | 910574073 | 424237334 |
| 3 | 2 | 891065062 | 929044270 |
| 4 | 3 | 720200356 | 174477668 |
| 5 | 4 | 654200764 | 541268051 |
| 6 | 5 | 1038908541 | 1844528431 |
| 7 | 6 | 402803193 | 1082025484 |
| 8 | 7 | 842892044 | 1072187562 |
| 9 | 8 | 136269361 | 267997051 |
| 10 | 9 | 2139912011 | 1732111275 |

Lower loss is better. Standard deviations use the sample definition (`ddof=1`),
and SEM is `std / sqrt(10)` for complete final results.
