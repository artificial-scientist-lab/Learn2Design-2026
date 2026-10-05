# Learn2Design 2026 Round 3 evaluation data

This public release provides detailed evaluation statistics for the 67 participants
on the Round 3 leaderboard. It includes the ten run scores, aggregate uncertainty, efficiency
and feasibility statistics, and selected convergence checkpoints.

## Contents

- `leaderboard.csv`: aggregate score and run statistics for every participant.
- `runs.csv`: exact final statistics for all 670 participant runs.
- `checkpoints.csv`: per-run values at 15, 30, 60, 120, 180 and 240 minutes.
- `checkpoint_summary.csv`: checkpoint mean, sample standard deviation and SEM.
- `seeds.csv`: the shared Round 3 topology and optimizer seed pairs.
- `participants/<participant_id>/`: an individual report and convergence plot.

## Shared seeds

All participants and organizer baselines used the same ten seed pairs. Round 3
public-evaluation topologies are retired; later rounds and the final evaluation
use different hidden topologies.

| Run | Seed index | Topology seed | Optimizer seed |
|---:|---:|---:|---:|
| 1 | 0 | 98591154 | 1791403674 |
| 2 | 1 | 650941223 | 887131975 |
| 3 | 2 | 1211049875 | 1078728795 |
| 4 | 3 | 1363760596 | 1142570359 |
| 5 | 4 | 1795982962 | 398006495 |
| 6 | 5 | 196619468 | 773385860 |
| 7 | 6 | 239637960 | 1603749681 |
| 8 | 7 | 1410548495 | 872722144 |
| 9 | 8 | 791199028 | 1571924013 |
| 10 | 9 | 1082220220 | 687175312 |

## Metric definitions and precision

Lower loss is better. The final score is the mean over ten runs. Standard deviations
use the sample definition (ddof=1), and SEM is standard deviation divided by the
square root of the number of available runs. At intermediate checkpoints that
number may be smaller than ten; runs_with_value and missing_run_count show it.
The convergence figures plot the checkpoint means and one SEM. Connecting lines
are visual guides, not continuous measurements.

Intermediate checkpoints are conservative upper bounds unless marked exact. A feasible improvement is included only when it is known to have occurred by that Objective time. Missing values are left blank. The 240 minute values are exact official run scores, including any explicitly marked RandomSearch fallback.

Breadcrumbs feasible improvement records were matched to the next standard
Objective progress timestamp. Rounded timestamps were rounded upward, so an
improvement is never assigned earlier than it can be verified. Total elapsed
wall time from run start provides a second conservative bound. Where available,
the exact final best timestamp from the Objective history takes precedence.
Times use the supplied Objective clock rather than total job wall time.

Feasibility counts and fractions describe the retained Objective history observations. Batched calls may retain only their lowest loss candidate, so this is not the fraction of all evaluated candidates. The observation count is provided as the denominator; evaluation_count separately includes all admitted candidates.

Time to best is averaged only across runs with an observed feasible result;
time_to_best_run_count gives that denominator. time_to_best_precision distinguishes
exact values from conservative upper bounds. Efficiency uses the actual measured
Objective duration, including runs that stopped before the four hour limit.

All 670 selected runs completed without a runtime failure. One run, doris_south
run 3, found no feasible candidate and receives the RandomSearch score from the
same seed pair. Its own performance and the substituted score are separate in
runs.csv. The failed_runs field counts runtime failures; fallback_runs counts
score substitutions. An early stop is not a runtime failure.

Schema version 1.1 explicitly identifies feasibility history observations and
their denominator, rather than treating them as a count of all batch candidates.
