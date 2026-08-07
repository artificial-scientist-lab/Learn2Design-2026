# Submission rules

## What you submit

A submission is a single `.py` file uploaded through the competition portal.
The file must contain exactly one class that subclasses
`OptimizationAlgorithm` from `dfbench.core`. The class may import any
pip-installable Python package (see [Dependencies](#dependencies) for the one
exception).

Competition entries are not submitted as public pull requests. This keeps
unreleased methods private until the organizers evaluate them, which reduces
copying between participants.

You may submit as often as you want. For each monthly evaluation, the organizers
evaluate the last submission received before that month's deadline.

If you need extra Python packages installed into the eval environment, ship a
`requirements.txt` next to your `.py` file. If you need to bundle weights or
data files, place them in the same directory and load them by relative path;
your whole submission directory is mounted read-only inside the container.

---

## Using the Differometor-30k dataset

Submissions are **encouraged** to use the publicly released [`Differometor-30k`](../dataset/README.md)
dataset for pre-training surrogates, initial-point models, warm-starts, or
any other purpose. There is no penalty for spending the entire budget on
surrogate inference while only calling `objective.value()` once. Just keep in
mind that at least one (feasible) logging entry needs to be provided for a run
to be valid.

---

## Time budget

Official budget per topology: exactly 4 hours of wall-clock time after
`objective.start_logging()`. JIT warmup before `objective.start_logging()` does
not count.

The full per-topology container runtime is capped at 4 h 30 min from the start
of the participant run. This includes any work before `objective.start_logging()`
and after the Objective budget is exhausted.

Concretely:

- The clock starts when your algorithm calls `objective.start_logging()`.
- Everything after that call counts toward the 4 hours, including Objective
  evaluation calls, batched / `vmap_*` variants, optimizer updates, surrogate
  inference, candidate generation, adaptation, and submission-side logging.
- Once the 4-hour wall-clock budget is exhausted, `objective.budget_exceeded`
  becomes `True`. Further calls to `objective.value` return immediately
  without re-evaluating; the best feasible loss found so far is what counts.
- JIT compilation time from warmup calls before `objective.start_logging()` is
  **not** counted against the 4 hours.
- Before calling `objective.start_logging()`, submissions may only perform setup
  that is independent of the specific evaluation problem instance. They must not
  use the provided `Objective`, its wrapped problem, topology, bounds, random
  samples, losses, gradients, auxiliary diagnostics, or any derived information
  to select, rank, train, adapt, filter, or otherwise improve candidate
  solutions before logging starts.

---

## Dependencies

Your submission may import any pip-installable Python package, with one exception:

- `import differometor` (and any submodule of it) is forbidden. All
  interaction with the simulator must go through the `Objective` instance you
  are handed.

The evaluation environment ships with a common base (`dfbench`, `differometor`,
`jax`, `jaxlib` CUDA 13, `numpy`, `scipy`, `optax`, `cma`, `cmaes`, `torch`,
`botorch`, `nevergrad`, `evosax`). If your submission needs anything else, ship
a `requirements.txt` next to your submission file and we will install it into
the eval environment before running you.

---

## Evaluation procedure

1. Your submission is placed in an isolated Docker container with no
   network access. The filesystem is writable; the container is ephemeral
   and discarded after the run, so anything you write is gone afterwards.
2. For each evaluation, your submission is run on 10 new hidden topologies. For each topology, your `optimize()` is called once with a
   fresh [`Objective`](dfbench/Objective-API-Reference.md), a fixed random seed, and the budget described above.
3. After the run, the result for that topology is the minimum loss among logged setups with `is_feasible=True`.
4. If your run has no feasible setup, that topology result is replaced by the best feasible loss from the organizers' random-search baseline on the same topology.
5. Your score for the month is the arithmetic mean of the 10 topology results.

---

## Prize eligibility and source disclosure

Submissions ranking in the top 10 of the final private leaderboard must provide
full source code to the organizers for manual review. Code need not be made
public, but the organizers must be able to verify that no banned operations were
used. Prize money is withheld until review is complete.

---

## Disqualification criteria

- Encoding knowledge of private topology specifications in any form.
- Using information from the provided evaluation `Objective` or problem instance
  to improve candidate solutions before `objective.start_logging()`.
- Submitting work that is not your own without attribution.
- Violations of the [Code of Conduct](../CODE_OF_CONDUCT.md).
