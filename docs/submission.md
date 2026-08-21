# Submission rules

## Submission format

Upload a ZIP archive to the [submission portal](https://submit.learn2design2026.com/)
containing the following two mandatory files at its root. Do not place them
inside an enclosing directory.

### `submission.py` — mandatory

This file must define exactly one Python class that subclasses
`dfbench.OptimizationAlgorithm`. The evaluator calls:

```python
MyAlgorithm.optimize(...)
```

### `requirements.txt` — mandatory

List all required Python packages using [PEP 508](https://peps.python.org/pep-0508/)
syntax, with one dependency per line:

```text
<package1>==1.2.3
<package2>==4.5.6
```

### Additional files — optional

You may include other files, such as Python modules or pretrained neural-network
weights, in the archive.

The evaluation script runs from the root of the extracted archive, so your code
can access these files using relative paths.

Competition entries are not submitted as public pull requests. This keeps
unreleased methods private until the organizers evaluate them, which reduces
copying between participants.

You may submit as often as you want. For each monthly evaluation, the organizers
evaluate only the last submission received before that month's deadline. Earlier
submissions are not evaluated or scored.

---

## Using the Differometor-30k dataset

Submissions are **encouraged** to use the publicly released [`Differometor-30k`](../dataset/README.md)
dataset for pre-training surrogates, initial-point models, warm-starts, or
any other purpose. There is no penalty for spending the entire budget on
surrogate inference while only calling `objective.value()` once. Just keep in
mind that at least one (feasible) logging entry needs to be provided for a run
to be valid.

---

## Evaluation hardware

Each participant run is allocated:

- 1× NVIDIA H100 SXM5 GPU (Hopper, 80 GB HBM3)
- 16 vCPUs on an Intel Xeon Platinum 8468 host
- 200 GiB RAM
- 40 GiB SSD
- Ubuntu 22.04 LTS with CUDA 12

---

## Time budget

Official budget per topology: exactly 4 hours of wall-clock time after
`objective.start_logging()`. Objective-provided JIT warmup before
`objective.start_logging()` does not count.

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
- JIT compilation time from Objective-provided `warmup_*()` calls before
  `objective.start_logging()` is not counted against the 4 hours.
- Before calling `objective.start_logging()`, submissions may inspect documented
  public context such as `bounds`, `n_params`, `problem_name`, `problem_spec`,
  and `optimization_pairs`. They may generate random parameters, inspect budget
  configuration, use public pre-run setters, initialize candidates or models
  from that context, and call Objective-provided `warmup_*()` methods.
- Result-producing evaluation methods, raw callable getters such as
  `value_function()`, and `log_evaluation()` are unavailable before logging and
  raise `RuntimeError`. Compiling a custom raw-callable evaluation path therefore
  counts against the 4-hour budget. Accessing private internals or the wrapped
  problem to bypass this lifecycle is forbidden.

---

## Dependencies

Your submission may import any pip-installable Python package, with one exception:

- `import differometor` (and any submodule of it) is forbidden. All
  interaction with the simulator must go through the `Objective` instance you
  are handed.

The evaluation environment ships with a common base (`dfbench`, `differometor`,
`jax`, `jaxlib` CUDA 12, `numpy`, `scipy`, `optax`, `cma`, `cmaes`, `torch`,
`botorch`, `nevergrad`, `evosax`). The evaluator installs the packages listed in
the mandatory root-level `requirements.txt` before running your algorithm.

---

## Evaluation procedure

1. Your ZIP archive is extracted into an isolated Docker container. The
   evaluator verifies the two mandatory root files and installs the packages
   listed in `requirements.txt`.
2. The evaluator runs from the extracted archive root with no network access
   and loads the algorithm class from `submission.py`. The filesystem is
   writable; the container is ephemeral and discarded after the run, so
   anything you write is gone afterwards.
3. For each evaluation, your submission is run on 10 new hidden topologies. For
   each topology, your `optimize()` is called once with a fresh
   [`Objective`](dfbench/Objective-API-Reference.md), a fixed random seed, and
   the budget described above.
4. After the run, the result for that topology is the minimum loss among logged
   setups with `is_feasible=True`.
5. If your run has no feasible setup, that topology result is replaced by the
   best feasible loss from the organizers' random-search baseline on the same
   topology.
6. Your score for the month is the arithmetic mean of the 10 topology results.

---

## Prize eligibility and source disclosure

Submissions ranking in the top 10 of the final private leaderboard must provide
full source code to the organizers for manual review. Code need not be made
public, but the organizers must be able to verify that no banned operations were
used. Prize money is withheld until review is complete.

---

## Disqualification criteria

- Encoding knowledge of private topology specifications in any form.
- Bypassing the `Objective` limitations to evaluate, obtain raw evaluation
  callables early, or record results before `objective.start_logging()`.
- Submitting work that is not your own without attribution.
- Violations of the [Code of Conduct](../CODE_OF_CONDUCT.md).
