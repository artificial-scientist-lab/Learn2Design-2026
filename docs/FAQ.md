# Learn2Design-2026 & dfbench FAQ

## Competition Overview

### What is the goal of the Learn2Design competition?
Learn2Design focuses on the automated design of highly sensitive gravitational-wave detectors under experimental constraints. Given a quasi-universal interferometer (UIFO) [topology](dfbench_overview.md#explanation-of-topology), your task is to write an algorithm that tunes roughly 200 continuous parameters within a fixed compute budget to minimize a simulator-defined [loss](dfbench/FAQ.md#what-objective-function-is-optimized-for-uifo).

### What data and resources are provided?
Participants are provided with two major resources:
* **Differometor:** An auto-differentiable JAX-based physics simulator.
* **Precomputed Design Corpus:** Approximately 30,000 high-quality detector designs generated through a 360,000 GPU-hour exploration campaign; see the [dataset README](../dataset/README.md).
* **Baselines:** Over 10 baseline optimization strategies spanning gradient-based, evolutionary, surrogate-based, and generative methods.

### What is the prize for winning?
Algorithms will be ranked by average performance on hidden evaluation tasks:
* First place: EUR 10,000
* Second place: EUR 6,000
* Third place: EUR 3,000

Additionally, there are two special prizes of EUR 3,000 each. These are judged by a committee and awarded for:

* The most surprising or creative solution
* The simplest strong-performing solution

---

## Objective Function & UIFO

### What is a topology?
A "[topology](dfbench_overview.md#explanation-of-topology)" fixes the discrete choice and placement of optical components (like lasers, squeezers, detectors, and beam splitters) for a Quasi-Universal Interferometer (UIFO). When optimizing a given topology, you only tune the continuous parameters attached to it, such as laser power, mirror reflectivity, and grid distances.

<img src="../media/UIFO.png" alt="UIFO Topology Picture" width="520" />

### How can I directly define the topology of a UIFO (not through seed)?
There are two ways besides the `topology_seed` to set the UIFO topology when initializing a `UIFOProblem`:
1. **`topology` string:** A compact encoding using single-character codes (e.g., `topology="AECGCCHEG-SLLSSHLLLLS"`). The first 9 characters encode the center components. The other 12 encode the boundary components. See [Explanation of "Topology"](dfbench_overview.md#explanation-of-topology) for the exact encoding scheme.
2. **`centers` + `boundaries` dictionaries:** Explicit component placement using coordinate dictionaries that match Differometor's native format (e.g., `centers={"11": ("beamsplitter", "left")}, boundaries={"01": "squeezer"}`). Refer to the [Differometor documentation](https://github.com/artificial-scientist-lab/Differometor/blob/6aa8592e4e7c9c48ff925e423aae5649185f4e88/differometor/setups.py#L752) for the exact format.

### How long does the objective function evaluation take?
Runtime depends on the topology, batching strategy, and implementation. Official evaluations use the [standard H100 evaluation VM](submission.md#evaluation-hardware); benchmark representative UIFO topologies locally and plan against the official wall-clock budget.

### How long does it take to JIT-compile the objective function?
JIT compilation can take several minutes and varies with the compiled function. Because compilation is expensive, we provide `warmup_*()` methods (e.g., `objective.warmup_value()`) so you can compile the functions for free before the official evaluation clock starts via `objective.start_logging()`.

Before logging, you may also inspect documented context such as bounds,
dimension, `problem_spec`, and `optimization_pairs`, generate random parameters,
and use public pre-run setters. Result-producing methods, raw callable getters,
and `log_evaluation()` raise `RuntimeError` until `start_logging()` has run.
Custom raw-callable compilation is therefore timed.

---

## Getting Started & Submissions

### What exactly do I submit?
You submit a ZIP archive, not a fixed design or parameter vector. At the archive
root it must contain `submission.py`, defining exactly one
`dfbench.OptimizationAlgorithm` subclass, and `requirements.txt`, listing
dependencies one per line using `<pacakge1>==1.2.3` syntax. Additional files, such as
supporting modules or pretrained weights, may be included in the archive and
accessed using relative paths. Upload the archive through the
[submission portal](https://submit.learn2design2026.com/) and see the
[submission rules](submission.md#submission-format) for complete requirements.

### What is the `Objective` class and why must I use it?
The [`Objective`](dfbench/Objective-API-Reference.md) class is the sole interface between your algorithm and the underlying physics simulation. It transparently handles:
* Dispatching to bounded or unbounded objective functions.
* Preparing `jax.grad`, `jax.hessian`, `jax.value_and_grad`, and batched `vmap` variants.
* Recording synchronized histories of losses, gradients, parameters, and timestamps.
* Enforcing wall-clock time and evaluation-count budgets fairly.
* Providing deterministic random sampling via a splittable JAX PRNG. (Set a seed via `objective.set_seed(seed)` for reproducibility of random params generation).

By requiring all algorithms to communicate through `Objective`, we ensure that benchmarking is completely standardized and reproducible.

---

## Algorithms & Optimization

### Should I optimize in bounded or unbounded space?
It depends entirely on your chosen algorithmic approach:
* **Bounded Space (`unbounded=False`):** Best for Evolutionary and Surrogate-based algorithms (like Random Search, PSO, CMA-ES, or BO) because populations and acquisitions naturally respect physical box constraints.
* **Unbounded Space (`unbounded=True`):** Best for Gradient-based methods (like Adam or L-BFGS). Optimization in clipped-bounded space can produce zero gradients at boundaries. Setting `unbounded=True` applies a sigmoid transform so gradients remain smooth and non-zero everywhere.

Note: The `Objective` handles the scaling and transformations (also of gradients) automatically. You can also implement your own transformation via `set_space_mode(unbounded, unit_mapping=None, inverse_unit_mapping=None)`. See the [bounded/unbounded guide](dfbench/Implementing-a-New-Algorithm.md#bounded-vs-unbounded-detailed-guide) for details.

### I want to use PyTorch. Is that supported?
Yes. Differometor and `Objective` are built in JAX, but many optimization libraries (like EvoX or BoTorch) use PyTorch. We provide lightweight conversion utilities (`t2j` to convert PyTorch to JAX, and `j2t` to convert JAX to PyTorch). The conversion routes through NumPy and has negligible overhead.

### Why is my optimizer returning NaN/Inf?
We checked for common issues that can cause NaN or Inf values. These should not happen. If your optimizer still returns NaN or Inf, check these three common culprits:
1. **Missing loss logs:** Ensure you are calling `objective.value_and_grad(params)` instead of just `objective.grad(params)`. The `grad()` method computes the gradient but does not compute or log the loss value.
2. **Invalid configuration:** For certain parameter combinations, there may be no signal reaching a detector or other extreme cases that cause the simulator to return NaN or Inf. Perturbate your parameters very slightly.
3. **Box constraints are zeroing gradients:** If parameters hit the hard edges of your bounds, gradients may become zero. Try initializing your `Objective` with `unbounded=True`.

---

## Evaluation & Benchmarking

### How will my algorithm be evaluated and ranked?
Algorithms will be [ranked](scoring.md) by their arithmetic mean performance on 10 new hidden topologies per evaluation. You may submit as often as you want; each monthly evaluation uses the last submission before that month's deadline. For each topology, the score uses the minimum loss among setups with `is_feasible=True` during the 4-hour wall-clock budget after `objective.start_logging()`. If a run has no feasible setup, that topology result is replaced by the best feasible loss from the organizers' random-search baseline on the same topology.

### Why is the budget measured in wall-clock time instead of iterations?
Different algorithms have vastly different per-evaluation computational costs. Parallelizing over VMAP batches can further amplify these differences. Measuring wall-clock time after `objective.start_logging()` ensures a fair comparison of what can be achieved within a fixed compute budget.

### What does the loss mean?

The loss is designed to be interpretable: If there are no power violations, a loss of -1 means a 10x mean increase in sensitivity, while -2 corresponds to a 100x mean increase. Similarily, a loss of 1 means a 10x decrease in sensitivity while 2 means a 100x decrease.

A loss of zero means the optimizer has matched the mean sensitivity accross frequencies of the human-designed "Voyager" gravitational wave detector. Achieving a loss below zero means your algorithm has discovered a setup that outperforms current reference designs in the context of the simulation which takes into account specific noise characteristics. Negative losses are completely possible and expected.

To build intuition, open the [interactive sensitivity loss explorer](sensitivity_loss_explorer.html).

---

## Troubleshooting & HPC Environments

### JAX is using all my GPU memory and crashing.
By default, JAX pre-allocates 75% of available GPU memory. If you are running multiple processes or memory-heavy parallel batches, disable this behavior by setting the following environment variable so JAX allocates memory on demand:
`export XLA_PYTHON_CLIENT_PREALLOCATE=false`

### I am getting a PermissionError from matplotlib on my HPC cluster.
On shared HPC filesystems, matplotlib's default configuration directory (`~/.config/matplotlib`) may be read-only, or multiple jobs might cause race conditions by writing to it simultaneously. 
**Solution:** Always `import dfbench` before importing `matplotlib`. The `dfbench` initialization script automatically detects this and redirects `MPLCONFIGDIR` to a safe, isolated temporary directory.
