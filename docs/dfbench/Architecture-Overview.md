# Architecture Overview

## Module Map

```
src/dfbench/
├── __init__.py               # Public API surface
├── core/
│   ├── _init_env.py          # Pre-import environment setup (matplotlib, HPC)
│   ├── algorithm.py          # OptimizationAlgorithm ABC + AlgorithmType enum
│   ├── config.py             # CLI argument parser helper
│   ├── objective.py          # Objective wrapper (central piece)
│   ├── problem.py            # ContinuousProblem ABC + problem registry / spec
│   ├── storage/              # Modular checkpointing & export
│   │   ├── state.py          # RunState + RunMetadata (canonical data contract)
│   │   ├── backends.py       # StorageBackend protocol + LocalFilesystemBackend
│   │   ├── serializers.py    # NpzCheckpointSerializer, JsonCheckpointSerializer
│   │   ├── resolver.py       # RunPathResolver (structured path layout)
│   │   ├── exporter.py       # RunDataExporter (human-readable JSON + PNG view)
│   │   └── manager.py        # CheckpointManager (facade used by Objective)
│   └── utils.py              # torch<->jax conversion, inverse sigmoid
├── algorithms/
│   ├── derivative_free/      # OMADS, PDFO/Py-BOBYQA, NelderMead, Powell
│   ├── global_search/        # RandomSearch, BasinHopping, DualAnnealing
│   ├── evolutionary/         # EvoxPSO, EvoxES, Nevergrad, CMA family
│   ├── gradient_based/
│   │   ├── optax/            # 34 Optax-based optimizers (OptaxAdam, OptaxLAMB, ...)
│   │   ├── scipy/            # 13 SciPy-based optimizers (BFGS, TNC, SLSQP, ...)
│   │   ├── custom_jax.py     # Native-JAX custom/hybrid batch
│   │   └── *.py              # Custom-loop algorithms (AdamGD, LBFGSGD, SAGD, ...)
│   ├── surrogate_based/      # BoTorch/Ax/HEBO/SMAC/ReSTIR/TuRBO-LBFGS
│   └── generative/           # VAESampling
├── problems/
│   ├── base_problem.py       # OpticalSetupProblem (shared optics logic + spec helper)
│   ├── voyager/              # VoyagerProblem, VoyagerTuningProblem, ConstrainedVoyagerProblem
│   └── uifo/                 # UIFOProblem
└── benchmark/
    ├── benchmark.py          # Benchmark orchestrator, AlgorithmConfig
    └── metrics.py            # Per-run, aggregation, and multi-run metrics
```

---

## Separation of Concerns

The framework is organised around **three strict boundaries**:

### 1. Problem Layer (`core/problem.py`, `problems/`)

A problem defines *what* is being optimised. Every problem subclasses `ContinuousProblem` and exposes:

| Attribute | Purpose |
|-----------|---------|
| `objective_function` | Loss in **bounded** parameter space (used by e.g. evolutionary / surrogate algorithms) |
| `bounds` | `(2, n_params)` lower / upper limits |
| `optimization_pairs` | `[(component, property), ...]` mapping each parameter index to a Differometor component |
| `to_spec() -> dict` | Reconstructive spec: a small, JSON-serialisable dict sufficient to rebuild an equivalent problem instance (see [Problems](Problems)) |
| `to_problem_spec() -> ProblemSpec` | Typed container wrapping `to_spec()`; carries `type`, `version`, `params`. This is what checkpoints embed. |

**Rationale (one bounded problem function):** Some optimization methods benefit from unconstrained $(-\infty, +\infty)$ space where gradients flow smoothly without hitting box-constraint boundaries. `Objective` provides that mapping layer so problem implementations only need to define the bounded loss.

**Rationale (reconstructive spec):** A checkpoint is only useful for resume or provenance if the originating problem can be rebuilt. `to_spec()` encodes the problem's constructor arguments (and, for UIFO, its topology string); `to_problem_spec()` wraps that into a typed `ProblemSpec` container (`type`, `version`, `params`) so a saved run is fully self-describing and consumers get a schema-validated identity. See [Storage Layer](#4-storage-layer-corestorage).

### 2. Objective Layer (`core/objective.py`)

`Objective` is the **sole interface** between any algorithm and its problem. It transparently:

- Maps unbounded coordinates into problem bounds when needed, then evaluates the bounded problem objective
- Prepares `jax.grad`, `jax.hessian`, `jax.value_and_grad`, and `jax.vmap` variants
- Records every admitted call with aligned loss / gradient / Hessian / params / timestamp histories
- Enforces wall-clock time and evaluation-count budgets
- Provides deterministic random sampling via a splittable JAX PRNG
- Delegates all file I/O to the modular `dfbench.core.storage` layer (see [Storage Layer](#4-storage-layer-corestorage))

**Rationale: Why a wrapper instead of bare functions?** Without it, every algorithm would need to independently implement timing, budget checks, history logging, checkpointing, and bounded<->unbounded transforms. This both duplicates code and makes cross-algorithm comparison unreliable because each implementation might measure time or count evaluations slightly differently.

**Rationale of decoupled storage:** `Objective` only builds/applies the canonical `RunState` data contract; the *how* (NPZ vs JSON) and *where* (local disk vs S3) of saving are handled by an internally-assembled `CheckpointManager`. The storage components (serializer, backend, resolver) are modular and testable in isolation but are not user-facing constructor parameters; the only storage knob exposed is `save_to_file_every`. No `./data/...` path is hardcoded in `Objective`.

### 3. Algorithm Layer (`core/algorithm.py`, `algorithms/`)

An algorithm defines *how* to search. Every algorithm subclasses `OptimizationAlgorithm` and provides:

| Attribute / Method | Purpose |
|--------------------|---------|
| `algorithm_str` | Unique identifier (e.g. `"adam_gd"`, `"evox_cmaes"`) |
| `algorithm_type` | One of `GRADIENT_BASED`, `EVOLUTIONARY`, `DERIVATIVE_FREE`, `GLOBAL_SEARCH`, `SURROGATE_BASED`, `GENERATIVE` |
| `optimize(objective, ...)` | Main entry point: receives a pre-configured `Objective`, runs the loop, returns it |

Algorithms **never** create their own `Objective`; they receive one from the caller (or from the `Benchmark` harness). This inversion of control ensures the harness can set budget limits, select seeds, and configure history storage uniformly.

### 4. Storage Layer (`core/storage/`)

A modular package that decouples *what* is saved from *how* and *where*:

- `RunState` / `RunMetadata`: the canonical, serializer-agnostic data contract (including the embedded `problem_spec`).
- `CheckpointSerializer`: format strategy (NPZ default, JSON alternative).
- `StorageBackend`: byte-level destination (local filesystem default, trivially swappable for memory / S3).
- `RunPathResolver`: structured path construction from components (no hardcoded paths in `Objective`).
- `RunDataExporter`: human-readable JSON + PNG view derived from `RunState`.
- `CheckpointManager`: the single facade `Objective` holds; wires the above together.

See [Objective I/O Methods](Objective-API-Reference.md#io-methods) for the user-facing checkpointing calls.

---

## Data Flow

```
                      Algorithm.optimize()
                            │
         ┌────────────┬───────────────┬──────────────────────┐
         │            │               │                      │
   obj.value(p)  obj.value_and_grad(p)  obj.hessian(p)  obj.vmap_value(batch)
         │            │               │                      │
         └────────────┴───────┬───────┴──────────────────────┘
                              │
      Objective._value_func / derivative transforms / vmapped transforms
                              │
                              ▼
               ┌──────────────────────────────┐
               │ optional Objective mapping   │
               │ objective_function or        │
               │ objective_function_aux       │
               └──────────────┬───────────────┘
                              │
                              ▼
                      Differometor.simulate()
                              │
                              ▼
                 value / derivatives + optional aux
                              │
                              ▼
          _log(params, loss, grad, hessian, aux)
                              │
                              ▼
                       _log_evals(..., aux)
                              │
                  ┌───────────┴────────────┐
                  │ admitted               │ rejected
                  ▼                        ▼
          time + standard histories   no history mutation
          + best/stagnation + selected aux histories
                  │                        │
                  └───────────┬────────────┘
                              ▼
                         _log_to_file()
                              │
                              ▼
                    periodic checkpoint via
                CheckpointManager -> StorageBackend
```

Every call to `obj.value()`, `obj.value_and_grad()`, `obj.hessian()`, `obj.value_grad_and_hessian()`, or any `vmap_*` variant follows this pipeline. Value-bearing methods carry one primal aux pytree when aux storage is selected; derivative-only methods carry no aux and contribute `None` to selected aux histories. The internal `_log()` coordinator takes one time snapshot, delegates atomic admission and history tracking to `_log_evals()`, appends the timestamp only for an admitted call, and triggers `_log_to_file()` for periodic checkpoints. `_log_to_file()` calls `CheckpointManager.tick()`, which checks the cadence (`save_every`), lazily builds a `RunState` only when a checkpoint is due, saves it through the internal `StorageBackend`, and returns the save duration so the Objective can exclude it from the elapsed-time clock. The algorithm receives the computed result; logging is a side effect invisible to the caller.

After logging has started, algorithms with custom JIT-compiled evaluation loops (e.g. L-BFGS with line-search) can use `obj.value_function(...)` and `obj.value_function_aux(...)` for the same Objective-owned bounded/unbounded mapping without Python-side logging. The latter returns an aux-aware callable whose invocation yields `(loss, aux)`, or `None` when unsupported. Before logging, both getters raise `RuntimeError`. Only Objective-owned `warmup_*()` paths can evaluate without returning results. `obj.log_evaluation(params, loss, grad=None, hessian=None, aux=None)` records a completed custom evaluation through the same logging pipeline and also requires active logging. Manual logging never manufactures aux. Omitting it appends `None` to each enabled aux history. Do not call the private methods directly.

---

## Benchmark Orchestration

```
Benchmark.run()
    │
    ├─ for each AlgorithmConfig:
    │     ├─ for each run (1 ... n_runs):
    │     │     ├─ Create Objective (with budget, seed)
    │     │     ├─ algorithm.optimize(objective, **hyperparams)
    │     │     └─ RunData.from_objective(objective)
    │     └─ AlgorithmRunData (list of RunData)
    │
    ├─ for each AlgorithmRunData:
    │     └─ _evaluate_algorithm() -> BenchmarkResult
    │           └─ at each time sample t:
    │                 - slice histories at t
    │                 - compute per-run metrics
    │                 - aggregate across runs
    │
    └─ Save CSV + optional NPZ run data
```

The `Benchmark` class:

1. **Injects dependencies**: creates `Objective` instances with uniform budget and seed settings, then passes them to each algorithm.
2. **Collects raw data**: `RunData.from_objective()` extracts numpy arrays from each finished `Objective`.
3. **Evaluates at time checkpoints**: metrics are computed at `n_time_samples` evenly-spaced wall-clock times so algorithms with different per-evaluation costs remain comparable.
4. **Supports reload**: `save_run_data=True` persists raw histories to NPZ; `load_from=` re-evaluates metrics without re-running algorithms.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **JAX for physics, PyTorch for some algorithms** | Differometor is a JAX project. Some optimisation libraries (EvoX, BoTorch) require PyTorch tensors. The `t2j` / `j2t` utilities bridge the gap with negligible overhead. |
| **Sigmoid bounding option** | Optimization in clipped-bounded space can produce zero gradients at boundaries. The sigmoid map $\sigma(x) \cdot (\text{ub} - \text{lb}) + \text{lb}$ provides an alternative where gradients remain nonzero everywhere in unconstrained space. |
| **Wall-clock time as primary budget** | Evaluation cost varies across problems (12 ms for Voyager, 500 ms for UIFO). Time-based budgets make cross-problem comparisons meaningful. |
| **Time-sampled metrics** | Evaluating metrics at fixed time points (not iteration counts) normalises for per-eval cost differences between algorithms. |
| **Atomic checkpoints** | Long HPC jobs are killed without warning. The `LocalFilesystemBackend` writes to a sibling temp file and calls `os.replace`; a reader always sees either the previous complete file or the new one, never a partial one. The previous good file is never destroyed before the new one is in place. |
| **Modular storage layer** | `Objective` delegates all I/O to `dfbench.core.storage`, assembling the components internally with sensible defaults. Formats (NPZ/JSON), locations (disk/S3), and path layout are modular and testable in isolation but not user-facing. A single canonical `RunState` contract prevents format drift. |
| **Self-describing checkpoints** | Each checkpoint embeds `RunMetadata` (problem/algo/budget identity) plus the problem's `to_spec()` reconstructive dict, so a saved run can be audited and resumed in any process without the caller holding the original `Problem` object. |
| **`_init_env.py` setting `MPLCONFIGDIR`** | On shared HPC filesystems, matplotlib's default config directory may be read-only. Setting a temp directory before any import prevents cryptic crashes. |
| **`AlgorithmType` enum** | The enum mirrors the `algorithms/` package families. The benchmark uses it as a default hint: gradient-based algorithms typically get `unbounded=True`, while evolutionary, derivative-free, global-search, surrogate, and generative methods get `unbounded=False` unless their implementation overrides the mode. |
| **Reduced history properties** | Batched algorithms produce `(batch, ...)` shaped histories. The `*_reduced` properties collapse each batch to a single representative (argmin of loss) so downstream analysis code never needs to handle ragged shapes. |
