# Problems

All optimization problems in dfbench represent gravitational-wave detector design tasks. The goal is to find optical component parameters (mirror reflectivities, laser power, cavity lengths, etc.) that minimize the detector's strain sensitivity across a frequency range of 20-5000 Hz.

**Import:**

```python
from dfbench.problems import VoyagerProblem, VoyagerTuningProblem, ConstrainedVoyagerProblem, UIFOProblem
```

---

## Problem Hierarchy

```
ContinuousProblem          (ABC : core/problem.py)
  └── OpticalSetupProblem  (ABC : problems/base_problem.py)
        ├── VoyagerProblem
        ├── VoyagerTuningProblem
        ├── ConstrainedVoyagerProblem
        └── UIFOProblem
```

### `ContinuousProblem` (Abstract Base)

Defines the minimal interface every problem must implement:

| Attribute / Method | Type | Description |
|--------------------|------|-------------|
| `name` | `str` | Human-readable identifier. |
| `objective_function` | `Callable` | Loss in bounded parameter space. |
| `bounds` | `Array[2, n_params]` | `[lower_bounds, upper_bounds]` for each parameter. |
| `optimization_pairs` | `list` | Component/property context aligned by parameter index. Entries are normally `(component_name, property_name)` tuples. A coupled parameter can use a list of tuples. |
| `n_params` | `int` | Number of parameters = `len(optimization_pairs)`. |
| `to_spec() -> dict` | `dict` | Reconstructive spec: a small, JSON-serialisable dict sufficient to rebuild an equivalent problem instance (see [Reconstruction & Problem Spec](#reconstruction--problem-spec) below). |

**Rationale: bounded problem contract:** Problems expose the bounded loss only; `Objective` owns any mapping required by algorithms that search in unbounded coordinates.

**Rationale: reconstructive spec:** A checkpoint is only useful for resume or provenance if the originating problem can be rebuilt. `to_spec()` encodes the problem's constructor arguments so a saved run is fully self-describing (see [Storage & Checkpointing](Storage-and-Checkpointing)).

### `OpticalSetupProblem` (Optical Base)

Extends `ContinuousProblem` with optics-specific functionality shared by all Differometor problems:

- **Frequency grid:** A log-spaced array of frequencies from 20 Hz to 5 kHz (configurable via `n_frequencies`).
- **Target sensitivity:** Stored in `_target_sensitivities`, computed from the reference detector design at initialization.
- **`calculate_sensitivity(params)`:** Computes the sensitivity curve for a given parameter vector: used for plotting, not optimization.
- **`bounds_overrides`:** All concrete problems accept optional property-level bound overrides (narrowing only).
- **`signal_floor`:** All concrete problems floor detector signal magnitudes before sensitivity normalization. Defaults to `1e-20`.
- **`print_bounds()`:** Prints the effective per-parameter bounds currently used by the problem.

> **Note:** `OpticalSetupProblem` no longer has an `output_to_files` method. Human-readable JSON/PNG output is now a *derived view* produced by `RunDataExporter` from a `RunState` snapshot (see [I/O Methods](Objective-API-Reference#io-methods)). Keeping I/O on the problem was a responsibility violation: it mixed file layout, plotting, and timestamping into the mathematical problem definition.

---

## Available Problems

### `VoyagerProblem`

| Property | Value |
|----------|-------|
| Setup | LIGO Voyager with balanced homodyne detection |
| Parameters | ~25 (reflectivities, tunings, squeezing, power, masses, lengths, phases) |
| Noise model | Single quantum noise source |
| Speed | ~12 ms/eval on A100 GPU |
| Difficulty | Moderate: loss < 0 achievable without physical constraints |

```python
problem = VoyagerProblem(n_frequencies=50)
```

```python
problem = VoyagerProblem(
   n_frequencies=50,
   bounds_overrides={"tuning": (0, 45)},
   signal_floor=1e-20,
)
problem.print_bounds()
```

#### How the loss works

1. The Voyager reference setup is simulated to get a target sensitivity curve.
2. The target loss is $\sum \log_{10}(\text{sensitivity}_{\text{target}})$.
3. For a candidate parameter set, the loss is:
   $$\text{loss} = \sum \log_{10}(\text{sensitivity}_{\text{candidate}}) - \text{target loss}$$
4. **Loss < 0** means the candidate has better sensitivity than the reference design.

**Rationale for log-scale loss:** Strain sensitivities span many orders of magnitude ($10^{-20}$ to $10^{-24}$). Summing log-sensitivities treats improvements at all frequencies equally rather than being dominated by the worst band.

#### What the parameters represent

Each parameter corresponds to a `(component, property)` pair from the Voyager setup:

| Property | Bounds | Physical meaning |
|----------|--------|-----------------|
| `reflectivity` | [0, 1] | Fraction of light reflected by a mirror |
| `tuning` | [0, 90] | Phase tuning of a mirror in degrees |
| `db` | [0.01, 20] | Squeezing level in decibels |
| `angle` | [-180, 180] | Squeezing angle in degrees |
| `power` | [0.01, 200] | Laser power in watts |
| `mass` | [0.01, 200] | Mirror suspension mass in kg |
| `length` | [1, 4000] | Cavity arm length in meters |
| `phase` | [-180, 180] | Phase offset in degrees |

#### Caveat

`VoyagerProblem` does **not** enforce physical constraints (e.g. maximum mirror power absorption). A solution with loss < 0 may be physically unrealizable. For constrained optimization, use `ConstrainedVoyagerProblem`.

---

### `VoyagerTuningProblem`

| Property | Value |
|----------|-------|
| Setup | LIGO Voyager with balanced homodyne detection |
| Parameters | 6 (tuning only: `prm`, `itmy`, `etmy`, `itmx`, `etmx`, `srm`) |
| Noise model | Single quantum noise source |
| Speed | ~12 ms/eval on A100 GPU |
| Difficulty | Moderate: lower-dimensional than `VoyagerProblem`, useful for quick prototyping |

```python
problem = VoyagerTuningProblem(n_frequencies=50)
```

```python
problem = VoyagerTuningProblem(
   n_frequencies=50,
   bounds_overrides={"tuning": (0, 45)},
   signal_floor=1e-20,
)
problem.print_bounds()
```

#### How the loss works

1. The Voyager reference setup is simulated to get a target sensitivity curve.
2. For a candidate parameter set, the loss is:
   $$\text{loss} = \mathrm{mean}\left(\log_{10}\left(\frac{\text{sensitivity}_{\text{candidate}}}{\text{sensitivity}_{\text{target}}}\right)\right) $$
3. **Loss < 0** means the candidate has better average sensitivity than the reference design.

#### What the parameters represent

All optimized parameters are mirror tuning angles in degrees:

| Property | Bounds | Physical meaning |
|----------|--------|-----------------|
| `tuning` | [-180, 180] | Phase tuning of selected Voyager optics (`prm`, `itmy`, `etmy`, `itmx`, `etmx`, `srm`) |

#### Caveat

`VoyagerTuningProblem` does **not** enforce physical constraints (e.g. maximum mirror power absorption). For constrained optimization, use `ConstrainedVoyagerProblem`.

---

### `ConstrainedVoyagerProblem`

| Property | Value |
|----------|-------|
| Setup | LIGO Voyager with balanced homodyne detection |
| Parameters | ~25 (same as `VoyagerProblem`) |
| Noise model | Three sources: quantum, amplitude, frequency noise |
| Constraints | Power thresholds on mirrors, beamsplitters, and detectors |
| Speed | ~25 ms/eval on A100 GPU |
| Difficulty | Hard: loss < 0 is very difficult to achieve |

```python
problem = ConstrainedVoyagerProblem(n_frequencies=50, signal_floor=1e-20)
```

#### Differences from `VoyagerProblem`

1. **Realistic noise model:** Uses three separate modulation modes (quantum noise, amplitude noise, frequency noise) and combines their contributions into a single sensitivity curve. This produces a more accurate picture of real detector performance.

2. **Power constraints (ReLU-based):** The loss includes a penalty term for violating optical power limits:
   - `HARD_SIDE_POWER_THRESHOLD`: maximum power on mirror/beamsplitter side ports
   - `SOFT_SIDE_POWER_THRESHOLD`: softer limit with gradual penalty
   - `DETECTOR_POWER_THRESHOLD`: maximum power on detector ports

   For each component the configured `power_penalty_fn(value, threshold)` is called and the results are summed.  Three presets are provided:

   | Preset | Formula | Import |
   |--------|---------|--------|
   | `squashed_relu_penalty` (default) | $\frac{\max(v/t-1,\,0)}{1+\max(v/t-1,\,0)}$ | `from dfbench.problems import squashed_relu_penalty` |
   | `relu_penalty` | $\max(v/t-1,\,0)$ | `from dfbench.problems import relu_penalty` |
   | `zero_penalty` | $0$ | `from dfbench.problems import zero_penalty` |

   You can also pass any custom function with signature `fn(value, threshold) -> penalty`:

   ```python
   import jax.numpy as jnp

   def my_quadratic_penalty(value, threshold):
       relu = jnp.maximum(value / threshold - 1, 0)
       return relu ** 2

   problem = ConstrainedVoyagerProblem(power_penalty_fn=my_quadratic_penalty)
   ```

   The penalty function can also be swapped **after** the problem has been constructed (e.g. after wrapping it in an `Objective`), via `Objective.set_penalty_fn(fn)`. This rebuilds the problem's JIT-compiled scalar and aux objectives and re-binds the Objective's cached evaluation callables, so the new penalty takes effect on subsequent evaluations. It must be called before `Objective.start_logging()`:

   ```python
   from dfbench import Objective
   from dfbench.problems import zero_penalty

   problem = ConstrainedVoyagerProblem()
   obj = Objective(problem)
   obj.set_penalty_fn(zero_penalty)   # disable the penalty term
   obj.warmup_value()
   obj.start_logging()
   ```

   Only problems with a power-constraint path accept `set_penalty_fn`. The opt-in marker is the class attribute `_supports_power_penalty`; `ConstrainedVoyagerProblem` and `UIFOProblem` set it to `True`, while `VoyagerProblem` and `VoyagerTuningProblem` leave it `False` and raise `RuntimeError` if the call is attempted, even though they inherit the method from the optical base class.

**Rationale (penalty squashing):** A raw penalty can become orders of magnitude larger than the sensitivity loss, making gradient-based optimizers ignore sensitivity entirely. The default `squashed_relu_penalty` bounds the penalty contribution while preserving its gradient direction.\
It could very well be that other penalty functions work better for certain algorithms (or even Adam). Feel free to play around!

#### Aux diagnostics

The constrained problems also expose a JIT-compiled `objective_function_aux(params)` alongside `objective_function`. It returns `(loss, aux)` where `aux` is a pytree dict carrying the loss decomposition and physical diagnostics for that eval:

| Key | Shape | Description |
|-----|-------|-------------|
| `sensitivity_loss` | scalar | The unpenalised sensitivity loss. |
| `penalty` | scalar | The summed penalty contribution. |
| `is_feasible` | scalar bool | `True` iff every per-group power is at or below its threshold. This is a physical check, independent of the active `power_penalty_fn` preset, so it stays meaningful even when the penalty is disabled with `zero_penalty`. |
| `violations` | `(n_constraints,)` | Per-constraint penalty values. |
| `power_values` | dict with `hard`, `soft`, `detector` leaves | Raw per-group power arrays. |

Because `aux` is a JAX pytree, `objective_function_aux` vmapps cleanly: a batched call adds a leading batch dim to every leaf, including the `power_values` sub-arrays.

The `Objective` wraps this with `value_aux`, `value_and_grad_aux`, `vmap_value_aux`, and `vmap_value_and_grad_aux`. These explicit methods—and standard value-bearing methods when an aux save token is selected—thread aux through logging without a second forward pass. Each aux field has its own token (`sensitivity_loss`, `penalty`, `is_feasible`, `power_values`, `violations`) plus a `batched_*` variant and `aux` / `batched_aux` convenience aliases. Derivative-only calls do not produce a primal aux pytree and append `None` to enabled aux histories. When a `batched_*` token is off and the non-batched token is on, aux from every batched value-bearing call, including a singleton batch, is reduced to the representative point picked by the loss minimum. `Objective.best_is_feasible` reports the feasibility of the best-loss point from that recorded history. Full reference in the [Objective API Reference](Objective-API-Reference).

---

### `UIFOProblem`

| Property | Value |
|----------|-------|
| Setup | Quasi-Universal Interferometer (UIFO) |
| Parameters | 50-250+ depending on grid size |
| Noise model | Three sources (same as constrained Voyager) |
| Constraints | Power thresholds (same as constrained Voyager) |
| Speed | ~500 ms/eval on A100 GPU |
| Difficulty | Hard but achievable: the UIFO is overparameterized |

```python
# From a topology seed (random topology, deterministic from seed)
problem = UIFOProblem(size=3, n_frequencies=50, topology_seed=42)

# From a compact topology string
problem = UIFOProblem(size=3, topology="AECGCCHEG-SLLSSHLLLLS")

# From explicit dicts
problem = UIFOProblem(
    size=3,
    centers={"11": ("beamsplitter", "left"), ...},
    boundaries={"01": "squeezer", ...},
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `size` | `3` | Grid dimensions (3 = 3×3). Larger grids have more components and parameters. |
| `n_frequencies` | `50` | Frequency points for sensitivity calculation. |
| `topology_seed` | `42` | Seed for random topology generation. Set to `None` (with no other topology args) to generate a truly random topology. The seed is always printed to the console. Mutually exclusive with `topology` and `centers`/`boundaries`. |
| `topology` | `None` | Compact topology string (see below). Mutually exclusive with `topology_seed`. |
| `centers` | `None` | Interior cell dict. Must be paired with `boundaries`. Mutually exclusive with `topology_seed` and `topology`. |
| `boundaries` | `None` | Boundary cell dict. Must be paired with `centers`. Mutually exclusive with `topology_seed` and `topology`. |
| `power_penalty_fn` | `squashed_relu_penalty` | Per-element penalty function `fn(value, threshold)`. See presets above. |
| `signal_floor` | `1e-20` | Lower floor for detector signal magnitudes before sensitivity normalization. |

> **Backwards compatibility:** `RandomUIFOProblem` is an alias for `UIFOProblem`.

#### Topology specification

There are three mutually exclusive ways to specify a UIFO topology:

1. **`topology_seed`**: The simplest option. A random topology is generated deterministically from the seed (default: `42`). Pass `topology_seed=None` with no other topology arguments to generate a truly random topology: the seed is printed so you can reproduce it.
2. **`topology` string**: A compact encoding ideal for configs, papers, and sharing. Uses single-character codes:
   - **Interior cells:** `A`-`D` = beamsplitter (left/right/top/bottom), `E`-`H` = directional\_beamsplitter (left/right/top/bottom)
   - **Boundary cells:** `L` = laser, `S` = squeezer, `D` = detector, `H` = balanced\_homodyne
   - Format: `"<interior_chars>-<boundary_chars>"` in row-major order.
3. **`centers` + `boundaries` dicts**: Explicit component placement, matching Differometor's native format.

Conversion helpers are available:

```python
from dfbench.problems.uifo import topology_to_string, topology_from_string

topology_str = topology_to_string(centers, boundaries, size=3)
centers, boundaries = topology_from_string(topology_str, size=3)
```

#### What is a UIFO?

A Quasi-Universal Interferometer Field Optimization (UIFO) is a grid-based interferometer where beamsplitters, mirrors, lasers, and squeezers are placed on a grid and connected by spaces. The topology (which components are placed where and how they connect) is generated randomly from `topology_seed` (printed on initialization for reproducibility). Once the topology is fixed, only the continuous parameters (reflectivities, tunings, lengths, etc.) are optimized.

**Rationale: coupled grid-cell spaces:** Horizontal and vertical spaces at the same grid positions are constrained to have equal lengths via `constrain_inter_grid_cell_spaces()`. This preserves the physical grid structure and prevents the optimizer from "folding" the interferometer into a degenerate geometry.

#### Design note

The reference sensitivity target is always the Voyager detector. Since the UIFO is overparameterized (many more degrees of freedom than Voyager), it can in principle achieve better sensitivity but the large parameter space makes optimization harder.

---

## Reconstruction & Problem Spec

Every problem implements `to_spec() -> dict`, which returns a small, JSON-serialisable dict capturing everything needed to rebuild an equivalent instance in a separate process. This is the reconstructive contract that makes checkpoints self-describing.

Starting with dfbench 0.1.1, the raw `to_spec()` dict is wrapped in a typed `ProblemSpec` container (`dfbench.core.problem.ProblemSpec`) that carries an explicit schema `version` and a separated `params` field. Checkpoints embed the container (`ProblemSpec.to_dict()` -> `{"type", "version", "params"}`) in `RunMetadata.extra["problem_spec"]`, so consumers get a stable, schema-validated identity instead of an untyped dict. Legacy flat specs (`{"type", <kwargs>}`) written by older versions are still accepted on load via `ProblemSpec.from_dict`.

### How it works

1. Each problem subclass implements `to_spec()`, returning a dict with a `"type"` key (the registry name) plus its constructor arguments.
2. The `@register_problem` decorator registers the class in a module-level registry under its `__name__` (or a custom `spec_type`).
3. `ContinuousProblem.to_problem_spec()` wraps the `to_spec()` dict into a typed `ProblemSpec` container. Subclasses rarely need to override this; the default implementation is sufficient as long as `to_spec()` is correct.
4. `build_problem_from_spec(spec)` accepts either a `ProblemSpec` or a raw dict (typed container or legacy flat form; both are normalized via `ProblemSpec.from_dict`) and reconstructs the instance.
5. `Objective.problem_spec` exposes the same typed-container dict directly, without requiring a checkpoint.
6. `Objective._build_metadata()` stores that problem spec in `RunMetadata.extra["problem_spec"]`, so every checkpoint records its originating problem.

```python
from dfbench.core.problem import ProblemSpec, build_problem_from_spec

# A spec captured from a live problem
ps = problem.to_problem_spec()
# ProblemSpec(type="VoyagerProblem", params={"n_frequencies": 50, ...}, version=1)

# JSON-safe dict for embedding in checkpoint metadata
spec_dict = ps.to_dict()
# {"type": "VoyagerProblem", "version": 1, "params": {"n_frequencies": 50, ...}}

# The Objective exposes this same dict directly
spec_dict = obj.problem_spec

# Rebuild an equivalent problem later, in any process
problem2 = build_problem_from_spec(ps)
# or, equivalently, from the dict form:
problem2 = build_problem_from_spec(spec_dict)
```

### The `ProblemSpec` container

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Registry key matching a `@register_problem`-decorated class |
| `params` | `dict[str, Any]` | Constructor keyword arguments forwarded to the problem class on reconstruction |
| `version` | `int` | Container schema version (defaults to `PROBLEM_SPEC_VERSION = 1`); governs the `type`/`version`/`params` layout, not the per-problem constructor args |

`ProblemSpec.from_dict` accepts both the typed container and the legacy flat form, so checkpoints written before the typed container existed still load. `ProblemSpec.__post_init__` validates that `type` is a non-empty string, `params` is a dict, and `version` is an int. A malformed or tampered spec becomes a deterministic `ValueError` at the trust boundary instead of a silent corruption downstream.

### Per-problem spec contents

The `params` sub-dict is whatever each problem's `to_spec()` returns minus the `"type"` key:

| Problem | `params` fields | Reconstruction path |
|---------|-----------------|----------------------|
| `VoyagerProblem` | `n_frequencies`, `power_penalty_fn`, `bounds_overrides` | Direct constructor call |
| `VoyagerTuningProblem` | `n_frequencies`, `power_penalty_fn`, `bounds_overrides` | Direct constructor call |
| `ConstrainedVoyagerProblem` | `n_frequencies`, `power_penalty_fn`, `bounds_overrides` | Direct constructor call |
| `UIFOProblem` | `size`, `n_frequencies`, `topology` (string), `power_penalty_fn`, `bounds_overrides` | Rebuilt from explicit `topology` string (deterministic, RNG-independent) |

### Penalty function encoding

Callables like `power_penalty_fn` are encoded **by name** via a registry of presets (`squashed_relu_penalty`, `relu_penalty`, `zero_penalty`). This keeps the spec JSON-safe. Custom penalty functions that are not registered presets will raise on `to_spec()`. Register them or use the built-in presets.

### Reconstructing from a checkpoint

Reconstruction is a two-step process that crosses the storage/problem layer boundary:

```python
from dfbench.core.storage import CheckpointManager
from dfbench.core.problem import ProblemSpec, build_problem_from_spec

state = manager.load(path)
spec_dict = CheckpointManager.extract_problem_spec(state)  # -> dict | None
if spec_dict is not None:
    ps = ProblemSpec.from_dict(spec_dict)        # typed container (accepts legacy flat too)
    problem = build_problem_from_spec(ps)        # or pass spec_dict directly
```

`CheckpointManager.extract_problem_spec` returns `None` if the run did not record a problem spec (e.g. the problem did not implement `to_spec`). The relevant problem module must be imported so its class is registered.

### Implementing `to_spec` for a new problem

If you add a new `ContinuousProblem` subclass:

1. Decorate it with `@register_problem` (imported from `dfbench.core.problem` or `dfbench.problems.base_problem`).
2. Implement `to_spec()` returning a dict with `"type"` (the class name) plus every constructor argument needed for `build_problem_from_spec` to produce an equivalent instance.
3. Encode any callables by name against a registry (see the penalty-function pattern in `base_problem.py`).

You do not need to override `to_problem_spec()`; the default implementation wraps `to_spec()` into the typed container automatically.

See [I/O Methods](Objective-API-Reference#io-methods) for how the spec is embedded in checkpoints.

---
