# Utilities and Helpers

Small utility modules in `dfbench.core` that support the main API.

---

## Tensor Conversion: `t2j` / `j2t`

```python
from dfbench import t2j, j2t
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `t2j(tensor)` | `torch.Tensor -> jax.Array` | Detach -> CPU -> NumPy -> JAX |
| `j2t(arr)` | `jax.Array -> torch.Tensor` | JAX -> NumPy (writable copy) -> PyTorch |

**Why these exist:**
EvoX and BoTorch operate on PyTorch tensors; Differometor and the `Objective` use JAX arrays. Every algorithm that wraps a PyTorch library needs to convert back and forth. The conversion goes through NumPy because JAX's `dlpack` interop with PyTorch is unreliable for zero-copy transfers on some platforms.

**Why `j2t` makes a writable copy:**
`jax.numpy.array` values are immutable. `torch.from_numpy()` on a read-only array emits a warning. Creating a NumPy copy via `np.array(arr)` avoids this.

**Overhead:**
Negligible for the array sizes in this project (tens to hundreds of floats). A population of 100 × 25-parameter vectors copies ~20 KB, sub-microsecond.

---

## Inverse Sigmoid Bounding

```python
from dfbench.core.utils import inverse_sigmoid_bounding
```

```python
inverse_sigmoid_bounding(
    bounded_params: Float[Array, "N"],
    bounds: Float[Array, "2 N"],
) -> Float[Array, "N"]
```

Maps parameters from bounded space $[\text{lb}, \text{ub}]$ back to the unbounded space $(-\infty, +\infty)$ used by `Objective` in unbounded mode.

The default forward transform is:

$$x_{\text{bounded}} = \text{lb} + (\text{ub} - \text{lb}) \cdot \sigma(x_{\text{unbounded}})$$

The inverse is:

$$x_{\text{unbounded}} = \log\!\left(\frac{\hat{x}}{1 - \hat{x}}\right) \quad\text{where}\quad \hat{x} = \frac{x_{\text{bounded}} - \text{lb}}{\text{ub} - \text{lb}}$$

Values are clipped to $[10^{-7},\; 1 - 10^{-7}]$ before the logit to avoid $\pm\infty$.

**When you use this:**
When you have a bounded parameter vector and want to convert it to unbounded space (e.g., for initializing an algorithm that works in unconstrained space).

---

## CLI Argument Parser: `create_parser`

```python
from dfbench.core.config import create_parser
```

```python
create_parser(
    params: dict[str, Any],
    description: str = "Optimization configuration",
) -> argparse.ArgumentParser
```

Generates an `argparse.ArgumentParser` from a dictionary of parameter names and their default values.

**Behaviour:**
- Keys become CLI flags: `pop_size` -> `--pop-size`
- Types are inferred from defaults: `int`, `float`, `str`
- Booleans become store-true / store-false flags

**Example:**

```python
parser = create_parser({
    "pop_size": 100,
    "learning_rate": 0.01,
    "use_cuda": False,
})
args = parser.parse_args()
# python run.py --pop-size 200 --learning-rate 0.001 --use-cuda
```

**Why this exists:**
Batch scripts on HPC clusters pass hyperparameters as CLI arguments. This utility avoids writing boilerplate argparse code for each script.

---

## Environment Initialization: `_init_env`

```python
# Imported automatically by dfbench/__init__.py: users never import this directly.
```

This module runs **at import time** before any other dfbench code. It performs a single action:

```python
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mpl_config_")
```

**Why:** On shared HPC filesystems, the default matplotlib config directory (`~/.config/matplotlib`) may not be writable, or multiple jobs may race to write there simultaneously. Redirecting to a temporary directory avoids `PermissionError` and race conditions.

**Why it must come first:** The `MPLCONFIGDIR` environment variable must be set *before* `matplotlib` is imported anywhere, including transitively via dependencies. Since `_init_env.py` is the first import in `dfbench/__init__.py`, it runs before anything else.

---

## Public API Surface

The top-level `dfbench` package re-exports the submitter-facing symbols, the ones an optimization author needs to write and test an algorithm:

```python
from dfbench import (
    Objective,                    # core wrapper
    ContinuousProblem,           # problem ABC
    OptimizationAlgorithm,       # algorithm ABC
    AlgorithmType,               # enum
    t2j, j2t,                    # tensor conversion
    create_parser,               # CLI helper
)
```

Organizer-only symbols (storage, checkpointing, problem reconstruction) are not re-exported at the top level. Import them from their home modules:

```python
# Problem reconstruction (typed ProblemSpec container + registry)
from dfbench.core.problem import (
    ProblemSpec,                 # typed container: type, version, params
    build_problem_from_spec,     # rebuild a problem from a ProblemSpec or dict
    register_problem,            # decorator: register a problem class for reconstruction
    validate_spec_round_trip,    # rebuild + assert bounds/n_params match
)

# Modular storage (see Storage & Checkpointing)
from dfbench.core.storage import (
    CheckpointManager,           # facade orchestrating save/load
    CheckpointSerializer,        # serializer protocol
    NpzCheckpointSerializer,     # compressed-NPZ serializer (default)
    JsonCheckpointSerializer,    # pickle-free JSON serializer
    LocalFilesystemBackend,      # atomic local-FS storage backend (default)
    StorageBackend,              # storage backend protocol
    RunPathResolver,             # structured path construction
    RunDataExporter,             # human-readable JSON + PNG view
    RunState,                    # shared run data contract
    RunMetadata,                 # run identity + problem spec record
    validate_run_state,          # RunState invariant contract (scoring gate)
)
```

Algorithm and problem concrete classes are available from their subpackages:

```python
from dfbench.algorithms import AdamGD, EvoxPSO, BotorchBO, ...
from dfbench.problems import VoyagerProblem, VoyagerTuningProblem, ConstrainedVoyagerProblem, UIFOProblem
from dfbench.benchmark import Benchmark, AlgorithmConfig, BenchmarkResult
```

See [Storage Layer](Architecture-Overview.md#4-storage-layer-corestorage) for the full storage architecture and the individual component APIs, and [Problems](Problems.md) for the `ProblemSpec` container and the reconstructive contract.
