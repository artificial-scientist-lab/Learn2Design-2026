"""Standalone SciPy BFGS optimizer.

Optimizes in the Objective's unbounded (sigmoid-space) parameterization.
Its SciPy adapter calls the same Objective.value_and_grad() method as Adam, so
the Objective owns timing, logging, auxiliary feasibility, and budget accounting.
The optimizer restarts after convergence while budget remains, alternating
between perturbed-best and random initial points.

Requires: scipy (``uv add 'dfbench[scipy]'``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

try:
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for BFGS. Install with:  uv add 'dfbench[scipy]'"
    ) from exc

from dfbench import AlgorithmType, Objective, OptimizationAlgorithm


class SciPyBudgetExceeded(RuntimeError):
    """Raised internally to stop SciPy once the Objective budget is exhausted."""


@dataclass(slots=True)
class SciPyConfig:
    """Per-method SciPy integration settings."""

    method: str
    unbounded: bool
    use_bounds: bool
    use_jac: bool = True


class SciPyObjectiveAdapter:
    """Adapt Objective.value()/value_and_grad() to SciPy call signatures.

    Caches the latest point so that SciPy's separate fun() and jac() calls for
    the same x only trigger one Objective evaluation.
    """

    def __init__(self, obj: Objective, config: SciPyConfig) -> None:
        self.obj = obj
        self.config = config
        self._latest_x: np.ndarray | None = None
        self._latest_loss = None
        self._latest_grad = None

    @property
    def bounds(self) -> None:
        """Return SciPy bounds in physical space when requested."""
        if not self.config.use_bounds:
            return None
        problem_bounds = np.asarray(self.obj.bounds, dtype=float)
        from scipy.optimize import Bounds

        return Bounds(problem_bounds[0], problem_bounds[1], keep_feasible=True)

    @property
    def constraints(self) -> tuple:
        """Return supported SciPy constraints.

        This algorithm only supports box bounds; problems with constraint
        metadata are not handled here. This is a placeholder.
        """
        return ()

    def warmup(self) -> None:
        """Warm the same Objective method used by the SciPy callbacks."""
        if self.config.use_jac:
            self.obj.warmup_value_and_grad()
        else:
            self.obj.warmup_value()

    def _to_numpy_vector(self, x: Float[Array, "..."] | np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return np.atleast_1d(arr).astype(float, copy=False)

    def _same_point(self, x: np.ndarray) -> bool:
        return self._latest_x is not None and np.array_equal(self._latest_x, x)

    def _check_budget(self) -> None:
        if self.obj.budget_exceeded:
            raise SciPyBudgetExceeded("Objective budget exhausted inside SciPy.")

    # Penalty returned to SciPy when the objective produces non-finite values.
    _NAN_PENALTY: float = 1e10

    def _cache_value_grad(self, x, loss, grad) -> None:
        self._latest_x = np.array(x, copy=True)
        self._latest_loss = loss
        self._latest_grad = grad

    @staticmethod
    def _sanitize_loss(loss: float) -> float:
        """Replace non-finite loss with a large penalty so SciPy retreats."""
        if not np.isfinite(loss):
            return SciPyObjectiveAdapter._NAN_PENALTY
        return loss

    @staticmethod
    def _sanitize_grad(grad: np.ndarray) -> np.ndarray:
        """Replace non-finite gradient entries with zeros."""
        if not np.all(np.isfinite(grad)):
            return np.where(np.isfinite(grad), grad, 0.0)
        return grad

    def evaluate_value(self, x: np.ndarray) -> float:
        """Return loss only and log exactly once for this point."""
        x_np = self._to_numpy_vector(x)
        if self._same_point(x_np):
            return self._sanitize_loss(float(self._latest_loss))

        loss = self.obj.value(jnp.asarray(x_np))
        self._cache_value_grad(x_np, loss, None)
        self._check_budget()
        return self._sanitize_loss(float(loss))

    def evaluate_value_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Return loss/grad and log exactly once for this point."""
        if not self.config.use_jac:
            raise RuntimeError(
                "evaluate_value_and_grad() requested without jac-enabled adapter."
            )
        x_np = self._to_numpy_vector(x)
        if self._same_point(x_np):
            return (
                self._sanitize_loss(float(self._latest_loss)),
                self._sanitize_grad(np.asarray(self._latest_grad, dtype=float)),
            )

        loss, grad = self.obj.value_and_grad(jnp.asarray(x_np))
        self._cache_value_grad(x_np, loss, grad)
        self._check_budget()
        return (
            self._sanitize_loss(float(loss)),
            self._sanitize_grad(np.asarray(grad, dtype=float)),
        )

    def fun(self, x: np.ndarray) -> float:
        """SciPy fun callback."""
        if self.config.use_jac:
            loss, _ = self.evaluate_value_and_grad(x)
            return loss
        return self.evaluate_value(x)

    def jac(self, x: np.ndarray) -> np.ndarray:
        """SciPy jac callback."""
        _, grad = self.evaluate_value_and_grad(x)
        return grad


class BFGS(OptimizationAlgorithm):
    """SciPy BFGS in unbounded sigmoid space."""

    algorithm_str: str = "bfgs"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        self._last_result = None

    def _resolve_init_params(
        self,
        obj: Objective,
        init_params: Float[Array, "..."] | None,
    ) -> Float[Array, "n_params"]:
        if init_params is not None:
            return jnp.asarray(init_params)
        return obj.random_params_unbounded()

    def _perturbed_best_params(
        self,
        obj: Objective,
        scale: float = 0.1,
    ) -> Float[Array, "n_params"] | None:
        """Return the best-known point with Gaussian perturbation, or None."""
        best = obj.best_params
        if best is None:
            return None
        best_np = np.asarray(best, dtype=float)
        noise = scale * np.random.randn(*best_np.shape)
        return jnp.asarray(best_np + noise)

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        gtol: float = 1e-5,
        maxiter: int | None = None,
        xrtol: float = 0.0,
        tol: float | None = None,
        **scipy_kwargs,
    ) -> None:
        obj = objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)

        x0 = np.asarray(self._resolve_init_params(obj, init_params), dtype=float)
        adapter = SciPyObjectiveAdapter(
            obj,
            SciPyConfig(
                method="BFGS",
                unbounded=True,
                use_bounds=False,
                use_jac=True,
            ),
        )
        adapter.warmup()

        options = {"gtol": gtol, "maxiter": maxiter, "xrtol": xrtol, **scipy_kwargs}
        minimize_kwargs = {
            "fun": adapter.fun,
            "x0": x0,
            "method": "BFGS",
            "jac": adapter.jac,
            "bounds": adapter.bounds,
            "constraints": adapter.constraints,
            "tol": tol,
            "options": {k: v for k, v in options.items() if v is not None},
        }

        obj.start_logging()

        # Restart loop: when BFGS converges and budget remains, alternate
        # between perturbing the best-known point (exploit basin) and jumping
        # to a fresh random point (explore new basin).
        restart_count = 0
        while not obj.budget_exceeded:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    self._last_result = minimize(**minimize_kwargs)
            except SciPyBudgetExceeded:
                self._last_result = None
                return

            if not obj.budget_exceeded:
                restart_count += 1
                if restart_count % 2 == 0:
                    x0_new = self._perturbed_best_params(obj)
                    if x0_new is None:
                        x0_new = self._resolve_init_params(obj, None)
                else:
                    x0_new = self._resolve_init_params(obj, None)
                minimize_kwargs["x0"] = np.asarray(x0_new, dtype=float)
