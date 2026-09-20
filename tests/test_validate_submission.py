from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock


VALIDATOR_PATH = Path(__file__).resolve().parents[1] / "tools" / "validate_submission.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("round1_submission_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = load_validator()


def write_submission(path: Path, body: str | None = None) -> None:
    source = body or """
from dfbench import OptimizationAlgorithm

class Submission(OptimizationAlgorithm):
    algorithm_str = "test_algorithm"

    def optimize(self, objective, random_seed=None, **kwargs):
        return None
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("submission.py", source)
        archive.writestr("requirements.txt", "")


class ValidateSubmissionTests(unittest.TestCase):
    def test_static_validation_remains_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            archive = Path(raw) / "submission.zip"
            write_submission(archive)
            output = io.StringIO()
            with mock.patch.object(validator, "run_local_submission") as run:
                with contextlib.redirect_stdout(output):
                    status = validator.main([str(archive)])

        self.assertEqual(status, 0)
        run.assert_not_called()
        self.assertIn("VALID (static checks)", output.getvalue())
        self.assertIn("Pass --run-local", output.getvalue())

    def test_run_mode_uses_supplied_seeds_and_budget(self) -> None:
        result = validator.LocalRunResult(
            optimizer_class="Submission",
            algorithm_name="test_algorithm",
            topology_seed=12,
            optimizer_seed=34,
            max_time_seconds=5.0,
            wall_time_seconds=1.0,
            objective_time_seconds=0.8,
            evaluation_count=3,
            budget_exceeded=False,
            finite_candidate_count=3,
            feasible_candidate_count=2,
            missing_feasibility_candidate_count=0,
            best_feasible_loss=0.5,
        )
        with tempfile.TemporaryDirectory() as raw:
            archive = Path(raw) / "submission.zip"
            write_submission(archive)
            output = io.StringIO()
            with mock.patch.object(
                validator, "run_local_submission", return_value=result
            ) as run:
                with contextlib.redirect_stdout(output):
                    status = validator.main(
                        [
                            str(archive),
                            "--run-local",
                            "--topology-seed",
                            "12",
                            "--optimizer-seed",
                            "34",
                            "--max-time-seconds",
                            "5",
                        ]
                    )

        self.assertEqual(status, 0)
        run.assert_called_once_with(
            str(archive),
            topology_seed=12,
            optimizer_seed=34,
            max_time_seconds=5.0,
        )
        self.assertIn("Best feasible loss: 0.5", output.getvalue())

    def test_local_runner_loads_submission_and_scores_feasible_history(self) -> None:
        fake_dfbench = types.ModuleType("dfbench")
        fake_problems = types.ModuleType("dfbench.problems")

        class FakeOptimizationAlgorithm:
            pass

        class FakeProblem:
            def __init__(self, topology_seed: int) -> None:
                self.topology_seed = topology_seed

        class FakeObjective:
            def __init__(
                self,
                problem,
                *,
                verbose,
                max_time,
                print_every,
                save,
                display_mode,
                save_params_history=True,
                save_batched_params_history=True,
            ) -> None:
                self.problem = problem
                self.max_time = max_time
                self.save = save
                self.loss_history = []
                self.is_feasible_history = []
                self.eval_count = 0
                self.time_elapsed = 0.0
                self.budget_exceeded = False

        fake_dfbench.Objective = FakeObjective
        fake_dfbench.OptimizationAlgorithm = FakeOptimizationAlgorithm
        fake_dfbench.problems = fake_problems
        fake_problems.UIFOProblem = FakeProblem

        source = """
from dfbench import OptimizationAlgorithm

class Submission(OptimizationAlgorithm):
    algorithm_str = "integration_test"

    def optimize(self, objective, random_seed=None, **kwargs):
        objective.loss_history = [[0.1, 0.5], [0.7]]
        objective.is_feasible_history = [[False, True], [True]]
        objective.eval_count = 3
        objective.time_elapsed = 2.5
"""
        with tempfile.TemporaryDirectory() as raw:
            archive = Path(raw) / "submission.zip"
            write_submission(archive, source)
            with mock.patch.dict(
                sys.modules,
                {"dfbench": fake_dfbench, "dfbench.problems": fake_problems},
            ):
                result = validator.run_local_submission(
                    str(archive),
                    topology_seed=123,
                    optimizer_seed=456,
                    max_time_seconds=7.0,
                )

        self.assertEqual(result.optimizer_class, "Submission")
        self.assertEqual(result.algorithm_name, "integration_test")
        self.assertEqual(result.topology_seed, 123)
        self.assertEqual(result.optimizer_seed, 456)
        self.assertEqual(result.evaluation_count, 3)
        self.assertEqual(result.feasible_candidate_count, 2)
        self.assertEqual(result.best_feasible_loss, 0.5)


if __name__ == "__main__":
    unittest.main()

