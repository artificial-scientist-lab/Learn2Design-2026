#!/usr/bin/env python3
"""Local validator for Learn2Design-2026 submission ZIP files.

By default, the validator checks only the published archive and source-level
rules.  It does not install dependencies, import participant code, run an
optimizer, or access hidden problems.

The explicit ``--run-local`` option additionally executes the submission on one
freshly generated public UIFO topology with a four-hour Objective budget.  This
uses the current Python environment and local hardware.  It does not reproduce
the organizer infrastructure or predict the official score.

Usage:
    python3 validate_submission.py submission.zip
    python3 validate_submission.py submission.zip --run-local
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import importlib.util
import inspect
import math
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
import time
import tokenize
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


OFFICIAL_MAX_TIME_SECONDS = 4 * 60 * 60
MAX_RANDOM_SEED = 2**31 - 1


PROTECTED_PACKAGES = {
    "dfbench",
    "differometor",
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "jax-cuda13-plugin",
    "jax-cuda13-pjrt",
}

PRIVATE_OBJECTIVE_NAMES = {
    "_map_bounded_to_unbounded",
    "_map_unbounded_to_bounded",
    "_max_time",
    "_optimization_pairs",
    "_problem",
    "_topology_string",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    message: str
    path: str | None = None
    line: int | None = None

    def render(self) -> str:
        location = ""
        if self.path is not None:
            location = f" {self.path}"
            if self.line is not None:
                location += f":{self.line}"
            location += ":"
        return f"[{self.severity}]{location} {self.message}"


@dataclass(frozen=True)
class LocalRunResult:
    optimizer_class: str
    algorithm_name: str
    topology_seed: int
    optimizer_seed: int
    max_time_seconds: float
    wall_time_seconds: float
    objective_time_seconds: float | None
    evaluation_count: int
    budget_exceeded: bool
    finite_candidate_count: int
    feasible_candidate_count: int
    missing_feasibility_candidate_count: int
    best_feasible_loss: float | None


def decode_python(raw: bytes, path: str) -> str:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        return raw.decode(encoding)
    except (LookupError, SyntaxError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot decode Python source: {exc}") from exc


def is_forbidden_module(module: str | None) -> bool:
    return bool(module) and (
        module == "differometor" or module.startswith("differometor.")
    )


def dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def inspect_source(path: str, source: str) -> tuple[list[Finding], ast.Module | None]:
    findings: list[Finding] = []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        findings.append(
            Finding("FAIL", f"invalid Python syntax: {exc.msg}", path, exc.lineno)
        )
        return findings, None

    warned_private: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_forbidden_module(alias.name):
                    findings.append(
                        Finding(
                            "FAIL",
                            f"forbidden import of {alias.name!r}; use only the Objective API",
                            path,
                            node.lineno,
                        )
                    )
        elif isinstance(node, ast.ImportFrom) and is_forbidden_module(node.module):
            findings.append(
                Finding(
                    "FAIL",
                    f"forbidden import from {node.module!r}; use only the Objective API",
                    path,
                    node.lineno,
                )
            )
        elif isinstance(node, ast.Call):
            called = dotted_name(node.func)
            if called in {"__import__", "importlib.import_module"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if is_forbidden_module(first.value):
                        findings.append(
                            Finding(
                                "FAIL",
                                f"dynamic import of forbidden module {first.value!r}",
                                path,
                                node.lineno,
                            )
                        )
            if called == "getattr" and len(node.args) >= 2:
                name = node.args[1]
                if (
                    isinstance(name, ast.Constant)
                    and isinstance(name.value, str)
                    and name.value in PRIVATE_OBJECTIVE_NAMES
                ):
                    warned_private.add((node.lineno, name.value))
        if isinstance(node, ast.Attribute) and node.attr in PRIVATE_OBJECTIVE_NAMES:
            warned_private.add((node.lineno, node.attr))

    for line, name in sorted(warned_private):
        findings.append(
            Finding(
                "WARN",
                f"private implementation attribute {name!r}; prefer documented Objective metadata and helpers",
                path,
                line,
            )
        )
    return findings, tree


def optimization_subclasses(tree: ast.Module) -> list[ast.ClassDef]:
    direct_names = {"OptimizationAlgorithm"}
    dfbench_aliases = {"dfbench"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "dfbench":
            for alias in node.names:
                if alias.name == "OptimizationAlgorithm":
                    direct_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dfbench":
                    dfbench_aliases.add(alias.asname or "dfbench")

    matches: list[ast.ClassDef] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {dotted_name(base) for base in node.bases}
        accepted = direct_names | {
            f"{alias}.OptimizationAlgorithm" for alias in dfbench_aliases
        }
        if bases & accepted:
            matches.append(node)
    return matches


def normalize_member(name: str) -> str:
    if "\\" in name:
        raise ValueError("backslashes are not portable ZIP path separators")
    if not name or name.startswith("/"):
        raise ValueError("empty or absolute archive path")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe archive path component")
    return path.as_posix().rstrip("/")


def requirement_name(declaration: str) -> str | None:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", declaration)
    if not match:
        return None
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def validate_zip(path: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        return [Finding("FAIL", f"cannot open ZIP: {exc}")]

    with archive:
        normalized: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            try:
                name = normalize_member(info.filename)
            except ValueError as exc:
                findings.append(Finding("FAIL", str(exc), info.filename))
                continue
            if name in normalized:
                findings.append(Finding("FAIL", "duplicate normalized path", name))
                continue
            normalized[name] = info
            if info.flag_bits & 0x1:
                findings.append(Finding("FAIL", "encrypted ZIP members are unsupported", name))
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if kind == stat.S_IFLNK:
                findings.append(Finding("FAIL", "symbolic links are not allowed", name))
            elif kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                findings.append(Finding("FAIL", "special filesystem entries are not allowed", name))

        root_files = {
            name for name, info in normalized.items() if "/" not in name and not info.is_dir()
        }
        for required in ("submission.py", "requirements.txt"):
            if required not in root_files:
                findings.append(
                    Finding("FAIL", f"required root-level file {required!r} is missing")
                )

        parsed: dict[str, ast.Module] = {}
        for name, info in normalized.items():
            if info.is_dir() or not name.endswith(".py"):
                continue
            try:
                source = decode_python(archive.read(info), name)
            except (OSError, ValueError) as exc:
                findings.append(Finding("FAIL", str(exc), name))
                continue
            source_findings, tree = inspect_source(name, source)
            findings.extend(source_findings)
            if tree is not None:
                parsed[name] = tree

        entry_tree = parsed.get("submission.py")
        if entry_tree is not None:
            classes = optimization_subclasses(entry_tree)
            if len(classes) != 1:
                findings.append(
                    Finding(
                        "FAIL",
                        "submission.py must define exactly one direct dfbench.OptimizationAlgorithm subclass; "
                        f"found {len(classes)}",
                        "submission.py",
                    )
                )
            elif not any(
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == "optimize"
                for item in classes[0].body
            ):
                findings.append(
                    Finding(
                        "FAIL",
                        f"{classes[0].name} does not define optimize()",
                        "submission.py",
                        classes[0].lineno,
                    )
                )

        req = normalized.get("requirements.txt")
        if req is not None and not req.is_dir():
            try:
                text = archive.read(req).decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                findings.append(Finding("FAIL", f"cannot read requirements.txt: {exc}"))
            else:
                for line_number, raw in enumerate(text.splitlines(), 1):
                    declaration = raw.split("#", 1)[0].strip()
                    if not declaration:
                        continue
                    if declaration.startswith("-") or re.search(
                        r"(?:https?|file|git\+[^:]+)://|\s@\s*(?:https?|file|git\+)",
                        declaration,
                        re.IGNORECASE,
                    ):
                        findings.append(
                            Finding(
                                "FAIL",
                                "installer directives and URL/VCS dependencies are not allowed",
                                "requirements.txt",
                                line_number,
                            )
                        )
                    package = requirement_name(declaration)
                    if package in PROTECTED_PACKAGES:
                        findings.append(
                            Finding(
                                "WARN",
                                f"{package!r} is organizer-managed; this declaration is ignored",
                                "requirements.txt",
                                line_number,
                            )
                        )

    return findings


def positive_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def random_seed() -> int:
    return secrets.randbelow(MAX_RANDOM_SEED + 1)


def safe_extract(archive_path: str, destination: Path) -> None:
    """Extract an already validated archive without preserving special modes."""

    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = normalize_member(info.filename)
            target = destination.joinpath(*PurePosixPath(name).parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


@contextlib.contextmanager
def submission_directory(path: Path) -> Iterator[None]:
    previous_directory = Path.cwd()
    sys.path.insert(0, str(path))
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_directory)
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


def load_optimizer_class(submission_path: Path, base_class: type[Any]) -> type[Any]:
    module_name = f"_learn2design_local_submission_{secrets.token_hex(8)}"
    spec = importlib.util.spec_from_file_location(module_name, submission_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    classes = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and value is not base_class
        and value.__module__ == module_name
        and issubclass(value, base_class)
    ]
    if len(classes) != 1:
        raise RuntimeError(
            "submission.py must load exactly one OptimizationAlgorithm subclass; "
            f"found {len(classes)}"
        )
    return classes[0]


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize_objective(objective: Any) -> dict[str, Any]:
    """Compute the exact feasible best from scalar and batched public histories."""

    import numpy as np

    losses = list(getattr(objective, "loss_history", []))
    feasibility = list(getattr(objective, "is_feasible_history", []))
    finite_candidates = 0
    feasible_candidates = 0
    missing_feasibility = 0
    best_feasible_loss: float | None = None

    for history_index, loss_value in enumerate(losses):
        flat_losses = np.asarray(loss_value).reshape(-1)
        finite_mask = np.isfinite(flat_losses)
        finite_candidates += int(np.count_nonzero(finite_mask))
        if not np.any(finite_mask):
            continue

        if history_index >= len(feasibility) or feasibility[history_index] is None:
            missing_feasibility += int(np.count_nonzero(finite_mask))
            continue
        flat_feasibility = np.asarray(
            feasibility[history_index], dtype=bool
        ).reshape(-1)
        if flat_feasibility.size != flat_losses.size:
            missing_feasibility += int(np.count_nonzero(finite_mask))
            continue

        feasible_mask = finite_mask & flat_feasibility
        feasible_candidates += int(np.count_nonzero(feasible_mask))
        if not np.any(feasible_mask):
            continue
        candidate = float(np.min(flat_losses[feasible_mask]))
        if best_feasible_loss is None or candidate < best_feasible_loss:
            best_feasible_loss = candidate

    return {
        "objective_time_seconds": finite_float(
            getattr(objective, "time_elapsed", None)
        ),
        "evaluation_count": int(getattr(objective, "eval_count", 0)),
        "budget_exceeded": bool(getattr(objective, "budget_exceeded", False)),
        "finite_candidate_count": finite_candidates,
        "feasible_candidate_count": feasible_candidates,
        "missing_feasibility_candidate_count": missing_feasibility,
        "best_feasible_loss": best_feasible_loss,
    }


def run_local_submission(
    archive_path: str,
    *,
    topology_seed: int,
    optimizer_seed: int,
    max_time_seconds: float,
) -> LocalRunResult:
    """Run one archive on one public random UIFO topology in the current env."""

    try:
        from dfbench import Objective, OptimizationAlgorithm
        from dfbench.problems import UIFOProblem
    except ImportError as exc:
        raise RuntimeError(
            "local execution requires the official project environment and the "
            "submission dependencies to be installed"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="learn2design_local_validation_") as raw:
        work_dir = Path(raw)
        safe_extract(archive_path, work_dir)
        with submission_directory(work_dir):
            optimizer_class = load_optimizer_class(
                work_dir / "submission.py", OptimizationAlgorithm
            )
            optimizer = optimizer_class()
            problem = UIFOProblem(topology_seed=topology_seed)
            objective_kwargs: dict[str, Any] = {
                "verbose": 1,
                "max_time": max_time_seconds,
                "print_every": 100,
                "save": [
                    "is_feasible",
                    "batched_loss",
                    "batched_is_feasible",
                ],
                "display_mode": "log",
            }
            objective_parameters = inspect.signature(Objective).parameters
            if "save_params_history" in objective_parameters:
                objective_kwargs["save_params_history"] = False
            if "save_batched_params_history" in objective_parameters:
                objective_kwargs["save_batched_params_history"] = False
            objective = Objective(problem, **objective_kwargs)

            started = time.monotonic()
            optimizer.optimize(objective, random_seed=optimizer_seed)
            wall_time_seconds = time.monotonic() - started
            summary = summarize_objective(objective)
            algorithm_name = str(
                getattr(
                    optimizer,
                    "algorithm_str",
                    getattr(
                        optimizer_class,
                        "algorithm_str",
                        optimizer_class.__name__,
                    ),
                )
            )

    return LocalRunResult(
        optimizer_class=optimizer_class.__name__,
        algorithm_name=algorithm_name,
        topology_seed=topology_seed,
        optimizer_seed=optimizer_seed,
        max_time_seconds=max_time_seconds,
        wall_time_seconds=wall_time_seconds,
        **summary,
    )


def print_local_result(result: LocalRunResult) -> None:
    print("\nLOCAL RUN COMPLETED")
    print(f"Optimizer: {result.optimizer_class} ({result.algorithm_name})")
    print(f"Topology seed: {result.topology_seed}")
    print(f"Optimizer seed: {result.optimizer_seed}")
    print(f"Objective budget: {result.max_time_seconds:.0f} seconds")
    print(f"Submission wall time: {result.wall_time_seconds:.1f} seconds")
    if result.objective_time_seconds is None:
        print("Objective time: unavailable (start_logging() may not have been called)")
    else:
        print(f"Objective time: {result.objective_time_seconds:.1f} seconds")
    print(f"Objective budget reached: {'yes' if result.budget_exceeded else 'no'}")
    print(f"Evaluations: {result.evaluation_count}")
    print(f"Finite logged candidates: {result.finite_candidate_count}")
    print(f"Feasible logged candidates: {result.feasible_candidate_count}")
    if result.missing_feasibility_candidate_count:
        print(
            "WARNING: feasibility was unavailable for "
            f"{result.missing_feasibility_candidate_count} finite candidate(s)."
        )
    if result.best_feasible_loss is None:
        print("Best feasible loss: none found")
    else:
        print(f"Best feasible loss: {result.best_feasible_loss:.12g}")
    print(
        "This is one public random topology on local hardware, not an official "
        "leaderboard score."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Learn2Design-2026 submission ZIP."
    )
    parser.add_argument("archive", help="path to the submission ZIP")
    parser.add_argument(
        "--run-local",
        action="store_true",
        help=(
            "execute the archive on one fresh public UIFO topology in the current "
            "environment (runs participant code)"
        ),
    )
    parser.add_argument(
        "--topology-seed",
        type=int,
        help="reproduce a topology seed instead of generating a fresh one",
    )
    parser.add_argument(
        "--optimizer-seed",
        type=int,
        help="set the optimizer seed instead of generating a fresh one",
    )
    parser.add_argument(
        "--max-time-seconds",
        type=positive_seconds,
        default=float(OFFICIAL_MAX_TIME_SECONDS),
        help=(
            "Objective budget for --run-local; default: 14400 (the official four "
            "hours). Use a smaller value only for a smoke test"
        ),
    )
    args = parser.parse_args(argv)

    for name in ("topology_seed", "optimizer_seed"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= MAX_RANDOM_SEED:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and {MAX_RANDOM_SEED}")
    if not args.run_local and (
        args.topology_seed is not None
        or args.optimizer_seed is not None
        or args.max_time_seconds != float(OFFICIAL_MAX_TIME_SECONDS)
    ):
        parser.error("seed and time options require --run-local")

    findings = validate_zip(args.archive)
    for finding in findings:
        print(finding.render())
    failures = sum(item.severity == "FAIL" for item in findings)
    warnings = sum(item.severity == "WARN" for item in findings)
    if failures:
        print(f"\nINVALID: {failures} failure(s), {warnings} warning(s).")
        return 1
    print(f"\nVALID (static checks): {warnings} warning(s).")
    if not args.run_local:
        print("This does not install dependencies or execute the submitted optimizer.")
        print("Pass --run-local to test one fresh topology with a four-hour budget.")
        return 0

    topology_seed = (
        args.topology_seed if args.topology_seed is not None else random_seed()
    )
    optimizer_seed = (
        args.optimizer_seed if args.optimizer_seed is not None else random_seed()
    )
    print("\nWARNING: --run-local executes code from the submitted ZIP.")
    print("Dependencies are not installed automatically; the current environment is used.")
    topology_label = (
        "Fresh topology seed" if args.topology_seed is None else "Topology seed"
    )
    print(f"{topology_label}: {topology_seed}")
    print(f"Optimizer seed: {optimizer_seed}")
    try:
        result = run_local_submission(
            args.archive,
            topology_seed=topology_seed,
            optimizer_seed=optimizer_seed,
            max_time_seconds=args.max_time_seconds,
        )
    except BaseException as exc:
        print(f"\nLOCAL RUN FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    print_local_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

