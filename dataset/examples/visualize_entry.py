#!/usr/bin/env python3
"""Visualize one UIFO dataset entry with Differometor."""

from __future__ import annotations

import argparse
import inspect
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

try:
    from differometor import visualize_setup
except ModuleNotFoundError as exc:
    raise SystemExit("Install the competition package first: pip install -e .") from exc

from dataset_utils import (
    DATASET_PATH,
    entry_to_dict,
    find_entry_index,
    load_bounded_params,
    load_power_data,
    reconstruct_uifo_setup,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "visualizations"
DEFAULT_MODE = "space_modulation"
DEFAULT_LAYOUT_ORIENTATION = "topology"
LAYOUT_ORIENTATIONS = ("topology", "detector-right", "native")
_CARDINAL_DIRECTIONS = ("left", "top", "right", "bottom")


def _rotate_quarter_turn(vector, steps):
    x, y = np.asarray(vector, dtype=float)
    return (
        np.array([x, y]),
        np.array([-y, x]),
        np.array([-x, -y]),
        np.array([y, -x]),
    )[int(steps) % 4]


def _rotate_layout_state_compat(positions, left_dirs, visible_nodes, steps):
    steps = int(steps) % 4
    visible_positions = [positions[name] for name in visible_nodes if name in positions]
    if steps == 0 or not visible_positions:
        return

    center = np.mean(np.vstack(visible_positions), axis=0)
    for name in positions:
        positions[name] = _rotate_quarter_turn(positions[name] - center, steps) + center

    for name, direction in list(left_dirs.items()):
        if direction in _CARDINAL_DIRECTIONS:
            direction_index = _CARDINAL_DIRECTIONS.index(direction)
            left_dirs[name] = _CARDINAL_DIRECTIONS[(direction_index - steps) % 4]


def _orient_layout_like_topology_compat(positions, left_dirs, _node_components, nodes_to_skip):
    grid_nodes = []
    for node_name, position in positions.items():
        coordinate = str(node_name).removeprefix("center")
        if node_name in nodes_to_skip or len(coordinate) != 2 or not coordinate.isdigit():
            continue
        row, column = (int(value) for value in coordinate)
        grid_nodes.append((row, column, np.asarray(position, dtype=float)))

    rows = {row for row, _, _ in grid_nodes}
    columns = {column for _, column, _ in grid_nodes}
    if len(rows) < 2 or len(columns) < 2:
        raise ValueError("Topology orientation requires UIFO nodes named like 'center11'.")

    source_points = np.vstack([position for _, _, position in grid_nodes])
    source_points -= np.mean(source_points, axis=0)
    target_points = np.asarray([[column, -row] for row, column, _ in grid_nodes], dtype=float)
    target_points -= np.mean(target_points, axis=0)
    normalization = float(np.linalg.norm(source_points) * np.linalg.norm(target_points))
    if normalization == 0:
        raise ValueError("Cannot infer topology orientation from coincident grid centers.")

    scores = []
    for steps in range(4):
        rotated = np.vstack([_rotate_quarter_turn(point, steps) for point in source_points])
        scores.append(float(np.sum(rotated * target_points) / normalization))

    visible_nodes = [name for name in positions if name not in nodes_to_skip]
    _rotate_layout_state_compat(positions, left_dirs, visible_nodes, int(np.argmax(scores)))


@contextmanager
def differometor_orientation_options(layout_orientation: str):
    """Return public 0.0.6 options or temporarily backport them to 0.0.5."""
    if "layout_orientation" in inspect.signature(visualize_setup).parameters:
        yield {"layout_orientation": layout_orientation}
        return

    import differometor.plot as differometor_plot

    original_orient = differometor_plot._orient_layout_detector_on_right
    original_rotate = differometor_plot._rotate_layout_state
    differometor_plot._rotate_layout_state = _rotate_layout_state_compat
    if layout_orientation == "native":
        differometor_plot._orient_layout_detector_on_right = lambda *_args: None
    elif layout_orientation == "topology":
        differometor_plot._orient_layout_detector_on_right = _orient_layout_like_topology_compat

    try:
        yield {}
    finally:
        differometor_plot._orient_layout_detector_on_right = original_orient
        differometor_plot._rotate_layout_state = original_rotate


def resolve_output_path(output: Path | None, *, entry_index: int, unique_hash: str) -> Path:
    if output is None:
        output = DEFAULT_OUTPUT_DIR / f"entry_{entry_index:06d}_{unique_hash[:8]}.html"
    else:
        output = output.expanduser()

    if not str(output).lower().endswith(".html"):
        output = Path(f"{output}.html")
    return output


def port_map_to_index(port_names: list[str], port_indices) -> dict[str, int]:
    return {name: int(index) for name, index in zip(port_names, port_indices)}


def iter_parameter_pairs(optimization_pair):
    if optimization_pair and isinstance(optimization_pair[0], list):
        yield from optimization_pair
    else:
        yield optimization_pair


def apply_bounded_params(setup, optimization_pairs, bounded_params) -> None:
    if len(bounded_params) != len(optimization_pairs):
        raise ValueError(
            f"Saved params have length {len(bounded_params)}, but the reconstructed setup expects "
            f"{len(optimization_pairs)} optimization parameters."
        )

    for value, optimization_pair in zip(bounded_params, optimization_pairs):
        for component_name, property_name in iter_parameter_pairs(optimization_pair):
            if "_" in component_name:
                setup.edges[component_name]["properties"][property_name] = float(value)
            else:
                setup.nodes[component_name]["properties"][property_name] = float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--index", type=int, default=0, help="Entry index to visualize.")
    parser.add_argument("--hash", dest="unique_hash", help="Entry unique_hash. Overrides --index.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML file. Defaults to dataset/examples/visualizations/.",
    )
    parser.add_argument(
        "--layout-orientation",
        choices=LAYOUT_ORIENTATIONS,
        default=DEFAULT_LAYOUT_ORIENTATION,
        help=(
            "Final diagram orientation: 'topology' follows topology-string row-major order "
            "(default), 'detector-right' preserves Differometor's automatic orientation, "
            "and 'native' disables final rotation."
        ),
    )
    args = parser.parse_args()

    index_arg = None if args.unique_hash else args.index

    with h5py.File(args.dataset, "r") as handle:
        entry_index = find_entry_index(handle, index=index_arg, unique_hash=args.unique_hash)
        entry = handle["entries"][entry_index]
        metadata = entry_to_dict(entry)
        bounded_params = load_bounded_params(handle, entry)
        power, power_port_names, power_port_indices = load_power_data(handle, entry)

    setup, optimization_pairs = reconstruct_uifo_setup(
        metadata["topology_string"],
        metadata["size"],
        mode=DEFAULT_MODE,
    )
    apply_bounded_params(setup, optimization_pairs, bounded_params)

    output_path = resolve_output_path(
        args.output,
        entry_index=entry_index,
        unique_hash=metadata["unique_hash"],
    )
    layout_orientation = args.layout_orientation.replace("-", "_")
    with differometor_orientation_options(layout_orientation) as orientation_options:
        visualize_setup(
            setup,
            output_file=output_path,
            port_to_index=port_map_to_index(power_port_names, power_port_indices),
            powers=power,
            **orientation_options,
        )

    print(f"entry_index: {entry_index}")
    print(f"unique_hash: {metadata['unique_hash']}")
    print(f"topology_string: {metadata['topology_string']}")
    print(f"size: {metadata['size']}")
    print(f"layout_orientation: {args.layout_orientation}")
    print(f"output: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
