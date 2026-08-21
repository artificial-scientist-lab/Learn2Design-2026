"""Small helpers for reading the UIFO HDF5 dataset lazily."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset.h5"
UIFO_OPTIMIZED_PROPERTIES = (
    "reflectivity",
    "tuning",
    "db",
    "angle",
    "power",
    "mass",
    "length",
)


def decode_h5_string(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def find_entry_index(handle: h5py.File, *, index: int | None, unique_hash: str | None) -> int:
    if (index is None) == (unique_hash is None):
        raise ValueError("Pass exactly one of index or unique_hash.")

    entry_count = handle["entries"].shape[0]
    if index is not None:
        if index < 0 or index >= entry_count:
            raise IndexError(f"index must be in [0, {entry_count}), got {index}")
        return index

    assert unique_hash is not None
    target = unique_hash.encode("utf-8")
    hashes = handle["entries"]["unique_hash"][:]
    matches = np.flatnonzero(hashes == target)
    if len(matches) == 0:
        raise KeyError(f"unique_hash not found: {unique_hash}")
    return int(matches[0])


def entry_to_dict(entry: np.void) -> dict[str, Any]:
    return {
        "unique_hash": decode_h5_string(entry["unique_hash"]),
        "initialized_from": decode_h5_string(entry["initialized_from"]),
        "topology_string": decode_h5_string(entry["topology_string"]),
        "size": int(entry["size"]),
        "loss": float(entry["loss"]),
        "complexity": int(entry["complexity"]),
        "source_file": decode_h5_string(entry["source_file"]),
        "run_id": decode_h5_string(entry["run_id"]),
        "param_offset": int(entry["param_offset"]),
        "param_length": int(entry["param_length"]),
        "power_offset": int(entry["power_offset"]),
        "power_length": int(entry["power_length"]),
        "power_rows": int(entry["power_rows"]),
        "power_cols": int(entry["power_cols"]),
        "power_map_id": int(entry["power_map_id"]),
    }


def load_bounded_params(handle: h5py.File, entry: np.void) -> np.ndarray:
    start = int(entry["param_offset"])
    stop = start + int(entry["param_length"])
    return handle["bounded_params"][start:stop]


def load_power_data(handle: h5py.File, entry: np.void) -> tuple[np.ndarray, list[str], np.ndarray]:
    start = int(entry["power_offset"])
    stop = start + int(entry["power_length"])
    rows = int(entry["power_rows"])
    cols = int(entry["power_cols"])
    values = handle["power_values"][start:stop].reshape(rows, cols)

    map_id = int(entry["power_map_id"])
    map_start = int(handle["power_port_map_offsets"][map_id])
    map_stop = int(handle["power_port_map_offsets"][map_id + 1])
    names = [decode_h5_string(value) for value in handle["power_port_map_names"][map_start:map_stop]]
    indices = handle["power_port_map_indices"][map_start:map_stop]
    return values, names, indices


def reconstruct_uifo_setup(topology_string: str, size: int, *, mode: str = "space_modulation"):
    """Rebuild a dataset UIFO and its parameter-index mapping without the full objective."""
    from dfbench.problems.uifo import topology_from_string
    from differometor.setups import constrain_inter_grid_cell_spaces, uifo

    centers, boundaries = topology_from_string(topology_string, size)
    # Dataset topologies specify every cell, so random=True matches UIFOProblem
    # construction without changing the decoded centers or boundaries.
    setup, _ = uifo(
        size=size,
        mode=mode,
        random=True,
        centers=centers,
        boundaries=boundaries,
    )
    optimization_pairs = constrain_inter_grid_cell_spaces(
        setup.parameters,
        UIFO_OPTIMIZED_PROPERTIES,
    )
    return setup, optimization_pairs
