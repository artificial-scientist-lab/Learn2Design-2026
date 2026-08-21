#!/usr/bin/env python3
"""Load one UIFO dataset entry and print its saved arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from dataset_utils import (
    DATASET_PATH,
    entry_to_dict,
    find_entry_index,
    load_bounded_params,
    load_power_data,
    reconstruct_uifo_setup,
)


def format_parameter_target(optimization_pair) -> str:
    if optimization_pair and isinstance(optimization_pair[0], (list, tuple)):
        targets = optimization_pair
    else:
        targets = [optimization_pair]

    labels = [f"{component_name}.{property_name}" for component_name, property_name in targets]
    if len(labels) == 1:
        return labels[0]
    return f"[{', '.join(labels)}]"


def print_parameter_map(params, metadata) -> None:
    try:
        _, optimization_pairs = reconstruct_uifo_setup(
            metadata["topology_string"],
            metadata["size"],
        )
    except ModuleNotFoundError as exc:
        raise SystemExit("Install the competition package first: pip install -e .") from exc
    if len(params) != len(optimization_pairs):
        raise ValueError(
            f"Saved params have length {len(params)}, but the reconstructed setup expects "
            f"{len(optimization_pairs)} optimization parameters."
        )

    print("parameter_map:")
    for index, (value, optimization_pair) in enumerate(
        zip(params, optimization_pairs, strict=True)
    ):
        target = format_parameter_target(optimization_pair)
        print(f"  [{index:03d}] {target} = {float(value):.12g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--index", type=int, default=0, help="Entry index to load.")
    parser.add_argument("--hash", dest="unique_hash", help="Entry unique_hash. Overrides --index.")
    parser.add_argument(
        "--show-parameter-map",
        action="store_true",
        help="Print every bounded parameter index with its component/property target.",
    )
    args = parser.parse_args()

    index_arg = None if args.unique_hash else args.index

    with h5py.File(args.dataset, "r") as handle:
        entry_index = find_entry_index(handle, index=index_arg, unique_hash=args.unique_hash)
        entry = handle["entries"][entry_index]
        metadata = entry_to_dict(entry)
        params = load_bounded_params(handle, entry)
        sensitivities = handle["sensitivities"][entry_index]
        frequencies = handle["frequency_values"][:]
        power, power_port_names, power_port_indices = load_power_data(handle, entry)
        named_power = power[power_port_indices]

    print(f"entry_index: {entry_index}")
    for key in (
        "unique_hash",
        "initialized_from",
        "topology_string",
        "size",
        "loss",
        "complexity",
        "source_file",
        "run_id",
    ):
        print(f"{key}: {metadata[key]}")

    print(f"bounded_params shape: {params.shape}")
    print(f"bounded_params first 5: {params[:5]}")
    print(f"frequency_values shape: {frequencies.shape}")
    print(f"frequency_values first 5: {frequencies[:5]}")
    print(f"sensitivities shape: {sensitivities.shape}")
    print(f"sensitivities first 5: {sensitivities[:5]}")
    print(f"power shape: {power.shape}")
    print(f"power first row first 5: {power[0, :5]}")
    print(f"power port count: {len(power_port_names)}")
    print(f"power port labels first 5: {power_port_names[:5]}")
    print(f"power port indices first 5: {power_port_indices[:5]}")
    print(f"named power first 5 rows: {named_power[:5].ravel()}")
    if args.show_parameter_map:
        print_parameter_map(params, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
