from __future__ import annotations

import os
from typing import Any, Literal, Optional

import h5py
import numpy as np
from numpy.typing import NDArray

from flametrack.analysis import user_config

HDF_FILE: Optional[h5py.File] = None
LOADED_FILE_PATH: Optional[str] = None


def create_h5_file(
    exp_name: Optional[str] = None, filename: Optional[str] = None
) -> h5py.File:
    """
    Create a new HDF5 file.
    Diese Funktion erzeugt KEINE experiment-spezifischen Gruppen.
    """
    global HDF_FILE, LOADED_FILE_PATH

    if filename is None:
        # -> filename soll aus exp_name abgeleitet werden: exp_name MUSS dann gesetzt sein.
        if exp_name is None:
            raise ValueError("exp_name must be provided when filename is None")
        filename = get_h5_file_path(exp_name)

    foldername = os.path.dirname(filename)
    os.makedirs(foldername, exist_ok=True)

    f = h5py.File(filename, "w")
    f.attrs["file_version"] = "1.0"

    HDF_FILE = f
    LOADED_FILE_PATH = filename
    return f


def init_h5_for_experiment(h5: h5py.File, experiment_type: str) -> None:
    """
    Lege die nötigen Gruppen je Experimenttyp an.
    Kann z. B. direkt nach create_h5_file(...) aufgerufen werden.
    """
    if experiment_type == "Lateral Flame Spread":
        h5.require_group("dewarped_data")
        h5.require_group("edge_results")
    elif experiment_type == "Room Corner":
        h5.require_group("dewarped_data_left")
        h5.require_group("dewarped_data_right")
        # edge_results_* werden später beim Schreiben/Erkennen angelegt
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def assert_h5_schema(h5: h5py.File, experiment_type: str) -> None:
    """
    Verifiziere, dass die nötigen Gruppen existieren.
    Praktisch für frühe, klare Fehlermeldungen.
    """
    required = {
        "Lateral Flame Spread": ["dewarped_data"],
        "Room Corner": ["dewarped_data_left", "dewarped_data_right"],
    }[experiment_type]
    missing = [g for g in required if g not in h5]
    if missing:
        raise RuntimeError(f"Missing groups for {experiment_type}: {missing}")


def get_h5_file_path(exp_name: str, left: bool = False) -> str:
    left_str = "_left" if left else ""
    return os.path.join(
        user_config.get_path(exp_name, "processed_data"),
        f"{exp_name}_results{left_str}.h5",
    )


def get_data(exp_name: str, group_name: str, left: bool = False) -> h5py.Dataset:
    """
    Retrieve dataset for given experiment and group.

    Args:
        exp_name: Experiment name.
        group_name: Group name in HDF5 file ('dewarped_data' or 'edge_results').
        left: Use left variant if True.

    Returns:
        h5py.Dataset: Dataset object.
    """
    f = get_file(exp_name, left=left)
    data = f[group_name]["data"]
    return data


def get_edge_results(exp_name: str, left: bool = False) -> h5py.Dataset:
    """Get edge results dataset from the experiment file."""
    return get_data(exp_name, "edge_results", left)


def get_dewarped_data(exp_name: str, left: bool = False) -> h5py.Dataset:
    """Get dewarped data dataset from the experiment file."""
    return get_data(exp_name, "dewarped_data", left)


def get_dewarped_metadata(exp_name: str, left: bool = False) -> dict:
    """Get metadata attributes from dewarped_data group."""
    f = get_file(exp_name, left=left)
    return dict(f["dewarped_data"].attrs)


def get_file(
    exp_name: str, mode: Literal["r", "a"] = "r", left: bool = False
) -> h5py.File:
    """
    Open or reuse HDF5 file for the experiment.
    """
    if mode == "w":
        raise ValueError("Use create_h5_file to create a new file")

    global HDF_FILE, LOADED_FILE_PATH

    wanted_path = get_h5_file_path(exp_name, left=left)

    need_reopen = (
        HDF_FILE is None
        or LOADED_FILE_PATH != wanted_path
        or (
            hasattr(HDF_FILE, "id") and not HDF_FILE.id.valid
        )  # falls bereits geschlossen
    )

    if need_reopen:
        if HDF_FILE is not None and hasattr(HDF_FILE, "id") and HDF_FILE.id.valid:
            HDF_FILE.close()
        HDF_FILE = h5py.File(wanted_path, mode)
        LOADED_FILE_PATH = wanted_path

    return HDF_FILE


def close_file() -> None:
    """Close the currently opened HDF5 file, if any."""
    global HDF_FILE
    global LOADED_FILE_PATH

    if HDF_FILE is not None:
        HDF_FILE.close()
        HDF_FILE = None
        LOADED_FILE_PATH = None


def save_edge_results(
    exp_name: str,
    edge_results: NDArray[np.floating] | NDArray[np.integer],
    left: bool = False,
) -> None:
    """
    Save edge results array to the experiment's HDF5 file.

    Opens a fresh file handle just for this write (no global handle usage).
    """
    filename = get_h5_file_path(exp_name, left=left)
    # Ephemeres Handle → kein Einfluss auf globale HDF_FILE/LOADED_FILE_PATH
    with h5py.File(filename, "a") as f:
        grp = f.require_group("edge_results")
        if "data" in grp:
            del grp["data"]
        grp.create_dataset("data", data=edge_results)
