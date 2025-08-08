import os
from typing import Optional

import h5py

from flametrack.analysis import user_config

HDF_FILE: Optional[h5py.File] = None
LOADED_EXP_NAME: Optional[str] = None


def create_h5_file(
    exp_name: Optional[str] = None, filename: Optional[str] = None
) -> h5py.File:
    """
    Create a new HDF5 file.
    Diese Funktion erzeugt KEINE experiment-spezifischen Gruppen.
    Gruppen wie 'dewarped_data' oder 'dewarped_data_left/right' werden
    von den jeweiligen Schreibern (z. B. Dewarping-Funktionen) angelegt
    oder über init_h5_for_experiment() erstellt.
    """
    global HDF_FILE
    global LOADED_EXP_NAME

    if filename is None:
        filename = get_h5_file_path(exp_name)
    foldername = os.path.dirname(filename)
    os.makedirs(foldername, exist_ok=True)

    f = h5py.File(filename, "w")
    f.attrs["file_version"] = "1.0"

    HDF_FILE = f
    LOADED_EXP_NAME = filename
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
    """
    Construct the path to the HDF5 file for the experiment.

    Args:
        exp_name: Experiment name.
        left: If True, use suffix '_left'.

    Returns:
        str: Path to the HDF5 file.
    """
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


def get_file(exp_name: str, mode: str = "r", left: bool = False) -> h5py.File:
    """
    Open or reuse HDF5 file for the experiment.

    Args:
        exp_name: Experiment name.
        mode: File mode ('r' or 'a'), 'w' not supported here.
        left: Use left variant if True.

    Returns:
        h5py.File: Opened HDF5 file handle.
    """
    if mode == "w":
        raise ValueError("Use create_h5_file to create a new file")

    global HDF_FILE
    global LOADED_EXP_NAME

    if HDF_FILE is None:
        filename = get_h5_file_path(exp_name, left=left)
        HDF_FILE = h5py.File(filename, mode)
        LOADED_EXP_NAME = filename

    elif LOADED_EXP_NAME != get_h5_file_path(exp_name, left=left):
        close_file()
        return get_file(exp_name, mode, left=left)

    return HDF_FILE


def close_file() -> None:
    """Close the currently opened HDF5 file, if any."""
    global HDF_FILE
    global LOADED_EXP_NAME

    if HDF_FILE is not None:
        HDF_FILE.close()
        HDF_FILE = None
        LOADED_EXP_NAME = None


def save_edge_results(exp_name: str, edge_results, left: bool = False) -> None:
    """
    Save edge results array to the experiment's HDF5 file.

    Args:
        exp_name: Experiment name.
        edge_results: Numpy array with edge results.
        left: Use left variant if True.
    """
    with get_file(exp_name, "a", left=left) as f:
        grp = f["edge_results"]
        if "data" in grp:
            del grp["data"]
        grp.create_dataset("data", data=edge_results)
    close_file()
