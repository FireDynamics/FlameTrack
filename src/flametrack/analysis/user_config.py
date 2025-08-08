from __future__ import annotations

import configparser
import os
from configparser import ConfigParser

# Standardpfad: im selben Ordner wie dieses Modul
_CONFIG_DIR_DEFAULT = os.path.dirname(__file__)
_CONFIG_PATH_DEFAULT = os.path.join(_CONFIG_DIR_DEFAULT, "config.ini")

# Override per Umgebungsvariable möglich (z. B. im CI oder in Tests)
CONFIG_PATH = os.getenv("FLAMETRACK_CONFIG", _CONFIG_PATH_DEFAULT)

# Aktiver Konfigurationsmodus
config_mode: str = "TESTING"


def __get_default_values() -> dict[str, str]:
    return {
        "experiment_folder": ".",
        "IR_folder": "exported_data/",
        "processed_data": "processed_data/",
    }


def __create_missing_config() -> None:
    """Erstellt die Config-Datei, falls sie fehlt."""
    cfg_dir = os.path.dirname(CONFIG_PATH)
    os.makedirs(cfg_dir, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        config = configparser.ConfigParser()
        config["DEFAULT"] = __get_default_values()
        with open(CONFIG_PATH, "w") as f:
            config.write(f)


def __get_value(name: str, config: ConfigParser, section: str = "DEFAULT") -> str:
    """Liest einen Wert aus einer Section; fallback auf DEFAULT."""
    if section not in config:
        raise KeyError(f"Config section '{section}' not found")
    value = config[section].get(name) or config["DEFAULT"].get(name)
    if value is None:
        raise KeyError(f"No value for '{name}' found in section '{section}' or DEFAULT")
    return value


def get_experiments() -> list[str]:
    config = get_config()
    exp_folder = __get_value("experiment_folder", config, config_mode)
    try:
        folders = os.listdir(exp_folder)
    except FileNotFoundError:
        return []
    return sorted(
        [
            folder
            for folder in folders
            if os.path.isdir(os.path.join(exp_folder, folder))
        ]
    )


def get_config() -> ConfigParser:
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config


def get_IR_path(exp_name: str) -> str:
    return get_path(exp_name, "IR_folder")


def get_path(exp_name: str, path_name: str) -> str:
    config = get_config()
    exp_folder = __get_value("experiment_folder", config, config_mode)
    path = __get_value(path_name, config, config_mode)
    return os.path.join(exp_folder, exp_name, path)


# Stelle sicher, dass eine Basis-Konfiguration vorhanden ist
__create_missing_config()
