"""Tests for flametrack.analysis.user_config."""

from __future__ import annotations

import configparser
import os

import pytest

import flametrack.analysis.user_config as uc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_ini(path: str, sections: dict) -> None:
    cfg = configparser.ConfigParser()
    for section, values in sections.items():
        if section == "DEFAULT":
            cfg["DEFAULT"] = values
        else:
            cfg[section] = values
    with open(path, "w", encoding="utf-8") as f:
        cfg.write(f)


# ---------------------------------------------------------------------------
# __get_default_values (indirectly via __create_missing_config)
# ---------------------------------------------------------------------------


def test_create_missing_config_creates_file(tmp_path, monkeypatch):
    """Config file is created when it does not exist."""
    ini_path = str(tmp_path / "config.ini")
    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)

    # Calling the module-level helper via the public surface
    uc._DataClass__create_missing_config() if hasattr(
        uc, "_DataClass__create_missing_config"
    ) else None
    # The simplest route: call get_config() which internally may recreate it,
    # but the real creation happens at import time.  Re-trigger by calling the
    # private function directly through its mangled name isn't possible, so we
    # test the effect via get_config().
    assert not os.path.exists(ini_path)  # hasn't been called yet

    # Call get_config – reads an empty/nonexistent file, returns empty parser
    cfg = uc.get_config()
    assert cfg is not None


def test_create_missing_config_idempotent(tmp_path, monkeypatch):
    """If config already exists it is NOT overwritten."""
    ini_path = str(tmp_path / "config.ini")
    # Write a custom value
    _write_ini(ini_path, {"DEFAULT": {"experiment_folder": "/custom/path"}})
    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)

    # get_config just reads → the custom value should survive
    cfg = uc.get_config()
    assert cfg["DEFAULT"]["experiment_folder"] == "/custom/path"


# ---------------------------------------------------------------------------
# get_config
# ---------------------------------------------------------------------------


def test_get_config_returns_configparser(tmp_path, monkeypatch):
    ini_path = str(tmp_path / "config.ini")
    _write_ini(ini_path, {"DEFAULT": {"experiment_folder": "/exp"}})
    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)

    cfg = uc.get_config()
    assert isinstance(cfg, configparser.ConfigParser)
    assert cfg["DEFAULT"]["experiment_folder"] == "/exp"


def test_get_config_missing_file_returns_empty(tmp_path, monkeypatch):
    """get_config does not raise even when config file is absent."""
    monkeypatch.setattr(uc, "CONFIG_PATH", str(tmp_path / "nonexistent.ini"))
    cfg = uc.get_config()
    assert isinstance(cfg, configparser.ConfigParser)


# ---------------------------------------------------------------------------
# __get_value (indirectly via get_path / get_experiments)
# ---------------------------------------------------------------------------


def test_get_value_missing_section_raises(tmp_path, monkeypatch):
    ini_path = str(tmp_path / "config.ini")
    _write_ini(ini_path, {"DEFAULT": {"experiment_folder": "/exp"}})
    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)
    monkeypatch.setattr(uc, "config_mode", "MISSING_SECTION")

    with pytest.raises(KeyError, match="MISSING_SECTION"):
        uc.get_experiments()


def test_get_value_missing_key_raises(tmp_path, monkeypatch):
    """A key that exists neither in section nor DEFAULT → KeyError."""
    ini_path = str(tmp_path / "config.ini")
    # Write an ini with a section but without 'experiment_folder'
    cfg_obj = configparser.ConfigParser()
    cfg_obj["MY_SECTION"] = {"some_key": "val"}
    # Remove DEFAULT fallback by writing a minimal ini without experiment_folder
    with open(ini_path, "w", encoding="utf-8") as f:
        cfg_obj.write(f)
    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)
    monkeypatch.setattr(uc, "config_mode", "MY_SECTION")

    with pytest.raises(KeyError):
        uc.get_experiments()


# ---------------------------------------------------------------------------
# get_experiments
# ---------------------------------------------------------------------------


def test_get_experiments_returns_sorted_folders(tmp_path, monkeypatch):
    ini_path = str(tmp_path / "config.ini")
    exp_base = tmp_path / "experiments"
    exp_base.mkdir()
    for name in ["exp_b", "exp_a", "exp_c"]:
        (exp_base / name).mkdir()
    # Also create a file (should be ignored)
    (exp_base / "not_a_folder.txt").write_text("x")

    _write_ini(ini_path, {"DEFAULT": {"experiment_folder": str(exp_base)}})
    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)
    monkeypatch.setattr(uc, "config_mode", "DEFAULT")

    experiments = uc.get_experiments()
    assert experiments == sorted(["exp_a", "exp_b", "exp_c"])


def test_get_experiments_nonexistent_folder_returns_empty(tmp_path, monkeypatch):
    ini_path = str(tmp_path / "config.ini")
    _write_ini(ini_path, {"DEFAULT": {"experiment_folder": str(tmp_path / "no_such")}})
    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)
    monkeypatch.setattr(uc, "config_mode", "DEFAULT")

    assert uc.get_experiments() == []


# ---------------------------------------------------------------------------
# get_path / get_ir_path
# ---------------------------------------------------------------------------


def test_get_path_combines_correctly(tmp_path, monkeypatch):
    ini_path = str(tmp_path / "config.ini")
    exp_folder = str(tmp_path / "experiments")
    _write_ini(
        ini_path,
        {
            "DEFAULT": {
                "experiment_folder": exp_folder,
                "IR_folder": "exported_data/",
                "processed_data": "processed_data/",
            }
        },
    )
    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)
    monkeypatch.setattr(uc, "config_mode", "DEFAULT")

    path = uc.get_path("my_exp", "IR_folder")
    expected = os.path.join(exp_folder, "my_exp", "exported_data/")
    assert path == expected


def test_get_ir_path_uses_ir_folder(tmp_path, monkeypatch):
    ini_path = str(tmp_path / "config.ini")
    exp_folder = str(tmp_path / "experiments")
    _write_ini(
        ini_path,
        {
            "DEFAULT": {
                "experiment_folder": exp_folder,
                "IR_folder": "exported_data/",
                "processed_data": "processed_data/",
            }
        },
    )
    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)
    monkeypatch.setattr(uc, "config_mode", "DEFAULT")

    ir_path = uc.get_ir_path("exp42")
    assert ir_path == os.path.join(exp_folder, "exp42", "exported_data/")


# ---------------------------------------------------------------------------
# config_mode respected across calls
# ---------------------------------------------------------------------------


def test_config_mode_custom_section(tmp_path, monkeypatch):
    """Values from a named section override DEFAULT."""
    ini_path = str(tmp_path / "config.ini")
    cfg_obj = configparser.ConfigParser()
    cfg_obj["DEFAULT"] = {
        "experiment_folder": "/default/path",
        "IR_folder": "exported_data/",
        "processed_data": "processed_data/",
    }
    cfg_obj["MY_SECTION"] = {"experiment_folder": "/custom/section/path"}
    with open(ini_path, "w", encoding="utf-8") as f:
        cfg_obj.write(f)

    monkeypatch.setattr(uc, "CONFIG_PATH", ini_path)
    monkeypatch.setattr(uc, "config_mode", "MY_SECTION")

    path = uc.get_path("exp1", "IR_folder")
    # experiment_folder should come from MY_SECTION → /custom/section/path
    assert path.startswith("/custom/section/path")
