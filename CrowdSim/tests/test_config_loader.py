from __future__ import annotations

import pytest

from CrowdSim.utils.config_loader import load_config


def test_nested_config_include_resolves_relative_path_and_local_override(
    tmp_path,
) -> None:
    common = tmp_path / "common.yaml"
    common.write_text("first: 1\nsecond: 2\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "section:\n"
        "  _include: common.yaml\n"
        "  second: 20\n"
        "  third: 3\n",
        encoding="utf-8",
    )
    assert load_config(config)["section"] == {
        "first": 1,
        "second": 20,
        "third": 3,
    }


def test_config_include_cycle_is_rejected(tmp_path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("_include: second.yaml\n", encoding="utf-8")
    second.write_text("_include: first.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="include cycle"):
        load_config(first)
