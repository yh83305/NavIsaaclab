# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lightweight YAML-subset loader for CrowdSim config files.

Extracted from ``crowd_sim.py`` and ``train_ppo.py`` to avoid duplication.
Supports a strict subset of YAML: nested dicts, scalars (bool/int/float/str/None),
inline lists, inline comments, and nested ``_include`` mappings.  Does not
require PyYAML.

Public API
----------
load_config(path)         -- load a YAML file → dict
cfg_path(path_like, root) -- resolve a (possibly relative) path against *root*
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Uses PyYAML when available; falls back to the built-in subset parser. A
    mapping containing ``_include: relative/path.yaml`` is replaced by the
    referenced mapping plus any local overrides.
    """
    return _load_config_with_includes(Path(path).expanduser().resolve(), ())


def _load_config_with_includes(
    path: Path,
    active_paths: tuple[Path, ...],
) -> dict[str, Any]:
    if path in active_paths:
        chain = " -> ".join(str(value) for value in (*active_paths, path))
        raise ValueError(f"Config include cycle: {chain}")
    data = _load_config_data(path)
    if not isinstance(data, dict):
        return {}
    return _resolve_config_includes(data, path, (*active_paths, path))


def _load_config_data(path: Path) -> dict[str, Any]:
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except ImportError:
        return _load_config_fallback(path)


def _resolve_config_includes(
    value: Any,
    source_path: Path,
    active_paths: tuple[Path, ...],
) -> Any:
    if isinstance(value, list):
        return [
            _resolve_config_includes(item, source_path, active_paths)
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    merged: dict[str, Any] = {}
    include_value = value.get("_include")
    if include_value is not None:
        if not isinstance(include_value, str) or not include_value:
            raise ValueError(f"Config _include must be a path string in {source_path}")
        include_path = Path(include_value).expanduser()
        if not include_path.is_absolute():
            include_path = source_path.parent / include_path
        merged.update(
            _load_config_with_includes(include_path.resolve(), active_paths)
        )

    for key, item in value.items():
        if key == "_include":
            continue
        resolved_item = _resolve_config_includes(item, source_path, active_paths)
        if isinstance(merged.get(key), dict) and isinstance(resolved_item, dict):
            merged[key] = _deep_merge_dicts(merged[key], resolved_item)
        else:
            merged[key] = resolved_item
    return merged


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply config overrides without dropping sibling settings."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _load_config_fallback(path: Path) -> dict[str, Any]:
    """Fallback parser for the small CrowdSim YAML subset (no PyYAML required).

    Supports:
    - Nested dicts via indentation
    - Scalars: bool, int, float, str, None/null
    - Inline lists: ``[a, b, c]`` or ``a, b, c``
    - Inline comments: ``key: value  # comment``
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"Invalid config line {line_number}: {raw_line}")

        key, raw_value = stripped.split(":", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation at line {line_number}: {raw_line}")

        parent = stack[-1][1]
        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(raw_value)

    return root


def parse_scalar(value: str):
    """Infer the Python type of a YAML scalar string."""
    text = strip_inline_comment(value).strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def strip_inline_comment(value: str) -> str:
    """Strip ``# ...`` comments that appear outside of quoted strings."""
    quote: str | None = None
    for idx, char in enumerate(value):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None:
            return value[:idx]
    return value


def cfg_path(path_like: str, project_root: Path) -> Path:
    """Resolve *path_like* to an absolute path.

    If *path_like* is already absolute it is returned as-is; otherwise it is
    resolved relative to *project_root*.
    """
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()
