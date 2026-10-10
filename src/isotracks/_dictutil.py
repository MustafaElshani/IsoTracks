"""Nested-dictionary merge used by the track configuration system."""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping


def _descend(config: MutableMapping[str, Any], keys: List[str]) -> MutableMapping[str, Any]:
    node = config
    for key in keys:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    return node


def update_dict(target: MutableMapping[str, Any], updates: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge *updates* into *target*, in place.

    Keys may be dotted paths to nested values:

        update_dict(cfg, {"exon": {"facecolor": "red"}})
        update_dict(cfg, {"exon.facecolor": "red"})
    """
    for key, value in updates.items():
        path = key.split(".")
        if len(path) > 1:
            node = _descend(target, path[:-1])
            key = path[-1]
        else:
            node = target

        if isinstance(value, dict):
            if value:
                if not isinstance(node.get(key), dict):
                    node[key] = {}
                update_dict(node[key], value)
            else:
                node[key] = {}
        elif isinstance(value, list):
            node[key] = list(value)
        else:
            node[key] = value
    return target


def deep_copy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a config tree deeply enough that nested edits cannot leak upward."""
    out: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out[key] = deep_copy_config(value)
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    return out
