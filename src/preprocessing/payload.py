"""Helpers for reading nested connector payloads from Ray batches."""

from typing import Any


def _nested_get(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def payload_values(batch: dict[str, list], key: str, default: Any = "") -> list:
    """Read a payload field from either flattened or nested batch columns."""
    flat_key = f"payload.{key}"
    if flat_key in batch:
        return list(batch[flat_key])

    if "payload" not in batch:
        available = ", ".join(sorted(batch.keys()))
        raise KeyError(f"Missing payload.{key}; available columns: {available}")

    path = key.split(".")
    return [_nested_get(payload, path, default) for payload in batch["payload"]]
